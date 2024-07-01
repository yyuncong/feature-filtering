from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import os
import torch
import json
from PIL import Image
import tqdm
from tqdm import tqdm
import cv2
from inference.models import YOLOWorld
from argparse import ArgumentParser
from collections import defaultdict
parser = ArgumentParser()
parser.add_argument('--l', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

class LlaVa_Encoder():
    def __init__(self):
        model_path = "/work/pi_chuangg_umass_edu/yuncong/llava-v1.5-7b"
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, llava_model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, add_multisensory_token=False)
        self.image_processor = image_processor
        llava_model = llava_model.float()
        llava_model = llava_model.eval()
        self.model = llava_model

    def encode(self, img):
        img = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values']
        img = torch.cat([img], dim=0).float()
        img = self.model.encode_images(img)

        return img
    

if __name__ == "__main__":

    scene_feature_path = "..."
    gt_seg_path = "..."
    new_gt_seg_path = "..."
    bbox_path = "hm3d_obj_bbox_merged"

    if not os.path.exists(scene_feature_path):
        os.mkdir(scene_feature_path)
    if not os.path.exists(new_gt_seg_path):
        os.mkdir(new_gt_seg_path)

    detection_model = YOLOWorld(model_id="yolo_world/x")

    llava_encoder = LlaVa_Encoder()
    if not os.path.exists(scene_feature_path):
        os.mkdir(scene_feature_path)
    
    print(len(os.listdir(gt_seg_path)))
    print(args.l)

    subset_length = 20
    for folder in tqdm(os.listdir(gt_seg_path)[args.l*subset_length: (args.l + 1)*subset_length]):
        if not os.path.isdir(os.path.join(gt_seg_path, folder)):
            continue
        scene_id = folder
        try:
            bboxes = json.load(open(os.path.join(bbox_path, folder+".json")))
        except:
            continue
        id_to_name = {bbox["id"]: bbox["class_name"] for bbox in bboxes}
        if not os.path.exists("%s/%s"%(scene_feature_path, scene_id)):
            os.mkdir("%s/%s"%(scene_feature_path, scene_id))

        if not os.path.exists("%s/%s"%(new_gt_seg_path, folder)):
            os.mkdir("%s/%s"%(new_gt_seg_path, folder))
        # if len(os.listdir("../../feature_dict/%s"%folder)): continue
        
        id_feature_dict = defaultdict(list)
        for file in os.listdir("%s/%s"%(gt_seg_path, folder)):
            if not "cropped" in file: continue
            if file.endswith(".pt"): continue
            id2 = file.split("_")[0]
            if id2 not in id_to_name.keys(): continue
            class_name = id_to_name[id2]
            img = Image.open(os.path.join("%s/%s"%(gt_seg_path, folder), file))

            # if img.size[0] * img.size[1] < 2000:
            #     continue
            
            uncropped_file = file.replace("_cropped", "")
            rgb = cv2.imread(os.path.join("%s/%s"%(gt_seg_path, folder), uncropped_file))
            detection_model.set_classes([class_name])
            # results = detection_model.infer(rgb, confidence=0.1)
            # if results.predictions is None or len(results.predictions) == 0:
            #     continue
            
            feature = llava_encoder.encode(img).mean(1)
            torch.save(feature, "%s/%s/%s.pt"%(new_gt_seg_path, folder, file.replace(".png", ".pt")))
            id_feature_dict[id2].append(feature)
            # img.save("../../original_2d_gt_seg_new/%s/%s"%(folder, file.replace(id2, class_name)))

        for id2, feature_list in id_feature_dict.items():
            feature_list = torch.cat(feature_list)
            feature = feature_list.mean(0)

            torch.save(feature, "%s/%s/%s.pt"%(scene_feature_path, scene_id, id2))
