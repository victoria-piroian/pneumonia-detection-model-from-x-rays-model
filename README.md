# Automated Pneumonia Detection from Chest X-Rays

## Title & Description
This project explores deep learning approaches for **automated pneumonia detection and localization** in chest X-ray (CXR) images using the RSNA Pneumonia Detection Challenge dataset.

We compare:
- **Image Classification Models** (ResNet50, DenseNet121) to detect pneumonia presence
- **Object Detection Models** (Faster R-CNN, YOLOv5n) to localize pneumonia regions
- A **Two-Stage Pipeline** combining classification + detection

The goal is to evaluate trade-offs between **accuracy, recall, and localization performance** in medical imaging workflows, where missing a positive case can have serious clinical consequences.

**Key Finding:**  
Two-stage detection (Faster R-CNN) outperformed faster single-stage models, while adding a classification filter *reduced* performance due to lost recall — highlighting the importance of sensitivity in medical AI systems.

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (recommended)
- PyTorch
- Kaggle account (for dataset access)

### Clone Repository
```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

## Install Dependencies
```
pip install -r requirements.txt
```

If installing manually, required libraries include:
```
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib scikit-learn
pip install albumentations pydicom pillow opencv-python
pip install ultralytics
```

## Dataset

This project uses the RSNA Pneumonia Detection Challenge dataset:
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge

## Download Instructions
```
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip
```

### Place files inside:
```
data/
 ├── stage_2_train_images/
 ├── stage_2_test_images/
 └── stage_2_train_labels.csv
```

## Usage
1. Preprocess DICOM Images

Converts raw medical scans into model-ready tensors and applies augmentation.
```
jupyter notebook Preprocessing.ipynb
```

This step:
- Loads DICOM files using pydicom
- Normalizes pixel values
- Converts grayscale → 3-channel images
- Resizes images (224×224 for classification, 512×512 for detection)
- Applies Albumentations transformations
- Exports processed datasets + bounding boxes

2. Train Classification Models

Binary classification: Pneumonia vs. Normal
```
jupyter notebook ClassificationModels.ipynb
```

Models evaluated:
- ResNet50 (transfer learning)
- DenseNet121 (feature reuse architecture)

Outputs:
- Accuracy, Precision, Recall, F1
- Best checkpoint saved automatically

3. Train Detection Models
Faster R-CNN (Two-Stage Detector)
```
jupyter notebook Faster-RCNN.ipynb
```

- ResNet50 + FPN backbone
- Region Proposal Network for precise localization
- Optimized for subtle medical anomalies

YOLOv5n (Single-Stage Detector)
```
jupyter notebook YOLO.ipynb
```

- Lightweight nano architecture (<2M parameters)
- Fast inference baseline

4. Train / Validation Split
```
jupyter notebook train_val_splitting.ipynb
```

- Stratified 80/20 split
- Maintains class imbalance distribution

5. Evaluate Models
```
jupyter notebook evaluate.ipynb
```

Generates:
- Confusion matrix
- ROC-AUC
- Precision–Recall curves
- mAP localization metrics

## Results
- Model	Task	Performance
- DenseNet121	Classification	85.67% Accuracy
- ResNet50	Classification	85.34% Accuracy
- Faster R-CNN	Detection	mAP 0.132
- YOLOv5n	Detection	mAP 0.097
- Two-Stage Pipeline	Combined	mAP ~0.07 (performance drop)

## Key Insights

- Detection-only pipelines outperform filtered pipelines.
- Classification recall (~53%) created a bottleneck.
- In healthcare AI, high recall > high accuracy.
- Two-stage detectors remain superior for fine-grained localization.

## Author
Victoria Piroian

Cornell University

Faculty of Operations Research & Information Engineering, 2025
