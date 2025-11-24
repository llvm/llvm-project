# Hardware AI Capabilities Quick Reference

**Classification:** NATO UNCLASSIFIED (EXERCISE)  
**Asset:** JRTC1-5450-MILSPEC  
**Date:** 2025-11-22  
**Purpose:** Quick reference for hardware AI capabilities

---

## Core SoC: Intel Core Ultra 7 165H

### NPU (Neural Processing Unit) - Intel NPU 3720

| Specification | Value |
|---------------|-------|
| **Compute** | 30 TOPS INT8 (military-optimized from 13 TOPS) |
| **Power** | 5-8W typical, 12W peak |
| **Latency** | <10ms typical inference |
| **Throughput** | 1000+ inferences/sec (small models) |
| **Quantization** | INT8 primary, INT4 experimental |

**Best For:**
- ✅ Real-time inference (<10ms)
- ✅ Edge AI, always-on models
- ✅ Power-efficient operation (5-8W)
- ✅ Small models (<500M parameters)
- ✅ Continuous monitoring

**Limitations:**
- ❌ No FP32 support
- ❌ Limited model size (<500M params)
- ❌ Shared memory bandwidth

**Optimal Layers:** 3, 4, 5, 7, 8

---

### iGPU (Integrated Graphics) - Intel Arc 8 Xe-cores

| Specification | Value |
|---------------|-------|
| **Compute** | 40 TOPS INT8 (military-tuned from 32 TOPS) |
| **Power** | 15-25W typical, 35W peak |
| **Latency** | 20-50ms for vision models |
| **Throughput** | 30-60 FPS video processing |
| **Quantization** | INT8, FP16, FP32 (XMX engines) |
| **Memory** | Shared 32GB LPDDR5x (120 GB/s) |

**Architecture:**
- 8 Xe-cores, 1024 ALUs
- XMX (Xe Matrix Extensions) engines
- Hardware matrix acceleration

**Best For:**
- ✅ Vision AI (CNN, ViT, YOLO)
- ✅ Graphics ML, image processing
- ✅ Multi-modal models (CLIP)
- ✅ Generative AI (small Stable Diffusion)
- ✅ Parallel processing

**Limitations:**
- ❌ Shared memory with CPU
- ❌ Higher power than NPU
- ❌ Limited to ~500M params efficiently

**Optimal Layers:** 3, 5, 7, 8

---

### CPU AMX (Advanced Matrix Extensions)

| Specification | Value |
|---------------|-------|
| **Compute** | 32 TOPS INT8 (all cores) |
| **Cores** | 6 P-cores + 8 E-cores + 2 LP E-cores |
| **Power** | 28W base, 64W turbo |
| **Latency** | 50-200ms (model dependent) |
| **Quantization** | INT8, BF16 |
| **Memory** | Full 32GB system RAM |

**Core Breakdown:**
- P-cores (Performance): 19.2 TOPS
- E-cores (Efficiency): 8.0 TOPS
- LP E-cores (Low Power): 4.8 TOPS

**Best For:**
- ✅ Transformer models (BERT, GPT, T5)
- ✅ LLM inference (up to 7B params)
- ✅ Matrix-heavy operations
- ✅ Batch processing
- ✅ High memory bandwidth workloads

**Limitations:**
- ❌ Higher power consumption
- ❌ Thermal constraints
- ❌ Requires AMX-optimized code

**Optimal Layers:** 4, 5, 6, 7, 9

---

### CPU AVX-512 (Vector Units)

| Specification | Value |
|---------------|-------|
| **Compute** | ~10 TOPS INT8 (vectorized) |
| **Width** | 512-bit vector registers |
| **Power** | Included in CPU TDP |
| **Latency** | <1ms for preprocessing |
| **Throughput** | 10-100 GB/s data processing |

**Best For:**
- ✅ Data preprocessing/normalization
- ✅ Post-processing (softmax, NMS)
- ✅ Classical ML (SVM, Random Forest)
- ✅ Vectorized operations
- ✅ Statistical computing

**Limitations:**
- ❌ Not optimized for deep learning
- ❌ Lower TOPS than specialized accelerators

**Optimal Layers:** All (preprocessing/post-processing)

---

## Hardware Selection Guide

### By Latency Requirement

| Latency Target | Use This | Typical Workload |
|----------------|----------|------------------|
| **<10ms** | NPU | Real-time classification, edge AI |
| **<50ms** | iGPU | Vision AI, object detection |
| **<200ms** | CPU AMX | NLP, transformers, decision support |
| **<1000ms** | CPU AMX + Custom | LLM inference, strategic analysis |

### By Model Type

| Model Type | Primary Accelerator | Secondary | Layers |
|------------|-------------------|-----------|--------|
| **CNN (Vision)** | iGPU | NPU | 3, 5, 7, 8 |
| **RNN/LSTM** | NPU | CPU AMX | 3, 4, 5 |
| **Transformers** | CPU AMX | iGPU | 4, 5, 7, 9 |
| **LLM (1-7B)** | CPU AMX + Custom | - | 7, 9 |
| **Generative AI** | iGPU | CPU AMX | 7 |
| **Classical ML** | AVX-512 | NPU | 3, 4, 5 |

### By Model Size

| Model Size | Accelerator | Quantization | Latency |
|------------|-------------|--------------|---------|
| **<100M params** | NPU | INT8 | <10ms |
| **100-500M params** | iGPU or CPU AMX | INT8/FP16 | <100ms |
| **500M-1B params** | CPU AMX | INT8 | <300ms |
| **1B-7B params** | CPU AMX + Custom | INT8 | <1000ms |

### By Power Budget

| Power Budget | Accelerators | Use Case |
|--------------|--------------|----------|
| **<10W** | NPU only | Edge AI, battery operation |
| **<30W** | NPU + iGPU | Mobile workstation |
| **<80W** | NPU + iGPU + CPU (base) | Standard operation |
| **<150W** | All accelerators | Full capability |

---

## Memory Considerations

### System Memory: 32GB LPDDR5x-7467

| Component | Allocation | Bandwidth |
|-----------|------------|-----------|
| **OS + Apps** | 8-12GB | Dynamic |
| **NPU Reserved** | 2-4GB | Shared |
| **iGPU Reserved** | 4-8GB | 120 GB/s |
| **AI Models** | 8-16GB | Dynamic |
| **Available** | 4-8GB | Buffer |

### Model Memory Requirements

| Model Size | INT8 | FP16 | FP32 |
|------------|------|------|------|
| **100M params** | 100MB | 200MB | 400MB |
| **500M params** | 500MB | 1GB | 2GB |
| **1B params** | 1GB | 2GB | 4GB |
| **7B params** | 7GB | 14GB | 28GB |

**Note:** INT8 quantization enables 7B models in 32GB RAM with headroom for OS and activations.

---

## Thermal & Power Management

### Thermal Limits

| Component | Max Temp | Sustained Temp | Throttle Point |
|-----------|----------|----------------|----------------|
| **CPU** | 100°C | 85°C | 90°C |
| **NPU** | 85°C | 75°C | 80°C |
| **iGPU** | 95°C | 85°C | 90°C |
| **M.2 Accelerators** | 80°C | 70°C | 75°C |

### Power States

| State | Power | Active Components | Use Case |
|-------|-------|-------------------|----------|
| **Idle** | 5-10W | NPU (low power) | Monitoring, standby |
| **Light** | 30-50W | NPU + iGPU | Real-time analytics |
| **Medium** | 80-120W | NPU + iGPU + CPU | Operational workloads |
| **Heavy** | 150W+ | All accelerators | Full capability |

---

## Performance Optimization Tips

### For NPU
1. **Quantize to INT8** - 4x speedup vs FP32
2. **Batch size 1-4** - Optimized for low latency
3. **Model size <500M** - Fits in NPU memory
4. **Avoid FP32** - Not supported, use INT8/INT4

### For iGPU
1. **Use XMX engines** - Hardware matrix acceleration
2. **FP16 quantization** - Good balance of speed/accuracy
3. **Batch processing** - Better GPU utilization
4. **Optimize memory transfers** - Minimize CPU-GPU copies

### For CPU AMX
1. **Use AMX intrinsics** - 8x faster than standard ops
2. **Tile-based computation** - Leverage 8x16 tiles
3. **BF16 for precision** - Better than FP32, faster than FP16
4. **Batch processing** - Amortize overhead

### For All Accelerators
1. **Model quantization** - INT8 primary, FP16 fallback
2. **Graph optimization** - Fuse operations, remove redundancy
3. **Memory management** - Minimize allocations
4. **Thermal monitoring** - Avoid throttling
5. **Power profiling** - Stay within budget

---

## Quick Decision Matrix

### "Which accelerator should I use?"

```
Is latency <10ms critical?
├─ YES → Use NPU (if model <500M params)
└─ NO → Continue...

Is it a vision/graphics workload?
├─ YES → Use iGPU (if model <500M params)
└─ NO → Continue...

Is it a transformer/LLM?
├─ YES → Use CPU AMX (up to 7B params with INT8)
└─ NO → Continue...

Is it classical ML or preprocessing?
├─ YES → Use AVX-512
└─ NO → Use combination based on model size
```

### "How much power will I use?"

```
Model Size + Latency Requirement = Power Budget

Small (<100M) + Fast (<10ms) = 5-10W (NPU)
Medium (100-500M) + Medium (<100ms) = 30-50W (NPU + iGPU)
Large (500M-1B) + Slow (<300ms) = 80-120W (NPU + iGPU + CPU)
Very Large (1B-7B) + Very Slow (<1000ms) = 150W+ (All)
```

---

## Software Stack

### Inference Engines
- **ONNX Runtime** - Cross-platform, optimized for NPU/iGPU
- **OpenVINO** - Intel-optimized, best for NPU/iGPU/CPU
- **TensorFlow Lite** - Mobile-optimized, good for NPU
- **PyTorch Mobile** - Research-friendly, CPU/GPU

### Quantization Tools
- **Intel Neural Compressor** - Best for Intel hardware
- **ONNX Quantization** - Cross-platform
- **PyTorch Quantization** - Native PyTorch
- **TensorFlow Quantization** - Native TensorFlow

### Optimization
- **Intel IPEX-LLM** - LLM optimization for Intel
- **OpenVINO Model Optimizer** - Graph optimization
- **ONNX Graph Optimization** - Cross-platform
- **TensorRT** - NVIDIA (if using discrete GPU)

---

## Example Configurations

### Configuration 1: Real-Time Edge AI
- **Accelerator:** NPU (30 TOPS)
- **Models:** MobileNet, EfficientNet, small YOLO
- **Latency:** <10ms
- **Power:** 5-10W
- **Layers:** 3, 8

### Configuration 2: Vision AI Workstation
- **Accelerators:** NPU + iGPU (70 TOPS combined)
- **Models:** ResNet-50, YOLOv8, ViT
- **Latency:** <50ms
- **Power:** 30-50W
- **Layers:** 3, 5, 7

### Configuration 3: NLP & Decision Support
- **Accelerators:** CPU AMX + NPU (62 TOPS)
- **Models:** BERT, T5, GPT-2
- **Latency:** <200ms
- **Power:** 80-120W
- **Layers:** 4, 5, 7

### Configuration 4: LLM Inference
- **Accelerators:** CPU AMX + Custom (1000+ TOPS)
- **Models:** LLaMA-7B, Mistral-7B (INT8)
- **Latency:** <1000ms
- **Power:** 150W+
- **Layers:** 7, 9

---

## Classification

**NATO UNCLASSIFIED (EXERCISE)**  
**Asset:** JRTC1-5450-MILSPEC  
**Date:** 2025-11-22

---

## Related Documentation

- **COMPLETE_AI_ARCHITECTURE_LAYERS_3_9.md** - Full system architecture
- **Hardware/INTERNAL_HARDWARE_MAPPING.md** - Detailed hardware mapping
- **AI_ARCHITECTURE_PLANNING_GUIDE.md** - Implementation planning

