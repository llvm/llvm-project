# Hardware Optimization Plan - Squeeze Every TOPS

**Current Bottleneck:** Ollama using 100% CPU, GPU completely idle!

**Date:** 2025-10-29

---

## 🔍 CURRENT UTILIZATION AUDIT

### What's Being Used

| Hardware | Capability | Current Use | Utilization |
|----------|------------|-------------|-------------|
| **CPU** | 20 cores | Ollama DeepSeek | ~100% during inference |
| **NPU** | 34 TOPS | Claude-backups Opus (ports 3451-3454) | Unknown% |
| **GPU** | 40 TOPS | **NOTHING** | **0%** ⚠️ |
| **NCS2** | 10 TOPS | **NOTHING** | **0%** ⚠️ |

**Total Available:** 84 TOPS
**Currently Used:** ~34 TOPS (NPU only)
**WASTED:** **50 TOPS** (GPU + NCS2 completely idle!)

---

## 🎯 OPTIMIZATION OPPORTUNITIES

### Priority 1: GPU Acceleration (MASSIVE WIN)

**Problem:** Ollama running on CPU (slow)
**Solution:** Enable Intel Arc GPU acceleration
**Gain:** 10-20× faster inference expected

**Approaches:**

#### Option A: Ollama GPU Support (If Available)
```bash
# Check if Ollama 0.12.5 supports Intel
ollama serve --help | grep -i intel

# Try Intel GPU env var
export OLLAMA_INTEL_GPU=1
sudo systemctl restart ollama
```

**Status:** Ollama primarily supports NVIDIA/AMD, Intel support uncertain

#### Option B: OpenVINO with GPU Backend ⭐ RECOMMENDED
```bash
# Use claude-backups OpenVINO models on GPU
# Already running on port 3452 (GPU config)!

# Test it:
curl http://localhost:3452/health
```

**Status:** Already installed, need to integrate

#### Option C: llama.cpp with SYCL (Intel GPU)
```bash
# Compile llama.cpp with Intel SYCL support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_SYCL=ON
make -j16

# Use Arc GPU directly
./llama-server --model deepseek-r1.gguf --n-gpu-layers 99
```

**Status:** Requires compilation, most reliable for Intel Arc

### Priority 2: Multi-Accelerator Orchestration

**Use different hardware for different tasks:**
- **GPU (40 TOPS):** Main model inference (generation)
- **NPU (34 TOPS):** Embeddings, small tasks
- **NCS2 (10 TOPS):** Routing, classification
- **CPU:** Coordination only

**Expected:** 2-5× throughput by parallelizing

### Priority 3: NCS2 Integration

**Current:** Detected but not used
**Needed:** OpenVINO integration
**Gain:** +10 TOPS for inference

```bash
# Already installed from claude-backups
ls /home/john/claude-backups/openvino/

# Need to add NCS2 device
```

---

## 📋 IMPLEMENTATION PLAN

### Phase 1: GPU Acceleration (30 min)

**Step 1: Check Claude-Backups GPU Server**
```bash
# Already running on port 3452!
curl http://localhost:3452/health

# Test inference
curl -X POST http://localhost:3452/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "max_tokens": 50}'
```

**Step 2: Compare Performance**
```bash
# CPU (current): 4-60s
time python3 /home/john/LAT5150DRVMIL/02-ai-engine/dsmil_ai_engine.py prompt "test"

# GPU (claude-backups): Should be much faster
time curl -X POST http://localhost:3452/generate -d '{"prompt":"test"}'
```

**Step 3: Integrate Best Performer**
- If GPU is faster, use it as default
- Update unified_orchestrator to route to fastest backend
- Keep DSMIL attestation for local

### Phase 2: Download Coding Models (20 min)

**Code-specialized models:**
```bash
# DeepSeek Coder (best for code)
ollama pull deepseek-coder:6.7b-instruct  # 3.8GB

# Qwen Coder (excellent quality)
ollama pull qwen2.5-coder:7b              # 4.7GB

# CodeLlama smaller (faster)
ollama pull codellama:13b-instruct        # 7.4GB
```

**If GPU works:** These will be 10-20× faster!

### Phase 3: Multi-Accelerator Setup (30 min)

**Strategy:**
```python
# Small/fast tasks → NCS2 or NPU
# Code generation → GPU (Arc 40 TOPS)
# Code review → CPU + GPU hybrid
# Complex analysis → All accelerators in parallel
```

**Implementation:**
- Route by task complexity
- Load balance across hardware
- Parallel execution where possible

---

## 🚀 MAXIMUM PERFORMANCE CONFIGURATION

### Ideal Setup (All Hardware Used)

```
User Query
    ↓
┌───────────┴──────────────┐
│  Task Classification     │  ← NCS2 (10 TOPS, <100ms)
└───────────┬──────────────┘
            │
    ┌───────┼────────┬──────────┐
    │       │        │          │
    ▼       ▼        ▼          ▼
  [GPU]   [NPU]   [CPU]     [NCS2]
  40 TOPS 34 TOPS 20-core  10 TOPS
    │       │        │          │
    └───────┴────────┴──────────┘
                 │
                 ▼
            Combined Result
```

**Expected Performance:**
- Simple queries: <1s (NCS2 classification + GPU inference)
- Code generation: 2-5s (GPU-accelerated)
- Complex tasks: 5-15s (multi-accelerator)
- Current: 5-60s (CPU-only)

**Total speedup:** 3-10× faster

---

## 🔧 IMMEDIATE ACTIONS

### 1. Test Claude-Backups GPU Server (NOW)

```bash
curl http://localhost:3452/health
```

**If working:** Use it instead of CPU Ollama!

### 2. Download Coding Models

```bash
ollama pull deepseek-coder:6.7b-instruct
ollama pull qwen2.5-coder:7b
```

### 3. Enable GPU in Ollama (if supported)

```bash
# Check Ollama capabilities
ollama serve --help | grep -i gpu

# Or compile llama.cpp with SYCL for guaranteed Intel GPU support
```

---

## 📊 EXPECTED GAINS

### Current State
- CPU: 100% during inference (bottleneck)
- GPU: 0% (wasted 40 TOPS)
- NPU: Used by claude-backups
- NCS2: 0% (wasted 10 TOPS)
- **Speed:** 4-60s

### After GPU Optimization
- CPU: 10-20% (coordination only)
- **GPU: 70-90%** (main inference)
- NPU: Used by claude-backups
- NCS2: Integrated
- **Speed:** 0.5-10s (5-10× faster)

### After Full Optimization
- CPU: 10% (coordination)
- GPU: 80% (generation)
- NPU: 60% (embeddings)
- NCS2: 80% (classification)
- **Speed:** 0.3-5s (10-20× faster)
- **Parallel:** Multiple queries simultaneously

---

**Want me to:**
1. Test claude-backups GPU server (port 3452)
2. Download coding models
3. Build multi-accelerator orchestration?

This will unlock that wasted 50 TOPS!
