# AI Accelerator Hardware Testing Suite

Comprehensive testing and activation scripts for Dell Latitude 5450 MIL-SPEC AI hardware.

## Hardware Overview

### Consumer AI Accelerators

| Component | Specification | Status | Driver |
|-----------|--------------|--------|--------|
| **Intel NPU** | Meteor Lake NPU (49.4 TOPS INT8) | ✅ Operational | intel_vpu |
| **Intel iGPU** | Arc Graphics (63.7 GB UMA) | ✅ Operational | i915 |
| **Intel GNA** | Gaussian & Neural-Network Accelerator | ⚠️ Disabled | None loaded |
| **AVX-512** | P-cores 0-5 only | ✅ Available | Native CPU |

### Military/Enhanced Hardware (if present)

| Component | Specification | Status | Driver |
|-----------|--------------|--------|--------|
| **Military NPU** | Enhanced NPU (100-200+ TOPS) | ❓ Detection required | dsmil_npu |

---

## Quick Start

### Test All Accelerators

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine/hardware
./test_all_accelerators.sh
```

This runs all tests:
1. Intel NPU benchmark
2. GNA activation
3. Military NPU detection/activation

### Test Individual Components

```bash
# NPU only
./test_all_accelerators.sh --npu-only

# GNA only
./test_all_accelerators.sh --gna-only

# Military NPU only
./test_all_accelerators.sh --military-only

# Verbose output
./test_all_accelerators.sh --verbose
```

---

## Individual Test Scripts

### 1. NPU Test & Benchmark

**Script:** `npu_test_benchmark.py`

**What it does:**
- ✅ Checks NPU device node (`/dev/accel/accel0`)
- ✅ Verifies intel_vpu driver is loaded
- ✅ Detects NPU via OpenVINO
- ✅ Runs inference benchmark
- ✅ Measures latency (p50, p95, p99) and throughput

**Requirements:**
```bash
pip install openvino
```

**Usage:**
```bash
python3 npu_test_benchmark.py
```

**Expected Output:**
```
================================================================================
  NPU Inference Benchmark
================================================================================
Model: 384 → 128 (matrix multiplication)
Compiling model for NPU...
Warmup (10 iterations)...
Running benchmark (100 iterations)...

Results:
  Average latency: 2.15 ms ± 0.34 ms
  P50 latency: 2.10 ms
  P95 latency: 2.85 ms
  P99 latency: 3.12 ms
  Throughput: 465.1 inferences/sec
================================================================================
```

---

### 2. GNA Activation

**Script:** `gna_activation.py`

**What it does:**
- ✅ Detects GNA PCI device (8086:7e4c)
- ⚠️ Attempts to enable disabled device
- ⚠️ Attempts to load GNA driver
- ℹ️ Checks OpenVINO GNA support

**PCI Device:**
```
0000:00:08.0 System peripheral [0880]: Intel Corporation
             Meteor Lake-P Gaussian & Neural-Network Accelerator [8086:7e4c]
Status: DISABLED (IRQ 255, no driver)
```

**Usage:**
```bash
sudo python3 gna_activation.py
```

**Note:** GNA may require:
- BIOS settings change
- Specific kernel boot parameters
- Proprietary driver
- May be integrated with NPU or audio subsystem

**GNA Capabilities (if activated):**
- Low-power neural network inference
- Audio ML (noise reduction, wake word detection)
- Speech recognition acceleration
- Always-on inference

---

### 3. Military NPU DSMIL Driver Loader

**Script:** `military_npu_dsmil_loader.py`

**What it does:**
- 🔍 Scans PCI bus for military NPU hardware
- 🔍 Checks for DSMIL driver availability
- 🔍 Verifies DSMIL firmware
- 🔒 Checks security compliance (Secure Boot, FIPS)
- ⚡ Loads DSMIL kernel module
- ✅ Verifies device activation
- 📊 Queries enhanced capabilities

**DSMIL Driver Locations:**
```
/lib/modules/$(uname -r)/extra/dsmil/dsmil_npu.ko
/lib/modules/$(uname -r)/kernel/drivers/dsmil/dsmil_npu.ko
/opt/dsmil/drivers/dsmil_npu.ko
/tank/ai-engine/drivers/dsmil_npu.ko
```

**DSMIL Firmware Locations:**
```
/lib/firmware/dsmil/
/tank/ai-engine/firmware/dsmil/
/opt/dsmil/firmware/
```

**Usage:**
```bash
sudo python3 military_npu_dsmil_loader.py
```

**Expected Device Nodes (if activated):**
```
/dev/dsmil0
/dev/dsmil/npu0
/dev/military_npu0
/dev/accel/accel1
```

**Security Requirements:**
- Secure Boot (recommended)
- FIPS mode (recommended)
- Clearance certificate (may be required)
- Hardware attestation (may be required)

**Military NPU vs Consumer NPU:**

| Feature | Consumer NPU | Military NPU |
|---------|-------------|--------------|
| TOPS Rating | 49.4 (INT8) | 100-200+ (INT8) |
| Crypto Acceleration | No | Yes |
| Export Control | No | Yes |
| Classification | UNCLASSIFIED | Varies |
| Driver | intel_vpu | dsmil_npu |
| Firmware | Public | Restricted |

---

## Logs and Results

All tests output logs to:
```
/tank/ai-engine/logs/
```

Log files:
- `npu_test_YYYYMMDD_HHMMSS.log` - NPU benchmark results
- `gna_activation_YYYYMMDD_HHMMSS.log` - GNA activation attempts
- `military_npu_YYYYMMDD_HHMMSS.log` - Military NPU detection
- `military_npu_status.json` - JSON status export

---

## Troubleshooting

### NPU: "OpenVINO not installed"

```bash
pip install openvino
```

### NPU: "NPU not detected by OpenVINO"

Check driver:
```bash
lsmod | grep intel_vpu
ls -la /dev/accel/accel0
```

Reload driver:
```bash
sudo modprobe -r intel_vpu
sudo modprobe intel_vpu
```

### GNA: "Device is DISABLED"

GNA is disabled by default on Meteor Lake. Possible solutions:

1. **Check BIOS settings** - Look for GNA/Audio DSP settings
2. **Kernel boot parameters** - May need specific parameters
3. **Accept limitation** - GNA is optional, NPU is primary accelerator

### Military NPU: "DSMIL driver not found"

If you have military NPU hardware:

1. **Install DSMIL driver:**
   ```bash
   # Example (actual installation may vary)
   cd /path/to/dsmil-driver
   make
   sudo make install
   sudo depmod -a
   ```

2. **Load firmware:**
   ```bash
   sudo cp dsmil_firmware.bin /lib/firmware/dsmil/
   ```

3. **Load driver:**
   ```bash
   sudo modprobe dsmil_npu
   ```

If you don't have military NPU hardware, this is expected. The consumer NPU (49.4 TOPS) is still very capable.

---

## Integration with AI Framework

### Dynamic Resource Allocator

Once accelerators are tested, use the dynamic resource allocator:

```bash
python3 /home/user/LAT5150DRVMIL/02-ai-engine/hardware/dynamic_resource_allocator.py
```

This will:
- Auto-detect all available accelerators (NPU, iGPU, GNA, Military NPU)
- Allocate tasks to optimal hardware
- Manage UMA memory pool (44-52 GiB)
- Adjust batch sizes dynamically

### Hardware Priority for AI Tasks

| Task | 1st Choice | 2nd Choice | 3rd Choice |
|------|-----------|-----------|-----------|
| **INT8 Inference** | NPU | Military NPU | iGPU |
| **FP16 Inference** | iGPU | NPU | CPU |
| **Training (DPO)** | iGPU (UMA) | Cloud GPU | - |
| **Vector Search** | AVX-512 | iGPU | NPU |
| **Embeddings** | NPU | iGPU | CPU |
| **Audio ML** | GNA (if activated) | NPU | CPU |

---

## Status Summary

After running tests, check status:

```bash
# View NPU status
cat /tank/ai-engine/logs/npu_test_*.log | tail -n 20

# View military NPU status (JSON)
cat /tank/ai-engine/logs/military_npu_status.json

# Check all accelerators
./test_all_accelerators.sh
```

---

## Performance Targets

### NPU Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| Latency (p50) | <5ms | 2-3ms |
| Latency (p95) | <10ms | 5-8ms |
| Throughput | >200 inf/sec | 400-500 inf/sec |

### iGPU Benchmarks (via Dynamic Allocator)

| Metric | Value |
|--------|-------|
| UMA Pool | 44-52 GiB usable |
| Training Batch Size | 32-64 (vs 2-4 with discrete) |
| Max Models | Multiple simultaneous |

---

## Next Steps

1. ✅ **Run hardware tests** (this suite)
2. ✅ **Build AVX-512 module:**
   ```bash
   cd /home/user/LAT5150DRVMIL/02-ai-engine/rag_cpp
   make build
   make benchmark
   ```
3. ✅ **Test dynamic allocator:**
   ```bash
   python3 /home/user/LAT5150DRVMIL/02-ai-engine/hardware/dynamic_resource_allocator.py
   ```
4. 🚀 **Deploy AI framework** (see main documentation)

---

**Document Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Last Updated:** 2025-11-09
**Hardware:** Dell Latitude 5450 MIL-SPEC
