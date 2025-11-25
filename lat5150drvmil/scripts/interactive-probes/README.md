# Interactive System Probes

**For Deployment on Dell Latitude 5450 MIL-SPEC Hardware**

Quick interactive scripts to test and explore each subsystem of the LAT5150DRVMIL AI framework when deployed on the actual Dell Latitude 5450 MIL-SPEC platform with:
- Intel NPU (Military-grade: 34-49.4 TOPS)
- Intel GNA 3.5
- Intel Arc GPU
- Intel NCS2 sticks (2-3 units)
- AVX-512 on P-cores (CPUs 0-5)

**Note**: These probes are designed to run on the actual hardware. Some features will not work on development machines without the MIL-SPEC hardware.

## Available Probes

### 01_test_quantization.py
Test hardware-aware quantization system:
- Hardware detection (NPU, GNA, Arc GPU, NCS2, AVX-512)
- Quantization recommendations for different model sizes
- Hardware-specific optimization strategies
- NCS2 deployment planning

```bash
python 01_test_quantization.py
```

### 02_test_moe_system.py
Test Mixture of Experts routing:
- Expert domain routing
- Multi-expert selection
- Aggregation strategies
- Routing statistics

```bash
python 02_test_moe_system.py
```

### 03_test_hardware.py
Comprehensive hardware detection:
- AVX-512 on P-cores
- Intel NPU (Military-grade)
- Intel GNA 3.5
- Intel Arc GPU
- Intel NCS2 sticks (2-3 units)
- Hardware summary and recommendations

```bash
python 03_test_hardware.py
```

### 04_test_dsmil_enumeration.py
DSMIL device enumeration and testing:
- DSMIL driver status
- Device node detection
- Military token validation (0x049e-0x04a3)
- 22 implemented devices (22/108)
- Launch full DSMIL TUI menu
- Device discovery and probing

```bash
python 04_test_dsmil_enumeration.py
```

**Requires**: DSMIL driver loaded on actual hardware

## Hardware Configuration (Target System)

**Dell Latitude 5450 MIL-SPEC Covert Edition**:
- CPU: Intel Core Ultra 7 165H (6 P-cores + 10 E-cores)
- AI Accelerators:
  - Intel NPU (Military: 34-49.4 TOPS)
  - Intel GNA 3.5
  - Intel Arc GPU (~8-16 TFLOPS)
  - Intel NCS2 sticks (2-3 units, ~1-2 TOPS each)
- AVX-512 on P-cores (CPUs 0-5)
- DSMIL: 108 hardware devices (22 implemented)
- Military Tokens: 6 tokens (0x049e-0x04a3)

**Total AI Compute**: ~45-70 TOPS equivalent

**⚠️ Important**: These probes are designed for the actual MIL-SPEC hardware. Running on development systems will show limited functionality.

## Deployment Instructions

### On Dell Latitude 5450 MIL-SPEC Hardware

1. **Clone repository to target system**:
```bash
git clone <repo-url> ~/LAT5150DRVMIL
cd ~/LAT5150DRVMIL/scripts/interactive-probes
```

2. **Load DSMIL driver** (for probe 04):
```bash
sudo modprobe dsmil-84dev   # legacy alias dsmil-72dev still works
# Or: sudo insmod /path/to/dsmil-84dev.ko (symlink dsmil-72dev.ko is provided)
```

3. **Enable AVX-512** (if not already):
```bash
cd ../../avx512-unlock
sudo ./unlock_avx512_advanced.sh enable
```

4. **Run probes**:
```bash
cd ../scripts/interactive-probes
python 01_test_quantization.py   # Hardware-aware quantization
python 02_test_moe_system.py     # MoE routing test
python 03_test_hardware.py       # Full hardware detection
python 04_test_dsmil_enumeration.py  # DSMIL device enumeration
```

### Quick Reference (On Target Hardware)

```bash
# Run all hardware checks
python 03_test_hardware.py  # Option 7: Run all checks

# Test quantization for your models
python 01_test_quantization.py  # Option 5: Run all tests

# Test MoE routing
python 02_test_moe_system.py    # Option 1: Test routing

# Enumerate DSMIL devices
python 04_test_dsmil_enumeration.py  # Option 9: Run all checks
```

### Hardware-Specific Recommendations

**For maximum throughput** (NPU):
```python
# Use INT8 quantization
# Deploy to NPU
# Expected: 3-4x speedup
```

**For best quality** (P-cores + AVX-512):
```python
# Use BF16 quantization
# Pin to P-cores: taskset -c 0-5
# Expected: 1.5-2x speedup
```

**For edge deployment** (NCS2):
```python
# Use OpenVINO IR (INT8)
# Shard across 2-3 NCS2 sticks
# Expected: 10-30 tokens/sec total
```

## Adding New Probes

Create new probe script:
```bash
touch 04_test_<subsystem>.py
chmod +x 04_test_<subsystem>.py
```

Template:
```python
#!/usr/bin/env python3
"""
Interactive <Subsystem> Probe

Brief description.

Usage:
    python 04_test_<subsystem>.py
"""

import sys
sys.path.insert(0, '../../02-ai-engine')

# Your imports here

def test_feature():
    """Test a specific feature"""
    pass

def interactive_menu():
    """Interactive menu"""
    while True:
        print("\n1. Test feature 1")
        print("0. Exit")
        choice = input("\nSelect: ").strip()
        if choice == "1":
            test_feature()
        elif choice == "0":
            break

if __name__ == "__main__":
    interactive_menu()
```
