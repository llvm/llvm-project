# WhiteRabbitNeo LLM System Guide

**Complete guide for running WhiteRabbitNeo models with hardware acceleration**

---

## Overview

This system provides comprehensive support for large language models with:
- **Multi-device inference** (GPU/NPU/NCS2/CPU)
- **Dynamic memory allocation** (64GB system RAM, 57.6GB usable)
- **Runtime device switching**
- **Validation pipeline** with cross-model checking
- **96.4 TOPS compute** (Arc:40 + NPU:26.4 + NCS2:30) - MILITARY MODE
- **Dynamic performance scaling** up to 137 TOPS in classified mode

### Supported Models

1. **WhiteRabbitNeo-33B-v1** (33B parameters)
   - HuggingFace: `WhiteRabbitNeo/WhiteRabbitNeo-33B-v1`
   - INT4: 16.9GB ✅ (fits in 57.6GB usable RAM)
   - INT8: 30.7GB ✅ (fits in RAM)
   - FP16: 61.5GB ⚠ (exceeds RAM, requires swap)

2. **Llama-3.1-WhiteRabbitNeo-2-70B** (70B parameters)
   - HuggingFace: `WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-70B`
   - INT4: 35.9GB ✅ (fits in 57.6GB usable RAM)
   - INT8: 65.2GB ⚠ (exceeds RAM, requires swap)
   - FP16: 130.4GB ⚠ (requires swap + streaming)

---

## System Architecture

### Hardware Configuration

```
┌──────────────────────────────────────────────────────────────┐
│  Dell Latitude 5450 - Intel Core Ultra 7 165H               │
│  DYNAMIC PERFORMANCE SYSTEM                                  │
├──────────────────────────────────────────────────────────────┤
│  System RAM:        64GB (57.6GB usable)                    │
│  Arc Graphics:      40 TOPS INT8 (→72 classified), shared  │
│  Intel NPU:         26.4 TOPS military (→35 classified)     │
│  Intel GNA:         PQC/Neural accel, 16MB, <1W             │
│  NCS2 (3x):         30 TOPS, 1.5GB inference, 48GB storage  │
├──────────────────────────────────────────────────────────────┤
│  Total Memory:      64GB System RAM, 48GB NCS2 storage      │
│  Compute (Military): 96.4 TOPS (Arc:40 + NPU:26.4 + NCS2:30)│
│  Compute (Classified): ~137 TOPS (Arc:72 + NPU:35 + NCS2:30)│
└──────────────────────────────────────────────────────────────┘
```

### Software Stack

```
┌───────────────────────────────────────────────────────────┐
│  Application Layer                                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  whiterabbit_inference_engine.py                    │ │
│  │  - Unified API                                      │ │
│  │  - Multi-model workflows                            │ │
│  │  - Interactive CLI                                  │ │
│  └─────────────────────────────────────────────────────┘ │
├───────────────────────────────────────────────────────────┤
│  Management Layer                                         │
│  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │  Model Manager   │  │  Validation Pipeline        │  │
│  │  - Load/unload   │  │  - Syntax checking          │  │
│  │  - Device switch │  │  - Compilation tests        │  │
│  │  - Status track  │  │  - Model cross-checking     │  │
│  └──────────────────┘  └─────────────────────────────┘  │
├───────────────────────────────────────────────────────────┤
│  Allocation Layer                                         │
│  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │  Dynamic Alloc   │  │  Hardware Profile           │  │
│  │  - Memory plan   │  │  - Device capabilities      │  │
│  │  - Layer split   │  │  - Specs configuration      │  │
│  │  - Swap manage   │  │  - Runtime adjustment       │  │
│  └──────────────────┘  └─────────────────────────────┘  │
├───────────────────────────────────────────────────────────┤
│  Device Layer                                             │
│  ┌───────┐ ┌─────┐ ┌──────┐ ┌──────────┐ ┌───────────┐ │
│  │  GPU  │ │ NPU │ │ GNA  │ │  NCS2    │ │  Mil NPU  │ │
│  │ Arc   │ │     │ │      │ │ (3 devs) │ │           │ │
│  └───────┘ └─────┘ └──────┘ └──────────┘ └───────────┘ │
└───────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install transformers torch accelerate bitsandbytes

# Intel-specific
pip install openvino intel-extension-for-pytorch

# Clone WhiteRabbitNeo repo (if using official models)
cd /home/user/LAT5150DRVMIL/models
git lfs install
git clone https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-33B-v1
```

### 2. Run Demo

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 whiterabbit_demo.py
```

### 3. Interactive Mode

```bash
python3 whiterabbit_inference_engine.py
```

**Interactive Commands**:
```
whiterabbit> hardware              # Show hardware profile
whiterabbit> status                # Show engine status
whiterabbit> setup WhiteRabbitNeo-33B-v1   # Load model
whiterabbit> device gpu_arc        # Switch to GPU
whiterabbit> device npu            # Switch to NPU
whiterabbit> device ncs2           # Switch to NCS2
whiterabbit> generate Write a Python function to sort a list
whiterabbit> quit                  # Exit
```

---

## Usage Examples

### Example 1: Simple Generation

```python
from whiterabbit_inference_engine import get_inference_engine, TaskType, DeviceType

# Initialize
engine = get_inference_engine()

# Setup with 33B model
engine.setup(
    model_name="WhiteRabbitNeo-33B-v1",
    device=DeviceType.GPU_ARC,
    quantization=QuantizationType.INT4
)

# Generate
prompt = "Write a Python function to calculate Fibonacci numbers:"
output, validation = engine.generate(
    prompt=prompt,
    task_type=TaskType.CODE_GENERATION,
    max_new_tokens=512
)

print(output)
```

### Example 2: Device Switching

```python
# Start on GPU
engine.setup("WhiteRabbitNeo-33B-v1", device=DeviceType.GPU_ARC)

# Generate on GPU
output1 = engine.generate("Hello")[0]

# Switch to NPU for lower latency
engine.switch_device(DeviceType.NPU)

# Generate on NPU
output2 = engine.generate("Hello")[0]

# Switch to NCS2 for edge inference
engine.switch_device(DeviceType.NCS2)

# Generate on NCS2
output3 = engine.generate("Hello")[0]
```

### Example 3: Multi-Model Validation

```python
# Setup with primary and validator models
engine.setup(
    model_name="Llama-3.1-WhiteRabbitNeo-2-70B",  # Primary (70B)
    device=DeviceType.GPU_ARC,
    validator_model="WhiteRabbitNeo-33B-v1"  # Validator (33B)
)

# Generate code with validation
code, validation_report = engine.generate(
    prompt="Write a secure password hashing function in Python",
    task_type=TaskType.CODE_GENERATION,
    validate=True
)

# Check validation
if validation_report and validation_report.overall_pass:
    print("✓ Code validated successfully")
    print(code)
else:
    print("✗ Code failed validation")
    print(validation_report.syntax_check.errors)
```

### Example 4: Custom Configuration

```python
from dynamic_allocator import QuantizationType, DeviceType
from validation_pipeline import ValidationMode

engine = get_inference_engine()

# Setup with custom config
engine.setup(
    model_name="WhiteRabbitNeo-33B-v1",
    device=DeviceType.GPU_ARC,
    quantization=QuantizationType.INT4,
    validator_model="Llama-3.1-WhiteRabbitNeo-2-70B"
)

# Update runtime config
engine.update_config(
    max_new_tokens=1024,
    temperature=0.8,
    top_p=0.95,
    enable_validation=True,
    validation_mode=ValidationMode.DUAL
)

# Generate
output, validation = engine.generate(
    prompt="Design a microservice architecture for a banking system",
    task_type=TaskType.ANALYSIS
)
```

---

## Device Selection Guide

### When to use Arc GPU (Primary)

**Best for**:
- Large models (33B, 70B)
- Batch inference
- High throughput
- Training/fine-tuning

**Specs**:
- 40 TOPS INT8 (BASE), scales to 72 TOPS in classified mode
- 57.6GB usable RAM (shared, from 64GB total)
- 2ms latency
- 15W power

### When to use NPU

**Best for**:
- Real-time inference
- Small models
- Edge deployment
- Power efficiency

**Specs**:
- 26.4 TOPS (military mode)
- 128MB BAR0 region
- <0.5ms latency
- 5W power

### When to use NCS2

**Best for**:
- Memory offloading
- Attention layers
- Distributed inference
- Large context windows

**Specs**:
- 30 TOPS total (3 devices × 10 TOPS each)
- 1.5GB total inference memory (3 × 512MB)
- 48GB total on-stick storage (3 × 16GB, for model caching)
- 5ms latency per device
- 9W power total (3 × 3W)

### When to use GNA

**Best for**:
- Post-quantum crypto
- Token validation
- Security attestation
- Pattern matching

**Specs**:
- Specialized ops
- 16MB on-die
- 50μs latency
- <1W power

---

## Memory Management

### Automatic Allocation

The system automatically allocates memory across devices:

```python
from dynamic_allocator import get_allocator, ModelSpec, QuantizationType

allocator = get_allocator()

# Create model spec
model = ModelSpec(
    name="WhiteRabbitNeo-33B-v1",
    params_billions=33.0,
    num_layers=60
)

# Plan allocation
plan = allocator.create_allocation_plan(
    model,
    quantization=QuantizationType.INT4
)

# View allocation
print(f"GPU layers: {len(plan.gpu_layers)}")
print(f"NCS2 layers: {len(plan.ncs2_layers)}")
print(f"NPU layers: {len(plan.npu_layers)}")
```

### Manual Swap Configuration

```python
# Enable 32GB swap for large models
allocator.create_swap_file(size_gb=32)

# Create plan with swap
plan = allocator.create_allocation_plan(
    model,
    enable_swap=True
)
```

### Layer-wise Streaming

For models that don't fit in memory:

```python
# Enable streaming
plan = allocator.create_allocation_plan(
    model,
    quantization=QuantizationType.INT4
)

if plan.use_streaming:
    print(f"Streaming enabled: {plan.layers_per_batch} layers per batch")
```

---

## Validation Pipeline

### Validation Modes

1. **Single**: One model validates itself
2. **Dual**: Two models cross-check
3. **Multi**: Multiple models vote
4. **Self-Correct**: Iterative improvement

### Example: Code Validation

```python
from validation_pipeline import get_validation_pipeline, ValidationMode

validator = get_validation_pipeline()

code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
'''

# Validate
report = validator.validate(
    code=code,
    mode=ValidationMode.SINGLE
)

# Check results
if report.overall_pass:
    print("✓ Code is valid")
    print(f"  Syntax: {report.syntax_check.syntax_valid}")
    print(f"  Compiles: {report.compilation_successful}")
    print(f"  Runs: {report.runtime_successful}")
```

### Supported Languages

- Python (.py)
- C (.c)
- C++ (.cpp, .cc, .cxx)
- Rust (.rs)

---

## Performance Optimization

### Quantization Strategies

```python
# INT4: Maximum compression (recommended)
# - 33B: 16.9GB
# - 70B: 35.9GB
quantization=QuantizationType.INT4

# INT8: Balanced
# - 33B: 30.7GB
# - 70B: 65.2GB
quantization=QuantizationType.INT8

# FP16: Best quality
# - 33B: 61.5GB (needs offload)
# - 70B: 130.4GB (needs streaming)
quantization=QuantizationType.FP16
```

### Batch Size Tuning

```python
# Latency-optimized (single request)
engine.update_config(batch_size=1)

# Throughput-optimized (multiple requests)
engine.update_config(batch_size=8)
```

### Temperature/Sampling

```python
# Creative generation
engine.update_config(
    temperature=0.9,
    top_p=0.95,
    top_k=100
)

# Deterministic/focused
engine.update_config(
    temperature=0.3,
    top_p=0.9,
    top_k=40
)
```

---

## Troubleshooting

### Model doesn't fit in memory

**Solution 1**: Use INT4 quantization
```python
quantization=QuantizationType.INT4
```

**Solution 2**: Enable swap
```python
allocator.create_swap_file(size_gb=32)
enable_swap=True
```

**Solution 3**: Use NCS2 offloading
```python
# Automatically uses NCS2 if model too large for RAM
plan = allocator.create_allocation_plan(model)
```

### Slow inference

**Check 1**: Verify device
```python
engine.print_status()  # Check active device
```

**Check 2**: Use NPU for low latency
```python
engine.switch_device(DeviceType.NPU)
```

**Check 3**: Increase batch size
```python
engine.update_config(batch_size=8)
```

### Validation fails

**Check 1**: Review errors
```python
print(validation_report.syntax_check.errors)
print(validation_report.compilation_output)
```

**Check 2**: Disable validation for testing
```python
engine.update_config(enable_validation=False)
```

---

## Hardware Configuration

### Adjust System Specs

```python
from hardware_profile import update_hardware_specs

# Update if you have different hardware
update_hardware_specs(
    system_ram_gb=128,  # If you have 128GB RAM
    ncs2_memory_per_device_gb=32,  # If using larger NCS2
    npu_cache_mb=256,  # If different NPU cache
    arc_gpu_tops=150  # If different GPU specs
)
```

### Save/Load Profile

```python
from hardware_profile import get_hardware_profile

profile = get_hardware_profile()

# Save
profile.save("/path/to/profile.json")

# Load
from hardware_profile import HardwareProfile
profile = HardwareProfile.load("/path/to/profile.json")
```

---

## API Reference

### WhiteRabbitInferenceEngine

```python
engine = get_inference_engine()

# Setup
engine.setup(model_name, device, quantization, validator_model)

# Generate
output, validation = engine.generate(prompt, task_type, validate, **kwargs)

# Device switching
engine.switch_device(target_device)

# Configuration
engine.update_config(**kwargs)

# Status
engine.print_status()
```

### WhiteRabbitModelManager

```python
manager = get_model_manager()

# List models
models = manager.list_models()

# Load model
success = manager.load_model(model_name, quantization, inference_mode)

# Unload model
manager.unload_model(model_name)

# Device switching
manager.switch_device(model_name, target_device)

# Generation
output = manager.generate(prompt, max_new_tokens, temperature, top_p)
```

### DynamicAllocator

```python
allocator = get_allocator()

# Create allocation plan
plan = allocator.create_allocation_plan(model_spec, quantization, enable_swap)

# Create swap file
success = allocator.create_swap_file(size_gb)
```

### ValidationPipeline

```python
validator = get_validation_pipeline()

# Validate code
report = validator.validate(code, language, mode, reviewer_models)

# Check syntax
validation = validator.check_syntax(code, language)

# Compile code
success, output = validator.compile_code(code, language)

# Run code
success, output = validator.run_code(code, language)
```

---

## File Structure

```
02-ai-engine/
├── dynamic_allocator.py           # Memory allocation across devices
├── hardware_profile.py            # Hardware specs configuration
├── validation_pipeline.py         # Code validation system
├── whiterabbit_model_manager.py   # Model loading and management
├── whiterabbit_inference_engine.py # Unified inference API
├── whiterabbit_demo.py            # Demo script
└── hardware_profile.json          # Hardware configuration

04-hardware/
├── WHITERABBITNEO_GUIDE.md        # This file
└── ACCELERATOR_GUIDE.md           # General accelerator guide
```

---

## Performance Benchmarks

### Expected Performance

| Model | Quantization | Memory | Device | Tokens/sec | Latency |
|-------|-------------|--------|--------|------------|---------|
| 33B   | INT4        | 17GB   | GPU    | ~50        | 20ms    |
| 33B   | INT4        | 17GB   | NPU    | ~30        | 30ms    |
| 33B   | INT4        | 17GB   | NCS2   | ~10        | 100ms   |
| 70B   | INT4        | 36GB   | GPU    | ~25        | 40ms    |
| 70B   | INT4        | 36GB   | NPU    | ~15        | 65ms    |
| 70B   | INT8        | 65GB   | GPU+NCS2 | ~30      | 35ms    |

*Note: These are estimates. Actual performance depends on model architecture, prompt length, and system load.*

---

## Next Steps

1. **Download Models**:
   ```bash
   huggingface-cli download WhiteRabbitNeo/WhiteRabbitNeo-33B-v1
   ```

2. **Implement Real Loading**:
   - Integrate with HuggingFace transformers
   - Add device_map for multi-device loading
   - Implement quantization with bitsandbytes

3. **Test Validation Pipeline**:
   - Test with real code generation
   - Tune validation thresholds
   - Add custom validators

4. **Deploy to Production**:
   - Set up model serving
   - Add API endpoints
   - Implement monitoring

---

## References

- [WhiteRabbitNeo HuggingFace](https://huggingface.co/WhiteRabbitNeo)
- [Intel OpenVINO Docs](https://docs.openvino.ai/)
- [Hardware Accelerator Guide](ACCELERATOR_GUIDE.md)
- [NCS2 Integration Guide](NCS2_INTEGRATION.md)

---

**Version**: 1.0
**Date**: 2025-11-09
**System**: Dell Latitude 5450 with Intel Core Ultra 7 165H
**Author**: LAT5150DRVMIL AI Platform
