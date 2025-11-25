#!/usr/bin/env python3
"""
WhiteRabbitNeo Demo Script
===========================
Demonstrates the complete WhiteRabbitNeo inference system with:
- Dynamic allocation across devices
- Runtime device switching
- Validation pipeline
- Multi-model workflows

Usage:
    python3 whiterabbit_demo.py

Author: LAT5150DRVMIL AI Platform
"""

import sys
sys.path.insert(0, '/home/user/LAT5150DRVMIL/02-ai-engine')

from dynamic_allocator import DeviceType, QuantizationType, ModelSpec, get_allocator
from hardware_profile import get_hardware_profile
from validation_pipeline import ValidationMode, get_validation_pipeline, CodeLanguage
from whiterabbit_model_manager import InferenceMode, get_model_manager
from whiterabbit_inference_engine import TaskType, get_inference_engine


def demo_hardware_profile():
    """Demo: Hardware profile and capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 1: Hardware Profile")
    print("=" * 70)

    profile = get_hardware_profile()
    profile.print_summary()

    input("\nPress Enter to continue...")


def demo_model_allocation():
    """Demo: Model allocation planning."""
    print("\n" + "=" * 70)
    print("DEMO 2: Model Allocation Planning")
    print("=" * 70)

    allocator = get_allocator()

    # Test both models
    models = [
        ModelSpec(name="WhiteRabbitNeo-33B-v1", params_billions=33.0, num_layers=60),
        ModelSpec(name="Llama-3.1-WhiteRabbitNeo-2-70B", params_billions=70.0, num_layers=80),
    ]

    for model in models:
        print(f"\nPlanning allocation for {model.name}...")
        plan = allocator.create_allocation_plan(
            model,
            quantization=QuantizationType.INT4,
            enable_swap=True
        )

    input("\nPress Enter to continue...")


def demo_validation_pipeline():
    """Demo: Code validation pipeline."""
    print("\n" + "=" * 70)
    print("DEMO 3: Validation Pipeline")
    print("=" * 70)

    validator = get_validation_pipeline()

    # Test Python code
    python_code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
'''

    print("\n** Testing Python Code **")
    print(python_code)

    report = validator.validate(
        code=python_code,
        language=CodeLanguage.PYTHON,
        mode=ValidationMode.SINGLE
    )

    validator.print_report(report)

    # Test C code
    c_code = '''
#include <stdio.h>

int main() {
    printf("Hello from WhiteRabbitNeo!\\n");
    return 0;
}
'''

    print("\n** Testing C Code **")
    print(c_code)

    report = validator.validate(
        code=c_code,
        language=CodeLanguage.C,
        mode=ValidationMode.SINGLE
    )

    validator.print_report(report)

    input("\nPress Enter to continue...")


def demo_model_manager():
    """Demo: Model manager."""
    print("\n" + "=" * 70)
    print("DEMO 4: Model Manager")
    print("=" * 70)

    manager = get_model_manager()

    print(f"\nAvailable models:")
    for model in manager.list_models():
        print(f"  • {model}")

    print(f"\nAttempting to load WhiteRabbitNeo-33B-v1...")
    print("(This is simulated - actual model loading not implemented)\n")

    success = manager.load_model(
        "WhiteRabbitNeo-33B-v1",
        quantization=QuantizationType.INT4,
        inference_mode=InferenceMode.AUTO
    )

    if success:
        print("\n✓ Model loaded (simulated)")
        manager.print_status()

        # Simulate device switching
        print("\nSwitching to NPU...")
        manager.switch_device("WhiteRabbitNeo-33B-v1", DeviceType.NPU)

        print("\nSwitching to NCS2...")
        manager.switch_device("WhiteRabbitNeo-33B-v1", DeviceType.NCS2)

    input("\nPress Enter to continue...")


def demo_inference_engine():
    """Demo: Unified inference engine."""
    print("\n" + "=" * 70)
    print("DEMO 5: Unified Inference Engine")
    print("=" * 70)

    engine = get_inference_engine()

    print("\n** Engine Status **")
    engine.print_status()

    print("\n** Setting up engine **")
    engine.setup(
        model_name="WhiteRabbitNeo-33B-v1",
        device=DeviceType.GPU_ARC,
        quantization=QuantizationType.INT4
    )

    print("\n** Test Generation (Simulated) **")
    prompt = "Write a Python function to calculate prime numbers:"

    generated, validation = engine.generate(
        prompt=prompt,
        task_type=TaskType.CODE_GENERATION,
        max_new_tokens=256
    )

    print(f"\nPrompt: {prompt}")
    print(f"\nGenerated:\n{generated}")

    input("\nPress Enter to continue...")


def demo_multi_device_workflow():
    """Demo: Multi-device workflow."""
    print("\n" + "=" * 70)
    print("DEMO 6: Multi-Device Workflow")
    print("=" * 70)

    print("""
This demonstrates a workflow using multiple devices:

1. Load model on GPU (Arc Graphics)
   - Primary inference device
   - 40 TOPS INT8 compute (BASE) → 72 TOPS (CLASSIFIED)
   - 57.6GB usable RAM (from 64GB total)

2. Offload attention layers to NCS2
   - 3 devices with 1.5GB total inference memory (3 × 512MB)
   - 48GB total on-stick storage (3 × 16GB, for model caching)
   - 30 TOPS total compute (3 × 10 TOPS)
   - USB 3.0 connection

3. Use NPU for small layers
   - 128MB BAR0 memory-mapped region
   - 26.4 TOPS (MILITARY) → 35 TOPS (CLASSIFIED)
   - Ultra-low latency (<0.5ms)

4. Use GNA for post-quantum crypto
   - Specialized acceleration
   - 5-48x speedup for PQC operations

5. Validation with second model
   - Cross-check generated code
   - Compilation testing
   - Runtime validation

Total System Capability:
- Memory: 64GB system RAM (57.6GB usable), 48GB NCS2 storage
- Compute (MILITARY): 96.4 TOPS (Arc:40 + NPU:26.4 + NCS2:30)
- Compute (CLASSIFIED): ~137 TOPS (Arc:72 + NPU:35 + NCS2:30)
- Latency: <1ms for most operations
- Dynamic performance scaling based on activation level
""")

    input("\nPress Enter to continue...")


def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              WhiteRabbitNeo Inference System Demo                    ║
║              DYNAMIC PERFORMANCE SYSTEM                              ║
║                                                                      ║
║  Dell Latitude 5450 - Intel Core Ultra 7 165H (Meteor Lake-P)       ║
║  64GB System RAM (57.6GB usable)                                    ║
║  96.4 TOPS Military Mode (Arc:40 + NPU:26.4 + NCS2:30)             ║
║  137 TOPS Classified Mode (Arc:72 + NPU:35 + NCS2:30)              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    demos = [
        ("Hardware Profile", demo_hardware_profile),
        ("Model Allocation", demo_model_allocation),
        ("Validation Pipeline", demo_validation_pipeline),
        ("Model Manager", demo_model_manager),
        ("Inference Engine", demo_inference_engine),
        ("Multi-Device Workflow", demo_multi_device_workflow),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user")
            break
        except Exception as e:
            print(f"\nError in demo: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("""
Next Steps:
1. Implement actual model loading (requires HuggingFace transformers)
2. Integrate with OpenVINO for NPU inference
3. Connect to NCS2 devices via USB
4. Deploy to production environment

For interactive use:
    python3 whiterabbit_inference_engine.py

For documentation:
    See: 04-hardware/WHITERABBITNEO_GUIDE.md
""")


if __name__ == "__main__":
    main()
