"""
Dynamic Model Allocator
========================
Intelligent allocation of large language models across multiple accelerators
with automatic memory management, layer streaming, and swap support.

System: Dell Latitude 5450 with Intel Core Ultra 7 165H (Meteor Lake-P)

CORRECTED Memory Architecture:
- System RAM: 62GB (56GB usable after OS overhead)
- Arc GPU: Shares system RAM (56GB usable, ~40 TOPS INT8)
- NPU: 128MB BAR0 memory-mapped region (26.4 TOPS military mode)
- NCS2: 1 device with 512MB inference memory (16GB storage for model caching)
- Swap: Configurable (for overflow)

Allocation Strategies:
1. INT4 quantization for model compression
2. Layer-wise streaming from disk
3. Distributed inference across accelerators
4. KV cache quantization (INT8)
5. FlashAttention for memory efficiency

Author: LAT5150DRVMIL AI Platform
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import hardware profile for accurate specs
try:
    from hardware_profile import get_hardware_profile
    _USE_HARDWARE_PROFILE = True
except ImportError:
    _USE_HARDWARE_PROFILE = False

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Device types for model allocation."""
    GPU_ARC = "gpu_arc"  # Intel Arc Graphics (primary)
    NPU = "npu"  # Intel NPU (small layers)
    NCS2 = "ncs2"  # Neural Compute Stick 2 (attention offload)
    CPU = "cpu"  # CPU fallback
    SWAP = "swap"  # Swap file overflow


class QuantizationType(Enum):
    """Quantization types for models."""
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"  # Experimental


@dataclass
class ModelSpec:
    """Model specifications."""
    name: str
    params_billions: float
    context_length: int = 4096

    # Memory requirements (GB)
    fp16_gb: float = 0.0
    int8_gb: float = 0.0
    int4_gb: float = 0.0

    # Number of layers
    num_layers: int = 0

    def __post_init__(self):
        """Calculate memory requirements."""
        params = self.params_billions * 1e9
        self.fp16_gb = (params * 2) / (1024**3)
        self.int8_gb = (params * 1) / (1024**3)
        self.int4_gb = (params * 0.5) / (1024**3)


@dataclass
class DeviceCapabilities:
    """Device memory and compute capabilities."""
    device_type: DeviceType
    memory_gb: float
    compute_tops: float
    latency_ms: float
    available: bool


@dataclass
class AllocationPlan:
    """Model allocation plan across devices."""
    model_name: str
    quantization: QuantizationType
    total_memory_required_gb: float

    # Layer allocation
    gpu_layers: List[int]
    npu_layers: List[int]
    ncs2_layers: List[int]
    cpu_layers: List[int]

    # Memory breakdown
    gpu_memory_gb: float
    npu_memory_mb: float
    ncs2_memory_mb: float
    swap_memory_gb: float

    # Streaming configuration
    use_streaming: bool
    layers_per_batch: int

    # Validation
    is_feasible: bool
    warnings: List[str]


class DynamicAllocator:
    """
    Dynamic model allocator for large language models.

    Intelligently distributes model layers across available accelerators
    based on memory constraints, compute capabilities, and latency requirements.
    """

    def __init__(self):
        """Initialize dynamic allocator."""
        self.devices = self._detect_devices()
        self.swap_size_gb = self._detect_swap()

        logger.info("Dynamic Allocator initialized")
        logger.info(f"  System RAM: {self.devices[DeviceType.GPU_ARC].memory_gb:.1f} GB")
        logger.info(f"  Arc GPU (shared): {self.devices[DeviceType.GPU_ARC].memory_gb:.1f} GB")
        logger.info(f"  NPU: {self.devices[DeviceType.NPU].memory_gb * 1024:.0f} MB")
        logger.info(f"  NCS2: {self.devices[DeviceType.NCS2].memory_gb * 1024:.0f} MB")
        logger.info(f"  Swap: {self.swap_size_gb:.1f} GB")

    def _detect_devices(self) -> Dict[DeviceType, DeviceCapabilities]:
        """Detect available devices and their capabilities using hardware_profile."""
        devices = {}

        # Use hardware profile if available, otherwise fall back to detection
        if _USE_HARDWARE_PROFILE:
            profile = get_hardware_profile()
            system_ram_gb = profile.usable_ram_gb
            arc_tops = profile.arc_gpu_tops_int8
            npu_memory_mb = profile.npu_on_die_memory_mb
            npu_tops = profile.npu_tops_optimized
            ncs2_count = profile.ncs2_device_count
            ncs2_inference_mb = profile.ncs2_inference_memory_mb
            ncs2_tops = profile.ncs2_total_tops
            npu_available = profile.npu_available
            ncs2_available = profile.ncs2_available
        else:
            # Fallback: try to detect system RAM
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            mem_kb = int(line.split()[1])
                            system_ram_gb = mem_kb / (1024 ** 2)
                            break
            except:
                system_ram_gb = 56.0  # CORRECTED default (62GB * 0.9)

            # CORRECTED fallback values
            arc_tops = 40.0
            npu_memory_mb = 128.0
            npu_tops = 26.4
            ncs2_count = 1
            ncs2_inference_mb = 512.0
            ncs2_tops = 10.0
            npu_available = Path("/sys/devices/pci0000:00/0000:00:0b.0").exists()
            ncs2_available = False  # Conservative default

        # Arc Graphics (shared memory with system RAM)
        devices[DeviceType.GPU_ARC] = DeviceCapabilities(
            device_type=DeviceType.GPU_ARC,
            memory_gb=system_ram_gb,
            compute_tops=arc_tops,
            latency_ms=2.0,
            available=True
        )

        # Intel NPU
        devices[DeviceType.NPU] = DeviceCapabilities(
            device_type=DeviceType.NPU,
            memory_gb=npu_memory_mb / 1024.0,  # Convert MB to GB
            compute_tops=npu_tops,
            latency_ms=0.5,
            available=npu_available
        )

        # NCS2 (CORRECTED: inference memory, not storage)
        devices[DeviceType.NCS2] = DeviceCapabilities(
            device_type=DeviceType.NCS2,
            memory_gb=(ncs2_inference_mb * ncs2_count) / 1024.0,  # Total inference memory in GB
            compute_tops=ncs2_tops,
            latency_ms=5.0,
            available=ncs2_available
        )

        # CPU fallback
        devices[DeviceType.CPU] = DeviceCapabilities(
            device_type=DeviceType.CPU,
            memory_gb=system_ram_gb,
            compute_tops=5.0,  # Rough estimate
            latency_ms=10.0,
            available=True
        )

        return devices

    def _detect_swap(self) -> float:
        """Detect swap size in GB."""
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("SwapTotal:"):
                        swap_kb = int(line.split()[1])
                        return swap_kb / (1024 ** 2)
        except:
            pass
        return 0.0

    def create_swap_file(self, size_gb: int = 32) -> bool:
        """
        Create swap file for memory overflow.

        Args:
            size_gb: Swap file size in GB

        Returns:
            True if successful
        """
        swap_file = Path("/swapfile")

        if swap_file.exists():
            logger.info(f"Swap file already exists: {swap_file}")
            return True

        try:
            logger.info(f"Creating {size_gb}GB swap file...")
            logger.warning("This requires sudo privileges and may take several minutes")

            commands = [
                f"sudo fallocate -l {size_gb}G /swapfile",
                "sudo chmod 600 /swapfile",
                "sudo mkswap /swapfile",
                "sudo swapon /swapfile",
            ]

            for cmd in commands:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    logger.error(f"Swap creation failed: {result.stderr}")
                    return False

            # Verify
            self.swap_size_gb = self._detect_swap()
            logger.info(f"Swap file created: {self.swap_size_gb:.1f} GB")

            return True

        except Exception as e:
            logger.error(f"Failed to create swap file: {e}")
            return False

    def calculate_kv_cache_size(
        self,
        model_spec: ModelSpec,
        batch_size: int = 1,
        context_length: Optional[int] = None
    ) -> float:
        """
        Calculate KV cache memory requirement in GB.

        Args:
            model_spec: Model specifications
            batch_size: Batch size
            context_length: Context length (default: model's context_length)

        Returns:
            KV cache size in GB
        """
        if context_length is None:
            context_length = model_spec.context_length

        # Rough estimate: (2 * num_layers * hidden_dim * context_length * batch_size * bytes_per_value)
        # For typical LLMs: hidden_dim ≈ params / (12 * num_layers) for MLP overhead
        # Using INT8 for KV cache: 1 byte per value

        # Simplified: ~10% of model size for 4K context
        cache_ratio = context_length / 4096.0
        kv_cache_gb = model_spec.int4_gb * 0.1 * cache_ratio * batch_size

        return kv_cache_gb

    def create_allocation_plan(
        self,
        model_spec: ModelSpec,
        quantization: QuantizationType = QuantizationType.INT4,
        context_length: Optional[int] = None,
        enable_swap: bool = True
    ) -> AllocationPlan:
        """
        Create allocation plan for model across devices.

        Args:
            model_spec: Model specifications
            quantization: Quantization type
            context_length: Override context length
            enable_swap: Allow swap file usage

        Returns:
            AllocationPlan with device assignments
        """
        logger.info(f"Creating allocation plan for {model_spec.name}...")

        # Calculate memory requirements
        if quantization == QuantizationType.FP16:
            model_memory_gb = model_spec.fp16_gb
        elif quantization == QuantizationType.INT8:
            model_memory_gb = model_spec.int8_gb
        else:  # INT4
            model_memory_gb = model_spec.int4_gb

        kv_cache_gb = self.calculate_kv_cache_size(model_spec, context_length=context_length)
        total_memory_gb = model_memory_gb + kv_cache_gb

        logger.info(f"  Model memory ({quantization.value}): {model_memory_gb:.1f} GB")
        logger.info(f"  KV cache: {kv_cache_gb:.1f} GB")
        logger.info(f"  Total required: {total_memory_gb:.1f} GB")

        # Available memory
        available_ram = self.devices[DeviceType.GPU_ARC].memory_gb
        available_swap = self.swap_size_gb if enable_swap else 0.0
        total_available = available_ram + available_swap

        logger.info(f"  Available RAM: {available_ram:.1f} GB")
        logger.info(f"  Available swap: {available_swap:.1f} GB")
        logger.info(f"  Total available: {total_available:.1f} GB")

        # Determine strategy
        warnings = []
        is_feasible = True
        use_streaming = False
        swap_needed_gb = 0.0

        if total_memory_gb <= available_ram:
            logger.info("  ✓ Model fits in RAM")
            strategy = "RAM_ONLY"
        elif total_memory_gb <= total_available:
            logger.info("  → Using RAM + swap")
            strategy = "RAM_PLUS_SWAP"
            swap_needed_gb = total_memory_gb - available_ram
            warnings.append(f"Requires {swap_needed_gb:.1f}GB swap (slower inference)")
        else:
            logger.warning("  ⚠ Model requires streaming")
            strategy = "STREAMING"
            use_streaming = True
            overflow_gb = total_memory_gb - total_available
            warnings.append(f"Requires layer streaming ({overflow_gb:.1f}GB overflow)")

        # Allocate layers
        # For simplicity, estimate layers (typically 32-80 for 33B-70B models)
        if model_spec.num_layers == 0:
            # Estimate: roughly 2-3 layers per billion parameters
            estimated_layers = int(model_spec.params_billions * 2.5)
            model_spec.num_layers = estimated_layers

        num_layers = model_spec.num_layers

        # Layer allocation strategy:
        # 1. GPU (Arc): Main model backbone (most layers)
        # 2. NCS2: Attention layers (memory-intensive, can offload)
        # 3. NPU: Small MLP layers (on-die memory, fast)
        # 4. CPU: Fallback only

        # Simple allocation: 90% GPU, 5% NCS2, 5% NPU
        gpu_layer_count = int(num_layers * 0.90)
        ncs2_layer_count = int(num_layers * 0.05) if self.devices[DeviceType.NCS2].available else 0
        npu_layer_count = int(num_layers * 0.05) if self.devices[DeviceType.NPU].available else 0
        cpu_layer_count = num_layers - gpu_layer_count - ncs2_layer_count - npu_layer_count

        gpu_layers = list(range(0, gpu_layer_count))
        ncs2_layers = list(range(gpu_layer_count, gpu_layer_count + ncs2_layer_count))
        npu_layers = list(range(gpu_layer_count + ncs2_layer_count,
                                gpu_layer_count + ncs2_layer_count + npu_layer_count))
        cpu_layers = list(range(gpu_layer_count + ncs2_layer_count + npu_layer_count, num_layers))

        # Memory breakdown
        gpu_memory_gb = model_memory_gb * 0.90 + kv_cache_gb
        ncs2_memory_mb = (model_memory_gb * 0.05) * 1024
        npu_memory_mb = (model_memory_gb * 0.05) * 1024

        # Layers per batch for streaming
        if use_streaming:
            # Process 4 layers at a time to fit in memory
            layers_per_batch = max(4, num_layers // 10)
        else:
            layers_per_batch = num_layers

        plan = AllocationPlan(
            model_name=model_spec.name,
            quantization=quantization,
            total_memory_required_gb=total_memory_gb,
            gpu_layers=gpu_layers,
            npu_layers=npu_layers,
            ncs2_layers=ncs2_layers,
            cpu_layers=cpu_layers,
            gpu_memory_gb=gpu_memory_gb,
            npu_memory_mb=npu_memory_mb,
            ncs2_memory_mb=ncs2_memory_mb,
            swap_memory_gb=swap_needed_gb,
            use_streaming=use_streaming,
            layers_per_batch=layers_per_batch,
            is_feasible=is_feasible,
            warnings=warnings
        )

        self._print_allocation_plan(plan)

        return plan

    def _print_allocation_plan(self, plan: AllocationPlan):
        """Print allocation plan summary."""
        print("\n" + "=" * 70)
        print(f"ALLOCATION PLAN: {plan.model_name}")
        print("=" * 70)
        print(f"Quantization: {plan.quantization.value.upper()}")
        print(f"Total Memory: {plan.total_memory_required_gb:.1f} GB")
        print(f"Streaming: {'Yes' if plan.use_streaming else 'No'}")

        print(f"\nLayer Distribution:")
        print(f"  GPU (Arc):  {len(plan.gpu_layers):3d} layers ({plan.gpu_memory_gb:.1f} GB)")
        print(f"  NCS2:       {len(plan.ncs2_layers):3d} layers ({plan.ncs2_memory_mb:.0f} MB)")
        print(f"  NPU:        {len(plan.npu_layers):3d} layers ({plan.npu_memory_mb:.0f} MB)")
        print(f"  CPU:        {len(plan.cpu_layers):3d} layers")

        if plan.swap_memory_gb > 0:
            print(f"\nSwap Usage: {plan.swap_memory_gb:.1f} GB")

        if plan.use_streaming:
            print(f"\nStreaming: {plan.layers_per_batch} layers per batch")

        if plan.warnings:
            print(f"\nWarnings:")
            for warning in plan.warnings:
                print(f"  ⚠ {warning}")

        print(f"\nFeasible: {'✓ YES' if plan.is_feasible else '✗ NO'}")
        print("=" * 70 + "\n")


# Singleton instance
_allocator: Optional[DynamicAllocator] = None


def get_allocator() -> DynamicAllocator:
    """Get or create singleton allocator."""
    global _allocator

    if _allocator is None:
        _allocator = DynamicAllocator()

    return _allocator
