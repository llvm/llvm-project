#!/usr/bin/env python3
"""
Dynamic Hardware Resource Allocator

Intelligently allocates hardware resources based on:
1. UMA memory availability (44-52 GiB pool)
2. Task type (inference vs training vs RAG)
3. Model size and batch size
4. Current system load

Dell Latitude 5450 MIL-SPEC Hardware:
- UMA Pool: 62.26 GiB total (44-48 GiB safe, 50-52 GiB aggressive)
- Intel iGPU (Meteor Lake, UMA shared memory)
- Intel NPU (49.4 TOPS INT8, inference only)
- Intel NCS2 (3x sticks, 3-9 TOPS with custom driver)
- AVX-512 (P-cores 0-5 only)

Key Features:
- Dynamic batch size adjustment
- Memory pressure detection
- Intelligent model placement (NPU/GPU/CPU)
- Task-specific optimization
- Real-time resource monitoring
"""

import os
import psutil
import torch
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Available compute devices"""
    NPU = "npu"          # Intel NPU (INT8 inference only)
    GPU = "gpu"          # Intel iGPU with UMA
    CPU = "cpu"          # CPU fallback
    NCS2 = "ncs2"        # Neural Compute Stick 2
    AVX512 = "avx512"    # AVX-512 on P-cores


class TaskType(Enum):
    """Task types with different resource requirements"""
    INFERENCE = "inference"           # Low memory, high throughput
    TRAINING = "training"             # High memory, sustained load
    RAG_EMBEDDING = "rag_embedding"   # Batch embedding generation
    RAG_SEARCH = "rag_search"         # Vector similarity search
    DPO = "dpo"                       # DPO training (high memory)
    PPO = "ppo"                       # PPO training (very high memory)


@dataclass
class MemoryStats:
    """Current memory statistics"""
    total_ram: float          # Total system RAM (GiB)
    available_ram: float      # Available RAM (GiB)
    uma_total: float          # Total UMA pool (GiB)
    uma_stolen: float         # Fixed stolen VRAM (GiB)
    uma_used: float           # Current GPU usage (GiB)
    uma_available: float      # Available for GPU (GiB)
    safe_budget: float        # Safe working budget (GiB)
    aggressive_budget: float  # Aggressive budget (GiB)

    @property
    def utilization(self) -> float:
        """Memory utilization percentage"""
        return (self.uma_used / self.uma_total) * 100


@dataclass
class ResourceAllocation:
    """Resource allocation decision"""
    device: DeviceType
    batch_size: int
    num_workers: int
    precision: str           # "fp32", "fp16", "bf16", "int8"
    gradient_accumulation: int
    max_memory_gb: float
    reason: str


class DynamicResourceAllocator:
    """
    Dynamically allocate hardware resources based on task and availability
    """

    def __init__(
        self,
        safe_mode: bool = True,  # Use conservative memory budget
        monitor_interval: float = 1.0  # Memory monitoring interval (seconds)
    ):
        self.safe_mode = safe_mode
        self.monitor_interval = monitor_interval

        # Hardware capabilities
        self.has_npu = self._check_npu()
        self.has_gpu = self._check_gpu()
        self.has_ncs2 = self._check_ncs2()
        self.has_avx512 = self._check_avx512()

        # Memory thresholds (GiB)
        self.SAFE_BUDGET = 48.0      # Conservative: leave 14 GiB for OS
        self.AGGRESSIVE_BUDGET = 52.0  # Aggressive: leave 10 GiB for OS
        self.CRITICAL_THRESHOLD = 8.0  # Never go below 8 GiB free

        logger.info("=" * 80)
        logger.info("  Dynamic Resource Allocator")
        logger.info("=" * 80)
        logger.info(f"Hardware detected:")
        logger.info(f"  NPU: {self.has_npu}")
        logger.info(f"  GPU (UMA): {self.has_gpu}")
        logger.info(f"  NCS2: {self.has_ncs2}")
        logger.info(f"  AVX-512: {self.has_avx512}")
        logger.info(f"Safe mode: {self.safe_mode}")

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics

        Reads from /proc/meminfo and Intel GPU sysfs
        """
        # System memory
        mem = psutil.virtual_memory()
        total_ram = mem.total / (1024**3)  # GiB
        available_ram = mem.available / (1024**3)

        # UMA pool (Intel iGPU)
        uma_total = total_ram  # UMA uses system RAM
        uma_stolen = 0.125     # 128 MiB fixed

        # Try to read actual GPU usage from sysfs
        uma_used = self._get_gpu_memory_usage()

        # Calculate available
        uma_available = available_ram

        # Budgets
        safe_budget = self.SAFE_BUDGET
        aggressive_budget = self.AGGRESSIVE_BUDGET

        return MemoryStats(
            total_ram=total_ram,
            available_ram=available_ram,
            uma_total=uma_total,
            uma_stolen=uma_stolen,
            uma_used=uma_used,
            uma_available=uma_available,
            safe_budget=safe_budget,
            aggressive_budget=aggressive_budget
        )

    def _get_gpu_memory_usage(self) -> float:
        """
        Get current GPU memory usage from Intel GPU sysfs

        Returns usage in GiB
        """
        try:
            # Try to read from Intel GPU debugfs
            # Path: /sys/kernel/debug/dri/0/i915_gem_objects
            result = subprocess.run(
                ["cat", "/sys/kernel/debug/dri/0/i915_gem_objects"],
                capture_output=True,
                text=True,
                timeout=1
            )

            if result.returncode == 0:
                # Parse output for memory usage
                for line in result.stdout.split('\n'):
                    if 'shrinkable_now' in line.lower():
                        # Extract memory value
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'shrinkable_now' in part.lower() and i + 1 < len(parts):
                                # Convert to GiB (assuming bytes)
                                value_str = parts[i + 1].replace(',', '')
                                try:
                                    bytes_val = float(value_str)
                                    return bytes_val / (1024**3)
                                except ValueError:
                                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass

        # Fallback: estimate from PyTorch
        if torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated() / (1024**3)
            except:
                pass

        # Default estimate
        return 0.9  # GiB

    def allocate_for_task(
        self,
        task_type: TaskType,
        model_size_gb: Optional[float] = None,
        preferred_batch_size: Optional[int] = None
    ) -> ResourceAllocation:
        """
        Allocate resources for a specific task

        Args:
            task_type: Type of task (inference, training, RAG, etc.)
            model_size_gb: Estimated model size in GiB
            preferred_batch_size: Preferred batch size (will be adjusted)

        Returns:
            ResourceAllocation with device, batch size, precision, etc.
        """
        stats = self.get_memory_stats()

        logger.info(f"\nAllocating resources for task: {task_type.value}")
        logger.info(f"  Available memory: {stats.uma_available:.2f} GiB")
        logger.info(f"  Current GPU usage: {stats.uma_used:.2f} GiB")
        logger.info(f"  Memory utilization: {stats.utilization:.1f}%")

        # Determine budget
        budget = stats.safe_budget if self.safe_mode else stats.aggressive_budget

        # Adjust for current usage
        available_for_task = min(stats.uma_available, budget) - stats.uma_used

        # Check critical threshold
        if stats.uma_available < self.CRITICAL_THRESHOLD:
            logger.warning(f"⚠️  Low memory: {stats.uma_available:.2f} GiB available")
            return self._allocate_minimal(task_type, available_for_task)

        # Task-specific allocation
        if task_type == TaskType.INFERENCE:
            return self._allocate_inference(available_for_task, model_size_gb)

        elif task_type == TaskType.TRAINING:
            return self._allocate_training(available_for_task, model_size_gb, preferred_batch_size)

        elif task_type == TaskType.DPO:
            return self._allocate_dpo(available_for_task, model_size_gb, preferred_batch_size)

        elif task_type == TaskType.PPO:
            return self._allocate_ppo(available_for_task, model_size_gb, preferred_batch_size)

        elif task_type == TaskType.RAG_EMBEDDING:
            return self._allocate_rag_embedding(available_for_task)

        elif task_type == TaskType.RAG_SEARCH:
            return self._allocate_rag_search(available_for_task)

        else:
            return self._allocate_default(available_for_task)

    def _allocate_inference(
        self,
        available_gb: float,
        model_size_gb: Optional[float]
    ) -> ResourceAllocation:
        """Allocate for inference (prefer NPU for efficiency)"""

        # Prefer NPU for small models (INT8 only)
        if self.has_npu and (model_size_gb is None or model_size_gb < 4.0):
            return ResourceAllocation(
                device=DeviceType.NPU,
                batch_size=32,
                num_workers=4,
                precision="int8",
                gradient_accumulation=1,
                max_memory_gb=2.0,
                reason="NPU for efficient INT8 inference"
            )

        # GPU for larger models or FP16
        if self.has_gpu and available_gb >= 8.0:
            batch_size = min(64, int(available_gb / 0.5))  # ~0.5 GiB per batch
            return ResourceAllocation(
                device=DeviceType.GPU,
                batch_size=batch_size,
                num_workers=6,
                precision="bf16",
                gradient_accumulation=1,
                max_memory_gb=min(available_gb * 0.8, 16.0),
                reason=f"GPU with large batch size ({batch_size}) for throughput"
            )

        # CPU fallback
        return ResourceAllocation(
            device=DeviceType.CPU,
            batch_size=8,
            num_workers=4,
            precision="fp32",
            gradient_accumulation=1,
            max_memory_gb=4.0,
            reason="CPU fallback"
        )

    def _allocate_training(
        self,
        available_gb: float,
        model_size_gb: Optional[float],
        preferred_batch_size: Optional[int]
    ) -> ResourceAllocation:
        """Allocate for training (maximize batch size)"""

        if not self.has_gpu:
            raise RuntimeError("Training requires GPU (UMA)")

        # Estimate memory requirements
        # Training memory = model + gradients + optimizer states + activations
        # Rule of thumb: 4x model size for Adam optimizer
        model_size = model_size_gb or 2.7  # Default: Phi-2
        training_overhead = model_size * 4.0

        # Calculate max batch size
        remaining_after_model = available_gb - training_overhead

        if remaining_after_model < 4.0:
            # Low memory: use gradient accumulation
            batch_size = 4
            grad_accum = preferred_batch_size // batch_size if preferred_batch_size else 8
            reason = f"Low memory: batch_size={batch_size} with grad_accum={grad_accum}"
        else:
            # Plenty of memory: use large batches
            # Estimate: ~1 GiB per batch for 2.7B model
            memory_per_batch = model_size * 0.4
            max_batch_size = int(remaining_after_model / memory_per_batch)
            batch_size = min(max_batch_size, preferred_batch_size or 32, 64)
            grad_accum = 1
            reason = f"High memory: batch_size={batch_size} (no grad accumulation needed)"

        return ResourceAllocation(
            device=DeviceType.GPU,
            batch_size=batch_size,
            num_workers=6,
            precision="bf16",
            gradient_accumulation=grad_accum,
            max_memory_gb=available_gb * 0.9,
            reason=reason
        )

    def _allocate_dpo(
        self,
        available_gb: float,
        model_size_gb: Optional[float],
        preferred_batch_size: Optional[int]
    ) -> ResourceAllocation:
        """Allocate for DPO training (needs policy + reference model)"""

        if not self.has_gpu:
            raise RuntimeError("DPO training requires GPU (UMA)")

        # DPO memory = 2x model (policy + reference) + optimizer + activations
        model_size = model_size_gb or 2.7
        dpo_overhead = model_size * 2.0 * 4.0  # 2 models * 4x for optimizer

        remaining = available_gb - dpo_overhead

        if remaining < 4.0:
            # Low memory: small batches with gradient accumulation
            batch_size = 2
            grad_accum = preferred_batch_size // batch_size if preferred_batch_size else 8
            reason = f"DPO with limited memory: batch_size={batch_size}, grad_accum={grad_accum}"
        else:
            # Good memory: larger batches
            memory_per_batch = model_size * 0.8  # DPO is more memory-intensive
            max_batch_size = int(remaining / memory_per_batch)
            batch_size = min(max_batch_size, preferred_batch_size or 16, 32)
            grad_accum = 1
            reason = f"DPO with good memory: batch_size={batch_size}"

        return ResourceAllocation(
            device=DeviceType.GPU,
            batch_size=batch_size,
            num_workers=6,
            precision="bf16",
            gradient_accumulation=grad_accum,
            max_memory_gb=available_gb * 0.9,
            reason=reason
        )

    def _allocate_ppo(
        self,
        available_gb: float,
        model_size_gb: Optional[float],
        preferred_batch_size: Optional[int]
    ) -> ResourceAllocation:
        """Allocate for PPO training (very memory intensive)"""

        if not self.has_gpu:
            raise RuntimeError("PPO training requires GPU (UMA)")

        # PPO memory = policy + value + reference + rollout buffer
        model_size = model_size_gb or 2.7
        ppo_overhead = model_size * 3.0 * 4.0  # 3 models * 4x for optimizer

        remaining = available_gb - ppo_overhead

        if remaining < 8.0:
            batch_size = 4
            grad_accum = preferred_batch_size // batch_size if preferred_batch_size else 8
            reason = f"PPO with limited memory: batch_size={batch_size}, grad_accum={grad_accum}"
        else:
            memory_per_batch = model_size * 1.0
            max_batch_size = int(remaining / memory_per_batch)
            batch_size = min(max_batch_size, preferred_batch_size or 16, 32)
            grad_accum = 1
            reason = f"PPO with good memory: batch_size={batch_size}"

        return ResourceAllocation(
            device=DeviceType.GPU,
            batch_size=batch_size,
            num_workers=6,
            precision="bf16",
            gradient_accumulation=grad_accum,
            max_memory_gb=available_gb * 0.9,
            reason=reason
        )

    def _allocate_rag_embedding(self, available_gb: float) -> ResourceAllocation:
        """Allocate for RAG embedding generation (batch processing)"""

        # Prefer NPU for efficiency
        if self.has_npu and available_gb >= 4.0:
            return ResourceAllocation(
                device=DeviceType.NPU,
                batch_size=64,
                num_workers=4,
                precision="int8",
                gradient_accumulation=1,
                max_memory_gb=4.0,
                reason="NPU for efficient batch embedding (INT8)"
            )

        # GPU with large batches
        if self.has_gpu and available_gb >= 8.0:
            batch_size = min(128, int(available_gb / 0.2))  # ~0.2 GiB per batch
            return ResourceAllocation(
                device=DeviceType.GPU,
                batch_size=batch_size,
                num_workers=6,
                precision="bf16",
                gradient_accumulation=1,
                max_memory_gb=min(available_gb * 0.8, 16.0),
                reason=f"GPU with very large batch size ({batch_size}) for embedding"
            )

        # CPU fallback
        return ResourceAllocation(
            device=DeviceType.CPU,
            batch_size=32,
            num_workers=4,
            precision="fp32",
            gradient_accumulation=1,
            max_memory_gb=4.0,
            reason="CPU for embedding"
        )

    def _allocate_rag_search(self, available_gb: float) -> ResourceAllocation:
        """Allocate for RAG vector search (prefer AVX-512)"""

        # Prefer AVX-512 for maximum speed
        if self.has_avx512:
            return ResourceAllocation(
                device=DeviceType.AVX512,
                batch_size=1,  # Single query at a time
                num_workers=6,  # 6 P-cores
                precision="fp32",
                gradient_accumulation=1,
                max_memory_gb=8.0,
                reason="AVX-512 on P-cores for 5x speedup"
            )

        # CPU fallback
        return ResourceAllocation(
            device=DeviceType.CPU,
            batch_size=1,
            num_workers=4,
            precision="fp32",
            gradient_accumulation=1,
            max_memory_gb=4.0,
            reason="CPU for vector search"
        )

    def _allocate_minimal(self, task_type: TaskType, available_gb: float) -> ResourceAllocation:
        """Minimal allocation under memory pressure"""
        logger.warning("⚠️  Memory pressure detected - using minimal allocation")

        return ResourceAllocation(
            device=DeviceType.CPU,
            batch_size=1,
            num_workers=1,
            precision="fp32",
            gradient_accumulation=16,
            max_memory_gb=min(available_gb * 0.5, 2.0),
            reason="Minimal allocation due to memory pressure"
        )

    def _allocate_default(self, available_gb: float) -> ResourceAllocation:
        """Default allocation"""
        return ResourceAllocation(
            device=DeviceType.GPU if self.has_gpu else DeviceType.CPU,
            batch_size=8,
            num_workers=4,
            precision="bf16" if self.has_gpu else "fp32",
            gradient_accumulation=2,
            max_memory_gb=min(available_gb * 0.7, 16.0),
            reason="Default allocation"
        )

    def _check_npu(self) -> bool:
        """Check if Intel NPU is available"""
        # Check for OpenVINO NPU device
        try:
            result = subprocess.run(
                ["python3", "-c", "from openvino.runtime import Core; print('NPU' in Core().available_devices)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "True" in result.stdout
        except:
            return False

    def _check_gpu(self) -> bool:
        """Check if Intel GPU is available"""
        return torch.cuda.is_available() or os.path.exists("/dev/dri/renderD128")

    def _check_ncs2(self) -> bool:
        """Check if Intel NCS2 is available"""
        # Check for MYRIAD devices
        try:
            result = subprocess.run(
                ["lsusb"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return "Movidius" in result.stdout or "Myriad" in result.stdout
        except:
            return False

    def _check_avx512(self) -> bool:
        """Check if AVX-512 is available"""
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                return "avx512f" in cpuinfo
        except:
            return False

    def monitor_memory(self, duration_seconds: float = 10.0):
        """
        Monitor memory usage over time

        Args:
            duration_seconds: How long to monitor
        """
        print("\n" + "=" * 80)
        print("  Memory Monitoring")
        print("=" * 80)

        start_time = time.time()
        samples = []

        try:
            while time.time() - start_time < duration_seconds:
                stats = self.get_memory_stats()
                samples.append(stats)

                print(f"\r  RAM: {stats.available_ram:5.2f} GiB | "
                      f"GPU: {stats.uma_used:5.2f} GiB | "
                      f"Util: {stats.utilization:5.1f}%", end="")

                time.sleep(self.monitor_interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")

        print("\n" + "=" * 80)

        # Statistics
        if samples:
            avg_available = sum(s.uma_available for s in samples) / len(samples)
            avg_used = sum(s.uma_used for s in samples) / len(samples)
            max_used = max(s.uma_used for s in samples)

            print(f"\nStatistics ({len(samples)} samples):")
            print(f"  Avg available: {avg_available:.2f} GiB")
            print(f"  Avg GPU usage: {avg_used:.2f} GiB")
            print(f"  Peak GPU usage: {max_used:.2f} GiB")


def main():
    """Test the dynamic resource allocator"""
    allocator = DynamicResourceAllocator(safe_mode=True)

    # Get memory stats
    stats = allocator.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"  Total RAM: {stats.total_ram:.2f} GiB")
    print(f"  Available: {stats.uma_available:.2f} GiB")
    print(f"  GPU usage: {stats.uma_used:.2f} GiB")
    print(f"  Safe budget: {stats.safe_budget:.2f} GiB")
    print(f"  Aggressive budget: {stats.aggressive_budget:.2f} GiB")

    # Test allocations
    print("\n" + "=" * 80)
    print("  Resource Allocation Tests")
    print("=" * 80)

    tasks = [
        (TaskType.INFERENCE, None, None),
        (TaskType.TRAINING, 2.7, 16),
        (TaskType.DPO, 2.7, 16),
        (TaskType.PPO, 2.7, 16),
        (TaskType.RAG_EMBEDDING, None, None),
        (TaskType.RAG_SEARCH, None, None),
    ]

    for task_type, model_size, batch_size in tasks:
        allocation = allocator.allocate_for_task(task_type, model_size, batch_size)
        print(f"\n{task_type.value}:")
        print(f"  Device: {allocation.device.value}")
        print(f"  Batch size: {allocation.batch_size}")
        print(f"  Precision: {allocation.precision}")
        print(f"  Grad accum: {allocation.gradient_accumulation}")
        print(f"  Max memory: {allocation.max_memory_gb:.2f} GiB")
        print(f"  Reason: {allocation.reason}")

    # Monitor for 5 seconds
    print("\n" + "=" * 80)
    allocator.monitor_memory(duration_seconds=5.0)


if __name__ == "__main__":
    main()
