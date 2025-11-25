#!/usr/bin/env python3
"""
Device-Aware Smart Router
==========================
Intelligent query routing to optimal hardware devices with NCS2 memory pooling.

Routes queries based on:
- Task complexity (simple → NPU, complex → GPU + NCS2)
- Model size (small → NPU, large → GPU + pooled NCS2)
- Latency requirements (fast → NPU, thorough → GPU)
- Device load balancing (distribute across NCS2 sticks)

Hardware:
- NPU: 26.4 TOPS, 128MB on-die, 0.5ms latency (small/fast tasks)
- GPU Arc: 40 TOPS, 56GB shared RAM, 2ms latency (medium tasks)
- NCS2 × 2: 20 TOPS pooled, 1GB pooled memory (large models, attention)
- When 3rd NCS2 arrives: 30 TOPS, 1.5GB pooled

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
import time

try:
    from hardware_profile import get_hardware_profile, HardwareProfile
    HARDWARE_PROFILE_AVAILABLE = True
except ImportError:
    HARDWARE_PROFILE_AVAILABLE = False

try:
    from whiterabbit_pydantic import (
        WhiteRabbitDevice,
        WhiteRabbitTaskType,
        WhiteRabbitQuantization,
        WHITERABBIT_AVAILABLE,
    )
except ImportError:
    WHITERABBIT_AVAILABLE = False
    # Fallback enums
    class WhiteRabbitDevice(str, Enum):
        SYSTEM = "system"
        NPU = "npu"
        GPU_ARC = "gpu_arc"
        NCS2 = "ncs2"
        AUTO = "auto"

logger = logging.getLogger(__name__)


# ============================================================================
# Device Selection Strategy
# ============================================================================

class TaskComplexity(str, Enum):
    """Task complexity levels"""
    TRIVIAL = "trivial"        # < 50 tokens, simple completion
    SIMPLE = "simple"          # 50-200 tokens, basic generation
    MEDIUM = "medium"          # 200-1000 tokens, code/analysis
    COMPLEX = "complex"        # 1000-4000 tokens, large generation
    MASSIVE = "massive"        # > 4000 tokens, research/documentation


class DeviceStrategy(str, Enum):
    """Device allocation strategies"""
    NPU_ONLY = "npu_only"                  # NPU only (fastest)
    GPU_ONLY = "gpu_only"                  # GPU only (balanced)
    GPU_NCS2_SINGLE = "gpu_ncs2_single"    # GPU + 1 NCS2
    GPU_NCS2_POOLED = "gpu_ncs2_pooled"    # GPU + both NCS2 (pooled memory)
    DISTRIBUTED = "distributed"             # GPU + NPU + all NCS2
    STREAMING = "streaming"                 # GPU + NCS2 + swap (layer streaming)


@dataclass
class DeviceAllocation:
    """Device allocation decision"""
    primary_device: WhiteRabbitDevice
    strategy: DeviceStrategy
    quantization: 'WhiteRabbitQuantization'
    estimated_latency_ms: float
    estimated_throughput_tps: float  # Tokens per second
    devices_used: List[str]
    memory_available_gb: float
    compute_tops: float
    reasoning: str


@dataclass
class NCS2Pool:
    """NCS2 device pool for distributed inference"""
    device_count: int
    pooled_memory_mb: float
    pooled_tops: float
    device_states: List[Dict[str, any]]

    def get_least_loaded(self) -> int:
        """Get index of least loaded NCS2 device"""
        if not self.device_states:
            return 0

        # Find device with lowest current load
        min_load = min(d.get('current_load', 0.0) for d in self.device_states)
        for i, device in enumerate(self.device_states):
            if device.get('current_load', 0.0) == min_load:
                return i

        return 0

    def allocate_task(self, device_index: int, task_memory_mb: float):
        """Allocate task to specific NCS2 device"""
        if device_index < len(self.device_states):
            self.device_states[device_index]['current_load'] += task_memory_mb
            self.device_states[device_index]['task_count'] += 1

    def release_task(self, device_index: int, task_memory_mb: float):
        """Release task from NCS2 device"""
        if device_index < len(self.device_states):
            self.device_states[device_index]['current_load'] = max(
                0, self.device_states[device_index]['current_load'] - task_memory_mb
            )
            self.device_states[device_index]['task_count'] = max(
                0, self.device_states[device_index]['task_count'] - 1
            )


# ============================================================================
# Device-Aware Router
# ============================================================================

class DeviceAwareRouter:
    """
    Intelligent device-aware query router.

    Routes queries to optimal hardware based on:
    - Task complexity (token count, task type)
    - Model size requirements
    - Latency constraints
    - Device availability and load
    """

    def __init__(self):
        """Initialize device-aware router"""
        # Load hardware profile
        if HARDWARE_PROFILE_AVAILABLE:
            self.hardware = get_hardware_profile()
        else:
            # Fallback: create default profile
            self.hardware = type('HardwareProfile', (), {
                'usable_ram_gb': 55.8,
                'arc_gpu_tops_int8': 40.0,
                'npu_available': True,
                'npu_on_die_memory_mb': 128.0,
                'npu_tops_optimized': 26.4,
                'ncs2_available': True,
                'ncs2_device_count': 2,
                'ncs2_inference_memory_mb': 512.0,
                'ncs2_total_tops': 20.0,
            })

        # Initialize NCS2 pool
        if self.hardware.ncs2_available:
            self.ncs2_pool = NCS2Pool(
                device_count=self.hardware.ncs2_device_count,
                pooled_memory_mb=self.hardware.ncs2_inference_memory_mb * self.hardware.ncs2_device_count,
                pooled_tops=self.hardware.ncs2_total_tops,
                device_states=[
                    {'id': i, 'current_load': 0.0, 'task_count': 0}
                    for i in range(self.hardware.ncs2_device_count)
                ]
            )
        else:
            self.ncs2_pool = None

        # Model size thresholds (billions of parameters)
        self.MODEL_SIZES = {
            'tiny': 0.5,      # < 500M params → NPU
            'small': 2.0,     # 500M-2B → NPU or GPU
            'medium': 7.0,    # 2-7B → GPU
            'large': 13.0,    # 7-13B → GPU + NCS2 single
            'xlarge': 33.0,   # 13-33B → GPU + NCS2 pooled
            'huge': 70.0,     # 33-70B → GPU + NCS2 + streaming
        }

        logger.info("Device-Aware Router initialized")
        logger.info(f"  NPU: {'✓' if self.hardware.npu_available else '✗'} ({self.hardware.npu_tops_optimized:.1f} TOPS)")
        logger.info(f"  GPU Arc: ✓ ({self.hardware.arc_gpu_tops_int8:.1f} TOPS)")
        if self.ncs2_pool:
            logger.info(f"  NCS2 Pool: ✓ {self.ncs2_pool.device_count} devices ({self.ncs2_pool.pooled_memory_mb:.0f}MB, {self.ncs2_pool.pooled_tops:.1f} TOPS)")
        else:
            logger.info(f"  NCS2 Pool: ✗ Not available")

    def estimate_complexity(
        self,
        prompt: str,
        task_type: Optional['WhiteRabbitTaskType'] = None,
        max_tokens: int = 512
    ) -> TaskComplexity:
        """
        Estimate task complexity based on prompt and parameters.

        Args:
            prompt: Input prompt
            task_type: Task type hint
            max_tokens: Maximum output tokens

        Returns:
            TaskComplexity level
        """
        prompt_tokens = len(prompt.split())  # Rough estimate
        total_tokens = prompt_tokens + max_tokens

        # Complexity based on total tokens
        if total_tokens < 100:
            return TaskComplexity.TRIVIAL
        elif total_tokens < 500:
            return TaskComplexity.SIMPLE
        elif total_tokens < 2000:
            return TaskComplexity.MEDIUM
        elif total_tokens < 6000:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.MASSIVE

    def estimate_model_size(self, model_name: str) -> float:
        """
        Estimate model size in billions of parameters.

        Args:
            model_name: Model name (e.g., "whiterabbit-neo-33b")

        Returns:
            Size in billions of parameters
        """
        # Extract size from model name
        import re

        # Common patterns: "7b", "33b", "70b", "1.5b"
        match = re.search(r'(\d+(?:\.\d+)?)[bB]', model_name)
        if match:
            return float(match.group(1))

        # Default estimates for known models
        if 'whiterabbit' in model_name.lower():
            return 33.0
        elif 'qwen' in model_name.lower() and 'coder' in model_name.lower():
            return 7.0
        elif 'deepseek' in model_name.lower() and 'r1' in model_name.lower():
            return 1.5
        elif 'codellama' in model_name.lower() and '70' in model_name.lower():
            return 70.0

        # Conservative default
        return 7.0

    def select_device_strategy(
        self,
        complexity: TaskComplexity,
        model_size_b: float,
        latency_sensitive: bool = False
    ) -> DeviceAllocation:
        """
        Select optimal device allocation strategy.

        Args:
            complexity: Task complexity
            model_size_b: Model size in billions of parameters
            latency_sensitive: Whether to prioritize low latency

        Returns:
            DeviceAllocation with device strategy
        """
        # Strategy selection matrix

        # TRIVIAL tasks (< 100 tokens)
        if complexity == TaskComplexity.TRIVIAL:
            if model_size_b <= self.MODEL_SIZES['tiny'] and self.hardware.npu_available:
                # Tiny model on NPU (fastest)
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.NPU,
                    strategy=DeviceStrategy.NPU_ONLY,
                    quantization='INT4',
                    estimated_latency_ms=0.5,
                    estimated_throughput_tps=100.0,
                    devices_used=['NPU'],
                    memory_available_gb=self.hardware.npu_on_die_memory_mb / 1024.0,
                    compute_tops=self.hardware.npu_tops_optimized,
                    reasoning="Trivial task, tiny model → NPU (fastest, 0.5ms latency)"
                )
            else:
                # GPU for slightly larger models
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_ONLY,
                    quantization='INT4',
                    estimated_latency_ms=2.0,
                    estimated_throughput_tps=80.0,
                    devices_used=['GPU'],
                    memory_available_gb=self.hardware.usable_ram_gb,
                    compute_tops=self.hardware.arc_gpu_tops_int8,
                    reasoning="Trivial task, small model → GPU (balanced)"
                )

        # SIMPLE tasks (100-500 tokens)
        elif complexity == TaskComplexity.SIMPLE:
            if model_size_b <= self.MODEL_SIZES['small'] and self.hardware.npu_available and not latency_sensitive:
                # Small model on NPU
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.NPU,
                    strategy=DeviceStrategy.NPU_ONLY,
                    quantization='INT8',
                    estimated_latency_ms=1.0,
                    estimated_throughput_tps=60.0,
                    devices_used=['NPU'],
                    memory_available_gb=self.hardware.npu_on_die_memory_mb / 1024.0,
                    compute_tops=self.hardware.npu_tops_optimized,
                    reasoning="Simple task, small model → NPU (fast, low power)"
                )
            else:
                # GPU for medium models
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_ONLY,
                    quantization='INT4',
                    estimated_latency_ms=3.0,
                    estimated_throughput_tps=50.0,
                    devices_used=['GPU'],
                    memory_available_gb=self.hardware.usable_ram_gb,
                    compute_tops=self.hardware.arc_gpu_tops_int8,
                    reasoning="Simple task, medium model → GPU"
                )

        # MEDIUM tasks (500-2000 tokens)
        elif complexity == TaskComplexity.MEDIUM:
            if model_size_b <= self.MODEL_SIZES['medium']:
                # GPU only for small-medium models
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_ONLY,
                    quantization='INT4',
                    estimated_latency_ms=5.0,
                    estimated_throughput_tps=45.0,
                    devices_used=['GPU'],
                    memory_available_gb=self.hardware.usable_ram_gb,
                    compute_tops=self.hardware.arc_gpu_tops_int8,
                    reasoning="Medium task, 7B model → GPU"
                )
            elif model_size_b <= self.MODEL_SIZES['large'] and self.ncs2_pool:
                # GPU + single NCS2 for attention offload
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_NCS2_SINGLE,
                    quantization='INT4',
                    estimated_latency_ms=7.0,
                    estimated_throughput_tps=40.0,
                    devices_used=['GPU', 'NCS2_0'],
                    memory_available_gb=self.hardware.usable_ram_gb + 0.5,
                    compute_tops=self.hardware.arc_gpu_tops_int8 + 10.0,
                    reasoning="Medium task, 13B model → GPU + NCS2 (attention offload)"
                )
            else:
                # GPU only fallback
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_ONLY,
                    quantization='INT4',
                    estimated_latency_ms=8.0,
                    estimated_throughput_tps=35.0,
                    devices_used=['GPU'],
                    memory_available_gb=self.hardware.usable_ram_gb,
                    compute_tops=self.hardware.arc_gpu_tops_int8,
                    reasoning="Medium task, large model → GPU (no NCS2 available)"
                )

        # COMPLEX tasks (2000-6000 tokens)
        elif complexity == TaskComplexity.COMPLEX:
            if model_size_b <= self.MODEL_SIZES['xlarge'] and self.ncs2_pool and self.ncs2_pool.device_count >= 2:
                # GPU + both NCS2 (pooled memory)
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_NCS2_POOLED,
                    quantization='INT4',
                    estimated_latency_ms=10.0,
                    estimated_throughput_tps=30.0,
                    devices_used=['GPU', 'NCS2_0', 'NCS2_1'],
                    memory_available_gb=self.hardware.usable_ram_gb + (self.ncs2_pool.pooled_memory_mb / 1024.0),
                    compute_tops=self.hardware.arc_gpu_tops_int8 + self.ncs2_pool.pooled_tops,
                    reasoning=f"Complex task, 33B model → GPU + {self.ncs2_pool.device_count}×NCS2 pooled (1GB pooled memory, 60 TOPS)"
                )
            elif self.ncs2_pool:
                # GPU + single NCS2
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_NCS2_SINGLE,
                    quantization='INT4',
                    estimated_latency_ms=12.0,
                    estimated_throughput_tps=25.0,
                    devices_used=['GPU', 'NCS2_0'],
                    memory_available_gb=self.hardware.usable_ram_gb + 0.5,
                    compute_tops=self.hardware.arc_gpu_tops_int8 + 10.0,
                    reasoning="Complex task → GPU + NCS2"
                )
            else:
                # GPU only
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_ONLY,
                    quantization='INT4',
                    estimated_latency_ms=15.0,
                    estimated_throughput_tps=20.0,
                    devices_used=['GPU'],
                    memory_available_gb=self.hardware.usable_ram_gb,
                    compute_tops=self.hardware.arc_gpu_tops_int8,
                    reasoning="Complex task → GPU (no NCS2)"
                )

        # MASSIVE tasks (> 6000 tokens)
        else:  # TaskComplexity.MASSIVE
            if self.ncs2_pool and self.hardware.npu_available:
                # Distributed: GPU + NPU + all NCS2
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.DISTRIBUTED,
                    quantization='INT4',
                    estimated_latency_ms=20.0,
                    estimated_throughput_tps=15.0,
                    devices_used=['GPU', 'NPU', 'NCS2_0', 'NCS2_1'],
                    memory_available_gb=self.hardware.usable_ram_gb + (self.ncs2_pool.pooled_memory_mb / 1024.0),
                    compute_tops=self.hardware.arc_gpu_tops_int8 + self.hardware.npu_tops_optimized + self.ncs2_pool.pooled_tops,
                    reasoning=f"Massive task → Distributed (GPU + NPU + {self.ncs2_pool.device_count}×NCS2, 86.4 TOPS total)"
                )
            elif self.ncs2_pool:
                # GPU + NCS2 pooled
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.GPU_NCS2_POOLED,
                    quantization='INT4',
                    estimated_latency_ms=25.0,
                    estimated_throughput_tps=12.0,
                    devices_used=['GPU', 'NCS2_0', 'NCS2_1'],
                    memory_available_gb=self.hardware.usable_ram_gb + (self.ncs2_pool.pooled_memory_mb / 1024.0),
                    compute_tops=self.hardware.arc_gpu_tops_int8 + self.ncs2_pool.pooled_tops,
                    reasoning="Massive task → GPU + NCS2 pooled"
                )
            else:
                # GPU with streaming
                return DeviceAllocation(
                    primary_device=WhiteRabbitDevice.GPU_ARC,
                    strategy=DeviceStrategy.STREAMING,
                    quantization='INT4',
                    estimated_latency_ms=40.0,
                    estimated_throughput_tps=8.0,
                    devices_used=['GPU'],
                    memory_available_gb=self.hardware.usable_ram_gb,
                    compute_tops=self.hardware.arc_gpu_tops_int8,
                    reasoning="Massive task → GPU with layer streaming (slow)"
                )

    def route_query(
        self,
        prompt: str,
        model_name: str = "whiterabbit-neo-33b",
        task_type: Optional['WhiteRabbitTaskType'] = None,
        max_tokens: int = 512,
        latency_sensitive: bool = False
    ) -> DeviceAllocation:
        """
        Route query to optimal device(s).

        Args:
            prompt: Input prompt
            model_name: Model to use
            task_type: Task type hint
            max_tokens: Maximum output tokens
            latency_sensitive: Prioritize low latency

        Returns:
            DeviceAllocation with routing decision
        """
        # Estimate complexity
        complexity = self.estimate_complexity(prompt, task_type, max_tokens)

        # Estimate model size
        model_size_b = self.estimate_model_size(model_name)

        # Select device strategy
        allocation = self.select_device_strategy(complexity, model_size_b, latency_sensitive)

        logger.info(f"Device routing decision:")
        logger.info(f"  Complexity: {complexity.value}")
        logger.info(f"  Model: {model_name} ({model_size_b:.1f}B params)")
        logger.info(f"  Strategy: {allocation.strategy.value}")
        logger.info(f"  Devices: {', '.join(allocation.devices_used)}")
        logger.info(f"  Compute: {allocation.compute_tops:.1f} TOPS")
        logger.info(f"  Reasoning: {allocation.reasoning}")

        return allocation

    def get_ncs2_allocation(self) -> Optional[Tuple[int, float]]:
        """
        Get least-loaded NCS2 device for task allocation.

        Returns:
            (device_index, available_memory_mb) or None
        """
        if not self.ncs2_pool:
            return None

        device_idx = self.ncs2_pool.get_least_loaded()
        device_state = self.ncs2_pool.device_states[device_idx]
        available_mb = self.hardware.ncs2_inference_memory_mb - device_state['current_load']

        return (device_idx, available_mb)

    def print_status(self):
        """Print current router status"""
        print("\n" + "="*70)
        print("DEVICE-AWARE ROUTER STATUS")
        print("="*70)

        print(f"\nHardware:")
        print(f"  NPU:     {'✓' if self.hardware.npu_available else '✗'} ({self.hardware.npu_tops_optimized:.1f} TOPS, {self.hardware.npu_on_die_memory_mb:.0f}MB)")
        print(f"  GPU Arc: ✓ ({self.hardware.arc_gpu_tops_int8:.1f} TOPS, {self.hardware.usable_ram_gb:.1f}GB)")

        if self.ncs2_pool:
            print(f"  NCS2 Pool:")
            print(f"    Devices: {self.ncs2_pool.device_count}")
            print(f"    Pooled Memory: {self.ncs2_pool.pooled_memory_mb:.0f}MB")
            print(f"    Pooled TOPS: {self.ncs2_pool.pooled_tops:.1f}")
            print(f"    Device States:")
            for i, state in enumerate(self.ncs2_pool.device_states):
                print(f"      NCS2_{i}: Load={state['current_load']:.0f}MB, Tasks={state['task_count']}")
        else:
            print(f"  NCS2: ✗ Not available")

        print(f"\nTotal Compute: {self.hardware.arc_gpu_tops_int8 + self.hardware.npu_tops_optimized + (self.ncs2_pool.pooled_tops if self.ncs2_pool else 0):.1f} TOPS")
        print("="*70 + "\n")


# ============================================================================
# Singleton
# ============================================================================

_router: Optional[DeviceAwareRouter] = None


def get_device_router() -> DeviceAwareRouter:
    """Get or create singleton device router"""
    global _router

    if _router is None:
        _router = DeviceAwareRouter()

    return _router


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Device-Aware Router Test")
    print("="*70)

    router = get_device_router()
    router.print_status()

    # Test cases
    test_cases = [
        ("Hello, world!", "whiterabbit-neo-33b", 50),
        ("Write a Python function to sort a list", "qwen2.5-coder:7b", 200),
        ("Explain quantum computing in detail", "whiterabbit-neo-33b", 1000),
        ("Generate a complete REST API with authentication", "whiterabbit-neo-33b", 2000),
        ("Write comprehensive documentation for a 10,000 line codebase", "whiterabbit-neo-33b", 8000),
    ]

    print("\nTest Cases:")
    print("="*70)

    for i, (prompt, model, max_tokens) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {prompt[:60]}...")
        print(f"  Model: {model}, Max Tokens: {max_tokens}")

        allocation = router.route_query(prompt, model, max_tokens=max_tokens)

        print(f"  → Strategy: {allocation.strategy.value}")
        print(f"  → Devices: {', '.join(allocation.devices_used)}")
        print(f"  → Latency: {allocation.estimated_latency_ms:.1f}ms")
        print(f"  → Throughput: {allocation.estimated_throughput_tps:.1f} tokens/sec")
        print(f"  → Reasoning: {allocation.reasoning}")

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
