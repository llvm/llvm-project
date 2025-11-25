#!/usr/bin/env python3
"""
Heterogeneous Execution Engine - NPU/GNA/CPU Routing

Intelligently routes AI workloads to optimal hardware backend:
- NPU: Small, real-time inference (34-49.4 TOPS military mode)
- GNA: Audio processing, low-power continuous inference
- CPU P-cores: Large models, complex reasoning with AVX512
- CPU E-cores: Background tasks, monitoring

Features:
- Automatic backend selection based on workload profile
- Graceful fallback if hardware unavailable
- Performance prediction and optimization
- AVX512 operations pinned to P-cores
"""

import os
import sys
import time
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


class Backend(Enum):
    """Hardware backends"""
    NPU = "NPU"          # Intel NPU (34-49.4 TOPS)
    GNA = "GNA"          # Gaussian Neural Accelerator
    CPU_PCORE = "PCORE"  # Performance cores + AVX512
    CPU_ECORE = "ECORE"  # Efficiency cores
    CPU = "CPU"          # Standard CPU fallback


@dataclass
class WorkloadProfile:
    """Workload characteristics for backend selection"""
    model_size_mb: float
    input_size_kb: float
    is_realtime: bool
    is_continuous: bool
    is_audio: bool
    complexity_score: float  # 0-1, higher = more complex
    latency_requirement_ms: float


@dataclass
class BackendCapability:
    """Backend capability profile"""
    backend: Backend
    available: bool
    tops: float
    latency_ns: float
    power_watts: float
    max_model_size_mb: float
    best_for: List[str]


class HeterogeneousExecutor:
    """
    Heterogeneous execution engine with military NPU/GNA support
    """

    def __init__(self):
        """Initialize executor with hardware detection"""

        print("=" * 70)
        print(" Heterogeneous Execution Engine")
        print("=" * 70)
        print()

        self.backends = self._detect_backends()
        self.performance_history: List[Dict] = []

        # Print available backends
        print("Available Backends:")
        for backend, caps in self.backends.items():
            if caps.available:
                print(f"  ✓ {backend.value}: {caps.tops} TOPS, {caps.latency_ns}ns latency")
            else:
                print(f"  ✗ {backend.value}: Not available (graceful fallback)")

        print()
        print("=" * 70)
        print()

    def _detect_backends(self) -> Dict[Backend, BackendCapability]:
        """Detect all available backends"""

        backends = {}

        # NPU (Military mode: 34-49.4 TOPS)
        npu_available = False
        npu_tops = 0

        if OPENVINO_AVAILABLE:
            try:
                core = ov.Core()
                devices = core.available_devices()
                if 'NPU' in devices:
                    npu_available = True
                    npu_tops = 34  # Conservative, can go up to 49.4 in military mode
            except:
                pass

        backends[Backend.NPU] = BackendCapability(
            backend=Backend.NPU,
            available=npu_available,
            tops=npu_tops,
            latency_ns=500,  # <500ns for small models
            power_watts=3.5,  # Low power
            max_model_size_mb=100,  # Small models only
            best_for=["realtime", "voice", "quick_inference"]
        )

        # GNA (Gaussian Neural Accelerator)
        gna_available = False
        if OPENVINO_AVAILABLE:
            try:
                core = ov.Core()
                if 'GNA' in core.available_devices():
                    gna_available = True
            except:
                pass

        backends[Backend.GNA] = BackendCapability(
            backend=Backend.GNA,
            available=gna_available,
            tops=1.0,  # Lower TOPS but ultra-low power
            latency_ns=1000,
            power_watts=0.5,  # Ultra low power
            max_model_size_mb=10,
            best_for=["audio", "continuous", "low_power"]
        )

        # CPU P-cores (with AVX512)
        try:
            with open('/proc/cpuinfo', 'r') as f:
                avx512 = 'avx512' in f.read()
        except:
            avx512 = False

        # Detect P-cores
        cpu_count = psutil.cpu_count(logical=True)
        p_core_count = min(12, cpu_count)  # Heuristic for 6P+8E

        backends[Backend.CPU_PCORE] = BackendCapability(
            backend=Backend.CPU_PCORE,
            available=avx512,
            tops=1.5 if avx512 else 0.8,  # AVX512 boost
            latency_ns=50000,  # 50μs
            power_watts=25.0,  # Higher power
            max_model_size_mb=10000,  # Large models
            best_for=["complex", "large_model", "batch"]
        )

        # CPU E-cores
        e_core_count = cpu_count - p_core_count if cpu_count > p_core_count else 0

        backends[Backend.CPU_ECORE] = BackendCapability(
            backend=Backend.CPU_ECORE,
            available=e_core_count > 0,
            tops=0.5,
            latency_ns=100000,  # 100μs
            power_watts=8.0,
            max_model_size_mb=1000,
            best_for=["background", "monitoring", "logging"]
        )

        # CPU fallback (always available)
        backends[Backend.CPU] = BackendCapability(
            backend=Backend.CPU,
            available=True,
            tops=0.8,
            latency_ns=75000,
            power_watts=15.0,
            max_model_size_mb=10000,
            best_for=["fallback", "general"]
        )

        return backends

    def select_backend(self, workload: WorkloadProfile) -> Backend:
        """
        Select optimal backend for workload

        Decision tree:
        1. Real-time + small -> NPU (if available)
        2. Audio + continuous -> GNA (if available)
        3. Large + complex -> CPU P-cores with AVX512
        4. Background -> CPU E-cores
        5. Fallback -> CPU
        """

        # Real-time, small model -> NPU
        if (workload.is_realtime and
            workload.model_size_mb < self.backends[Backend.NPU].max_model_size_mb and
            self.backends[Backend.NPU].available):
            return Backend.NPU

        # Audio, continuous -> GNA
        if (workload.is_audio and
            workload.is_continuous and
            workload.model_size_mb < self.backends[Backend.GNA].max_model_size_mb and
            self.backends[Backend.GNA].available):
            return Backend.GNA

        # Large, complex -> P-cores
        if (workload.complexity_score > 0.7 and
            workload.model_size_mb > 100 and
            self.backends[Backend.CPU_PCORE].available):
            return Backend.CPU_PCORE

        # Background tasks -> E-cores
        if (not workload.is_realtime and
            workload.latency_requirement_ms > 1000 and
            self.backends[Backend.CPU_ECORE].available):
            return Backend.CPU_ECORE

        # Fallback to CPU
        return Backend.CPU

    def execute(
        self,
        workload: WorkloadProfile,
        task_fn: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute task on optimal backend

        Args:
            workload: Workload profile
            task_fn: Function to execute
            *args, **kwargs: Arguments for task_fn

        Returns:
            Execution results with performance metrics
        """

        # Select backend
        selected_backend = self.select_backend(workload)

        # Pin to appropriate cores if needed
        if selected_backend == Backend.CPU_PCORE:
            self._pin_to_p_cores()
        elif selected_backend == Backend.CPU_ECORE:
            self._pin_to_e_cores()

        # Execute
        start_time = time.time()

        try:
            result = task_fn(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        execution_time_ms = (time.time() - start_time) * 1000

        # Record performance
        perf_record = {
            "backend": selected_backend.value,
            "workload": {
                "model_size_mb": workload.model_size_mb,
                "realtime": workload.is_realtime,
                "complexity": workload.complexity_score
            },
            "execution_time_ms": execution_time_ms,
            "success": success,
            "timestamp": time.time()
        }
        self.performance_history.append(perf_record)

        return {
            "success": success,
            "result": result,
            "error": error,
            "backend": selected_backend.value,
            "execution_time_ms": execution_time_ms,
            "predicted_latency_ms": self.backends[selected_backend].latency_ns / 1000000,
            "backend_caps": {
                "tops": self.backends[selected_backend].tops,
                "power_watts": self.backends[selected_backend].power_watts
            }
        }

    def _pin_to_p_cores(self):
        """Pin process to P-cores"""
        try:
            # Assume first 12 cores are P-cores (6 physical with HT)
            cpu_count = psutil.cpu_count(logical=True)
            p_cores = list(range(min(12, cpu_count)))

            p = psutil.Process()
            p.cpu_affinity(p_cores)
            print(f"✓ Pinned to P-cores: {p_cores}")
        except Exception as e:
            print(f"⚠️  Failed to pin to P-cores: {e}")

    def _pin_to_e_cores(self):
        """Pin process to E-cores"""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            e_cores = list(range(12, cpu_count)) if cpu_count > 12 else []

            if e_cores:
                p = psutil.Process()
                p.cpu_affinity(e_cores)
                print(f"✓ Pinned to E-cores: {e_cores}")
        except Exception as e:
            print(f"⚠️  Failed to pin to E-cores: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""

        if not self.performance_history:
            return {"total_executions": 0}

        # Group by backend
        by_backend = {}
        for record in self.performance_history:
            backend = record["backend"]
            if backend not in by_backend:
                by_backend[backend] = {
                    "count": 0,
                    "total_time_ms": 0,
                    "successes": 0,
                    "failures": 0
                }

            by_backend[backend]["count"] += 1
            by_backend[backend]["total_time_ms"] += record["execution_time_ms"]
            if record["success"]:
                by_backend[backend]["successes"] += 1
            else:
                by_backend[backend]["failures"] += 1

        # Calculate averages
        for backend, stats in by_backend.items():
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["count"]
            stats["success_rate"] = stats["successes"] / stats["count"]

        return {
            "total_executions": len(self.performance_history),
            "by_backend": by_backend
        }

    def optimize_backend_selection(self):
        """
        Optimize backend selection based on performance history

        Learns which backends perform best for which workload types
        """

        stats = self.get_performance_stats()

        print("\n" + "=" * 70)
        print(" Backend Performance Analysis")
        print("=" * 70 + "\n")

        for backend, data in stats.get("by_backend", {}).items():
            print(f"{backend}:")
            print(f"  Executions: {data['count']}")
            print(f"  Avg Time: {data['avg_time_ms']:.2f}ms")
            print(f"  Success Rate: {data['success_rate']*100:.1f}%")
            print()

        # TODO: Use performance history to adjust selection heuristics
        # This would be a learning component that improves over time


def main():
    """Demo / Test"""

    executor = HeterogeneousExecutor()

    # Test different workload profiles
    test_workloads = [
        # Real-time voice processing -> NPU
        WorkloadProfile(
            model_size_mb=50,
            input_size_kb=100,
            is_realtime=True,
            is_continuous=True,
            is_audio=True,
            complexity_score=0.3,
            latency_requirement_ms=100
        ),

        # Continuous audio -> GNA
        WorkloadProfile(
            model_size_mb=5,
            input_size_kb=10,
            is_realtime=False,
            is_continuous=True,
            is_audio=True,
            complexity_score=0.2,
            latency_requirement_ms=500
        ),

        # Large complex model -> P-cores
        WorkloadProfile(
            model_size_mb=500,
            input_size_kb=1000,
            is_realtime=False,
            is_continuous=False,
            is_audio=False,
            complexity_score=0.9,
            latency_requirement_ms=5000
        ),

        # Background monitoring -> E-cores
        WorkloadProfile(
            model_size_mb=20,
            input_size_kb=50,
            is_realtime=False,
            is_continuous=True,
            is_audio=False,
            complexity_score=0.1,
            latency_requirement_ms=10000
        ),
    ]

    workload_names = ["Voice (NPU)", "Audio (GNA)", "Large Model (P-cores)", "Monitoring (E-cores)"]

    print("\n" + "=" * 70)
    print(" Testing Workload Routing")
    print("=" * 70 + "\n")

    for workload, name in zip(test_workloads, workload_names):
        # Dummy task
        def task():
            time.sleep(0.01)  # Simulate work
            return {"status": "success"}

        result = executor.execute(workload, task)

        print(f"{name}:")
        print(f"  Selected Backend: {result['backend']}")
        print(f"  Execution Time: {result['execution_time_ms']:.2f}ms")
        print(f"  Backend TOPS: {result['backend_caps']['tops']}")
        print(f"  Power: {result['backend_caps']['power_watts']}W")
        print()

    # Show performance stats
    executor.optimize_backend_selection()


if __name__ == "__main__":
    main()
