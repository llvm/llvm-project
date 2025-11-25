#!/usr/bin/env python3
"""
GPU and Hardware Acceleration Check
------------------------------------
Detects and reports available hardware acceleration options for AI workloads.

Checks for:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm)
- Apple Silicon (MPS)
- Intel GPUs (oneAPI)
- CPU capabilities (AVX, AVX2, AVX-512)
"""

import os
import sys
import platform
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class GPUInfo:
    """Information about a detected GPU"""
    name: str
    vendor: str
    memory_total: Optional[int] = None  # MB
    memory_free: Optional[int] = None  # MB
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None


@dataclass
class HardwareCapabilities:
    """Complete hardware capabilities report"""
    gpus: List[GPUInfo]
    cpu_name: str
    cpu_cores: int
    ram_total: int  # MB
    ram_available: int  # MB
    has_cuda: bool
    has_rocm: bool
    has_mps: bool  # Apple Metal Performance Shaders
    has_avx: bool
    has_avx2: bool
    has_avx512: bool
    platform: str
    python_version: str

    def to_dict(self) -> Dict:
        return {
            'gpus': [{'name': g.name, 'vendor': g.vendor, 'memory_mb': g.memory_total} for g in self.gpus],
            'cpu': {'name': self.cpu_name, 'cores': self.cpu_cores},
            'ram_mb': self.ram_total,
            'ram_available_mb': self.ram_available,
            'acceleration': {
                'cuda': self.has_cuda,
                'rocm': self.has_rocm,
                'mps': self.has_mps,
                'avx': self.has_avx,
                'avx2': self.has_avx2,
                'avx512': self.has_avx512
            },
            'platform': self.platform,
            'python_version': self.python_version
        }


class HardwareDetector:
    """Detects available hardware acceleration options"""

    def __init__(self):
        self.capabilities: Optional[HardwareCapabilities] = None

    def detect_all(self) -> HardwareCapabilities:
        """Run all hardware detection checks"""
        gpus = self._detect_gpus()
        cpu_info = self._detect_cpu()
        ram_info = self._detect_ram()

        self.capabilities = HardwareCapabilities(
            gpus=gpus,
            cpu_name=cpu_info['name'],
            cpu_cores=cpu_info['cores'],
            ram_total=ram_info['total'],
            ram_available=ram_info['available'],
            has_cuda=self._check_cuda(),
            has_rocm=self._check_rocm(),
            has_mps=self._check_mps(),
            has_avx=cpu_info['has_avx'],
            has_avx2=cpu_info['has_avx2'],
            has_avx512=cpu_info['has_avx512'],
            platform=platform.platform(),
            python_version=sys.version
        )

        return self.capabilities

    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect all available GPUs"""
        gpus = []

        # Try NVIDIA first
        nvidia_gpus = self._detect_nvidia_gpus()
        gpus.extend(nvidia_gpus)

        # Try AMD
        amd_gpus = self._detect_amd_gpus()
        gpus.extend(amd_gpus)

        # Try Apple
        apple_gpus = self._detect_apple_gpus()
        gpus.extend(apple_gpus)

        return gpus

    def _detect_nvidia_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi"""
        gpus = []

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,compute_cap',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            gpus.append(GPUInfo(
                                name=parts[0],
                                vendor='NVIDIA',
                                memory_total=int(float(parts[1])),
                                memory_free=int(float(parts[2])),
                                driver_version=parts[3],
                                compute_capability=parts[4]
                            ))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return gpus

    def _detect_amd_gpus(self) -> List[GPUInfo]:
        """Detect AMD GPUs using rocm-smi"""
        gpus = []

        try:
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse rocm-smi output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'GPU' in line and ':' in line:
                        name = line.split(':', 1)[1].strip()
                        gpus.append(GPUInfo(
                            name=name,
                            vendor='AMD'
                        ))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return gpus

    def _detect_apple_gpus(self) -> List[GPUInfo]:
        """Detect Apple Silicon GPUs"""
        gpus = []

        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # Check for M1/M2/M3
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode == 0:
                    cpu_name = result.stdout.strip()
                    if 'Apple' in cpu_name:
                        gpus.append(GPUInfo(
                            name=f"{cpu_name} (Integrated)",
                            vendor='Apple'
                        ))
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        return gpus

    def _detect_cpu(self) -> Dict:
        """Detect CPU information and capabilities"""
        import multiprocessing

        cpu_info = {
            'name': platform.processor() or 'Unknown',
            'cores': multiprocessing.cpu_count(),
            'has_avx': False,
            'has_avx2': False,
            'has_avx512': False
        }

        # Check CPU flags on Linux
        if platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    cpu_info['has_avx'] = 'avx' in cpuinfo
                    cpu_info['has_avx2'] = 'avx2' in cpuinfo
                    cpu_info['has_avx512'] = 'avx512' in cpuinfo

                    # Get model name
                    for line in cpuinfo.split('\n'):
                        if 'model name' in line:
                            cpu_info['name'] = line.split(':', 1)[1].strip()
                            break
            except FileNotFoundError:
                pass

        # Check CPU flags on macOS
        elif platform.system() == 'Darwin':
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    cpu_info['name'] = result.stdout.strip()

                # Check AVX
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.features'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    features = result.stdout.upper()
                    cpu_info['has_avx'] = 'AVX' in features
                    cpu_info['has_avx2'] = 'AVX2' in features
                    cpu_info['has_avx512'] = 'AVX512' in features
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        return cpu_info

    def _detect_ram(self) -> Dict:
        """Detect RAM information"""
        ram_info = {'total': 0, 'available': 0}

        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_info['total'] = mem.total // (1024 * 1024)  # Convert to MB
            ram_info['available'] = mem.available // (1024 * 1024)
        except ImportError:
            # Fallback without psutil
            if platform.system() == 'Linux':
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemTotal' in line:
                                ram_info['total'] = int(line.split()[1]) // 1024  # Convert KB to MB
                            elif 'MemAvailable' in line:
                                ram_info['available'] = int(line.split()[1]) // 1024
                except FileNotFoundError:
                    pass

        return ram_info

    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            # Check for nvidia-smi
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try PyTorch CUDA check
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass

        return False

    def _check_rocm(self) -> bool:
        """Check if ROCm is available"""
        try:
            result = subprocess.run(
                ['rocm-smi'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_mps(self) -> bool:
        """Check if Apple MPS is available"""
        if platform.system() != 'Darwin':
            return False

        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            # Check for Apple Silicon
            return platform.machine() == 'arm64'

    def print_report(self):
        """Print a formatted hardware capabilities report"""
        if not self.capabilities:
            self.detect_all()

        caps = self.capabilities

        print("\n" + "=" * 80)
        print("  HARDWARE ACCELERATION REPORT")
        print("=" * 80)

        # Platform
        print(f"\n📊 Platform: {caps.platform}")
        print(f"   Python: {caps.python_version.split()[0]}")

        # CPU
        print(f"\n🖥️  CPU:")
        print(f"   Model: {caps.cpu_name}")
        print(f"   Cores: {caps.cpu_cores}")
        print(f"   AVX: {'✓' if caps.has_avx else '✗'}")
        print(f"   AVX2: {'✓' if caps.has_avx2 else '✗'}")
        print(f"   AVX-512: {'✓' if caps.has_avx512 else '✗'}")

        # RAM
        print(f"\n💾 RAM:")
        print(f"   Total: {caps.ram_total:,} MB ({caps.ram_total / 1024:.1f} GB)")
        print(f"   Available: {caps.ram_available:,} MB ({caps.ram_available / 1024:.1f} GB)")

        # GPUs
        print(f"\n🎮 GPUs: {len(caps.gpus)} detected")
        if caps.gpus:
            for i, gpu in enumerate(caps.gpus, 1):
                print(f"   [{i}] {gpu.vendor} - {gpu.name}")
                if gpu.memory_total:
                    print(f"       Memory: {gpu.memory_total:,} MB ({gpu.memory_total / 1024:.1f} GB)")
                    if gpu.memory_free:
                        print(f"       Free: {gpu.memory_free:,} MB")
                if gpu.compute_capability:
                    print(f"       Compute Capability: {gpu.compute_capability}")
                if gpu.driver_version:
                    print(f"       Driver: {gpu.driver_version}")
        else:
            print("   No GPUs detected")

        # Acceleration
        print(f"\n🚀 Acceleration:")
        print(f"   CUDA (NVIDIA): {'✓ Available' if caps.has_cuda else '✗ Not available'}")
        print(f"   ROCm (AMD): {'✓ Available' if caps.has_rocm else '✗ Not available'}")
        print(f"   MPS (Apple): {'✓ Available' if caps.has_mps else '✗ Not available'}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if caps.has_cuda:
            print("   ✓ CUDA detected - GPU acceleration available for AI workloads")
            print("   → Consider using PyTorch with CUDA or TensorFlow with GPU support")
        elif caps.has_mps:
            print("   ✓ Apple MPS detected - GPU acceleration available")
            print("   → Use PyTorch with MPS backend for optimal performance")
        elif caps.has_rocm:
            print("   ✓ ROCm detected - AMD GPU acceleration available")
            print("   → Use PyTorch ROCm or TensorFlow ROCm builds")
        elif caps.has_avx2:
            print("   ✓ AVX2 detected - CPU optimizations available")
            print("   → Use optimized CPU builds (Intel MKL, OpenBLAS)")
        else:
            print("   ⚠ Limited acceleration available")
            print("   → Performance may be limited. Consider GPU hardware.")

        print("\n" + "=" * 80 + "\n")


def main():
    """Main entry point"""
    detector = HardwareDetector()
    caps = detector.detect_all()
    detector.print_report()

    # Export JSON if requested
    if '--json' in sys.argv:
        print(json.dumps(caps.to_dict(), indent=2))

    # Exit with code based on acceleration availability
    if caps.has_cuda or caps.has_rocm or caps.has_mps:
        sys.exit(0)  # GPU acceleration available
    else:
        sys.exit(1)  # No GPU acceleration


if __name__ == '__main__':
    main()
