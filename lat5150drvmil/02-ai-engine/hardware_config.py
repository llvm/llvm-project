"""
Hardware Configuration and Detection
=====================================
Detects and configures hardware accelerators for LAT5150DRVMIL AI Platform.

Supported Accelerators:
- Intel NCS2 (Neural Compute Stick 2) - Movidius Myriad X VPU
- Intel NPU (Neural Processing Unit) - AI Boost in Meteor Lake
- CUDA GPUs (NVIDIA)
- ROCm GPUs (AMD)

Author: LAT5150DRVMIL AI Platform
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareCapabilities:
    """Hardware capabilities detected on the system."""

    # Intel NCS2 (Movidius Myriad X VPU)
    ncs2_available: bool = False
    ncs2_device_count: int = 0

    # Intel NPU (AI Boost)
    npu_available: bool = False
    npu_tops_int8: float = 0.0

    # CPU Features
    cpu_model: str = ""
    cpu_cores: int = 0
    avx2_available: bool = False
    avx512_available: bool = False

    # GPU
    cuda_available: bool = False
    cuda_devices: List[str] = None
    rocm_available: bool = False

    # Memory
    total_ram_gb: float = 0.0

    def __post_init__(self):
        if self.cuda_devices is None:
            self.cuda_devices = []


class HardwareDetector:
    """Detects available hardware accelerators."""

    @staticmethod
    def detect_ncs2() -> tuple[bool, int]:
        """
        Detect Intel NCS2 devices.

        Returns:
            Tuple of (available, device_count)
        """
        try:
            # Check if kernel driver is loaded
            result = subprocess.run(
                ["lsmod"],
                capture_output=True,
                text=True,
                check=True
            )

            if "movidius_x_vpu" not in result.stdout:
                logger.debug("NCS2 kernel driver not loaded")
                return False, 0

            # Count device nodes
            import glob
            device_paths = glob.glob("/dev/movidius_x_vpu_*")
            device_count = len(device_paths)

            if device_count > 0:
                logger.info(f"Detected {device_count} Intel NCS2 device(s)")
                return True, device_count
            else:
                return False, 0

        except Exception as e:
            logger.debug(f"NCS2 detection failed: {e}")
            return False, 0

    @staticmethod
    def detect_npu() -> tuple[bool, float]:
        """
        Detect Intel NPU (AI Boost).

        Returns:
            Tuple of (available, tops_int8)
        """
        try:
            # Check for Intel AI Boost (Meteor Lake and later)
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

            # Look for Meteor Lake or Arrow Lake CPU
            if "Intel" in cpuinfo and ("Meteor Lake" in cpuinfo or "Core Ultra" in cpuinfo):
                # Intel Core Ultra 7 155H has 11 TOPS INT8
                logger.info("Detected Intel NPU (AI Boost)")
                return True, 11.0

            return False, 0.0

        except Exception as e:
            logger.debug(f"NPU detection failed: {e}")
            return False, 0.0

    @staticmethod
    def detect_cpu_features() -> Dict:
        """
        Detect CPU model and features.

        Returns:
            Dictionary with CPU information
        """
        cpu_info = {
            "model": platform.processor(),
            "cores": os.cpu_count() or 0,
            "avx2": False,
            "avx512": False
        }

        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

            # Check for AVX2
            if "avx2" in cpuinfo:
                cpu_info["avx2"] = True

            # Check for AVX-512
            if "avx512" in cpuinfo:
                cpu_info["avx512"] = True

            # Get CPU model name
            for line in cpuinfo.split("\n"):
                if line.startswith("model name"):
                    cpu_info["model"] = line.split(":")[1].strip()
                    break

        except Exception as e:
            logger.debug(f"CPU feature detection failed: {e}")

        return cpu_info

    @staticmethod
    def detect_cuda() -> tuple[bool, List[str]]:
        """
        Detect NVIDIA CUDA GPUs.

        Returns:
            Tuple of (available, device_list)
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )

            devices = [line.strip() for line in result.stdout.split("\n") if line.strip()]

            if devices:
                logger.info(f"Detected {len(devices)} CUDA GPU(s): {', '.join(devices)}")
                return True, devices

            return False, []

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("CUDA not available")
            return False, []

    @staticmethod
    def detect_rocm() -> bool:
        """
        Detect AMD ROCm support.

        Returns:
            True if ROCm is available
        """
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                check=True
            )

            if "GPU" in result.stdout:
                logger.info("Detected AMD ROCm GPU")
                return True

            return False

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("ROCm not available")
            return False

    @staticmethod
    def detect_memory() -> float:
        """
        Detect total system RAM.

        Returns:
            Total RAM in GB
        """
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Extract memory in kB and convert to GB
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024 * 1024)
                        return round(mem_gb, 2)

        except Exception as e:
            logger.debug(f"Memory detection failed: {e}")

        return 0.0

    @classmethod
    def detect_all(cls) -> HardwareCapabilities:
        """
        Detect all hardware capabilities.

        Returns:
            HardwareCapabilities object
        """
        logger.info("Detecting hardware capabilities...")

        # NCS2
        ncs2_available, ncs2_count = cls.detect_ncs2()

        # NPU
        npu_available, npu_tops = cls.detect_npu()

        # CPU
        cpu_info = cls.detect_cpu_features()

        # CUDA
        cuda_available, cuda_devices = cls.detect_cuda()

        # ROCm
        rocm_available = cls.detect_rocm()

        # Memory
        total_ram = cls.detect_memory()

        caps = HardwareCapabilities(
            ncs2_available=ncs2_available,
            ncs2_device_count=ncs2_count,
            npu_available=npu_available,
            npu_tops_int8=npu_tops,
            cpu_model=cpu_info["model"],
            cpu_cores=cpu_info["cores"],
            avx2_available=cpu_info["avx2"],
            avx512_available=cpu_info["avx512"],
            cuda_available=cuda_available,
            cuda_devices=cuda_devices,
            rocm_available=rocm_available,
            total_ram_gb=total_ram
        )

        # Log capabilities
        logger.info("Hardware Capabilities:")
        logger.info(f"  CPU: {caps.cpu_model} ({caps.cpu_cores} cores)")
        logger.info(f"  AVX2: {caps.avx2_available}, AVX-512: {caps.avx512_available}")
        logger.info(f"  RAM: {caps.total_ram_gb} GB")

        if caps.ncs2_available:
            logger.info(f"  Intel NCS2: {caps.ncs2_device_count} device(s) - ENABLED ✓")
        else:
            logger.info(f"  Intel NCS2: Not detected")

        if caps.npu_available:
            logger.info(f"  Intel NPU: {caps.npu_tops_int8} TOPS INT8 - ENABLED ✓")
        else:
            logger.info(f"  Intel NPU: Not detected")

        if caps.cuda_available:
            logger.info(f"  CUDA: {len(caps.cuda_devices)} GPU(s) - ENABLED ✓")
        else:
            logger.info(f"  CUDA: Not detected")

        if caps.rocm_available:
            logger.info(f"  ROCm: ENABLED ✓")
        else:
            logger.info(f"  ROCm: Not detected")

        return caps


# Global hardware capabilities (lazy initialization)
_hardware_capabilities: Optional[HardwareCapabilities] = None


def get_hardware_capabilities() -> HardwareCapabilities:
    """
    Get hardware capabilities (cached).

    Returns:
        HardwareCapabilities object
    """
    global _hardware_capabilities

    if _hardware_capabilities is None:
        _hardware_capabilities = HardwareDetector.detect_all()

    return _hardware_capabilities


def is_ncs2_available() -> bool:
    """Check if Intel NCS2 is available."""
    return get_hardware_capabilities().ncs2_available


def is_npu_available() -> bool:
    """Check if Intel NPU is available."""
    return get_hardware_capabilities().npu_available


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return get_hardware_capabilities().cuda_available


def get_optimal_accelerator() -> str:
    """
    Get optimal accelerator for current system.

    Returns:
        Accelerator type: "ncs2", "npu", "cuda", "cpu"
    """
    caps = get_hardware_capabilities()

    # Priority: NCS2 > CUDA > NPU > CPU
    # NCS2 is prioritized for dedicated AI inference
    if caps.ncs2_available:
        return "ncs2"
    elif caps.cuda_available:
        return "cuda"
    elif caps.npu_available:
        return "npu"
    else:
        return "cpu"


def print_hardware_summary():
    """Print detailed hardware summary."""
    caps = get_hardware_capabilities()

    print("\n" + "=" * 60)
    print("  LAT5150DRVMIL Hardware Configuration")
    print("=" * 60)
    print()
    print(f"CPU: {caps.cpu_model}")
    print(f"  Cores: {caps.cpu_cores}")
    print(f"  AVX2: {'✓' if caps.avx2_available else '✗'}")
    print(f"  AVX-512: {'✓' if caps.avx512_available else '✗'}")
    print()
    print(f"Memory: {caps.total_ram_gb} GB RAM")
    print()
    print("Hardware Accelerators:")
    print(f"  Intel NCS2: {'✓ ' + str(caps.ncs2_device_count) + ' device(s)' if caps.ncs2_available else '✗ Not available'}")
    print(f"  Intel NPU: {'✓ ' + str(caps.npu_tops_int8) + ' TOPS INT8' if caps.npu_available else '✗ Not available'}")
    print(f"  NVIDIA CUDA: {'✓ ' + str(len(caps.cuda_devices)) + ' GPU(s)' if caps.cuda_available else '✗ Not available'}")
    print(f"  AMD ROCm: {'✓' if caps.rocm_available else '✗ Not available'}")
    print()
    print(f"Optimal Accelerator: {get_optimal_accelerator().upper()}")
    print("=" * 60)
    print()
