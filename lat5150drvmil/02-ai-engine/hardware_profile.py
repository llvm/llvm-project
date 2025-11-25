"""
Hardware Profile Configuration
===============================
Configurable hardware specifications for accurate resource allocation.

This module provides a centralized hardware profile that can be adjusted
to match the actual system specifications.

Author: LAT5150DRVMIL AI Platform
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Complete hardware profile."""

    # System Memory (CLARIFIED: 64GB physical, 2GB reserved by DSMIL firmware)
    physical_ram_gb: float = 64.0  # Physical RAM installed
    dsmil_reserved_gb: float = 2.0  # Reserved by DSMIL firmware (hidden from OS)
    system_ram_gb: float = 62.0  # OS-visible RAM (64 - 2 DSMIL reserve)
    usable_ram_gb: float = 55.8  # 90% of 62GB for applications

    # Intel Arc Graphics (Meteor Lake iGPU - DYNAMIC PERFORMANCE)
    arc_gpu_available: bool = True
    arc_gpu_shared_ram: bool = True  # Shares system RAM
    arc_gpu_usable_gb: float = 55.8  # Can use 90% of OS-visible RAM
    arc_gpu_tops_int8: float = 40.0  # BASE: ~40 TOPS INT8 (DSMIL docs, can scale to 72 TOPS classified)
    arc_gpu_compute_units: int = 128  # Meteor Lake Arc iGPU

    # Intel NPU (Neural Processing Unit - DYNAMIC PERFORMANCE)
    npu_available: bool = True
    npu_pci_id: str = "0000:00:0b.0"
    npu_on_die_memory_mb: float = 128.0  # 128MB BAR0 memory-mapped region
    npu_tops_int8: float = 11.0  # STANDARD: 11 TOPS baseline
    npu_tops_optimized: float = 26.4  # MILITARY: 26.4 TOPS (2.4x boost, DSMIL docs)
    npu_tops_classified: float = 35.0  # CLASSIFIED: ~35 TOPS (with classified pathways)
    npu_num_tiles: int = 2
    npu_streams_per_tile: int = 2

    # Intel GNA (Gaussian & Neural Accelerator)
    gna_available: bool = True
    gna_pci_id: str = "0000:00:08.0"
    gna_on_die_memory_mb: float = 16.0
    gna_power_w: float = 1.0

    # Intel NCS2 (Neural Compute Stick 2)
    ncs2_available: bool = True
    ncs2_device_count: int = 2  # CURRENT: 2 devices installed, 3rd in mail
    ncs2_storage_per_device_gb: float = 16.0  # 16GB on-stick STORAGE per device (for model caching)
    ncs2_inference_memory_mb: float = 512.0  # ~512MB actual inference memory per device
    ncs2_tops_per_device: float = 10.0  # 10 TOPS per device (Movidius Myriad X, DSMIL docs line 79)
    ncs2_total_tops: float = 20.0  # CURRENT: 2 devices × 10 TOPS = 20 TOPS (will be 30 when 3rd arrives)

    # Military NPU (if available)
    military_npu_available: bool = False
    military_npu_cache_mb: float = 128.0  # User-specified
    military_npu_tops: float = 100.0

    # CPU
    cpu_cores: int = 16
    cpu_p_cores: int = 6  # Performance cores (0-5)
    cpu_e_cores: int = 10  # Efficiency cores (6-15)
    cpu_avx512_available: bool = True  # On P-cores
    cpu_avx2_available: bool = True

    # Storage
    swap_available: bool = False
    swap_size_gb: float = 0.0

    # Total compute (DYNAMIC - scales with activation level and NCS2 count)
    total_system_tops: float = 86.4  # CURRENT (MILITARY + 2 NCS2): Arc:40 + NPU:26.4 + NCS2:20
    # When 3rd NCS2 arrives: 96.4 TOPS (Arc:40 + NPU:26.4 + NCS2:30)
    # CLASSIFIED MODE (2 NCS2): ~127 TOPS (Arc:72 + NPU:35 + NCS2:20)
    # CLASSIFIED MODE (3 NCS2): ~137 TOPS (Arc:72 + NPU:35 + NCS2:30)

    # Profile metadata
    profile_name: str = "Dell Latitude 5450 - Intel Core Ultra 7 165H"
    last_updated: str = ""

    def __post_init__(self):
        """Update calculated fields."""
        # Calculate totals
        self.ncs2_total_tops = self.ncs2_tops_per_device * self.ncs2_device_count

        total_tops = 0.0
        if self.ncs2_available:
            total_tops += self.ncs2_total_tops
        if self.npu_available:
            total_tops += self.npu_tops_optimized
        if self.arc_gpu_available:
            total_tops += self.arc_gpu_tops_int8
        if self.military_npu_available:
            total_tops += self.military_npu_tops

        self.total_system_tops = total_tops

        import datetime
        self.last_updated = datetime.datetime.now().isoformat()

    def get_total_memory_gb(self) -> float:
        """Get total available memory for inference (system RAM only)."""
        total = self.usable_ram_gb

        # NCS2 storage is NOT additive - it's for model caching only
        # NPU and GNA use on-die memory (not additive to system RAM)

        if self.swap_available:
            total += self.swap_size_gb

        return total

    def get_total_accelerator_memory_gb(self) -> float:
        """Get total accelerator-specific inference memory (excluding system RAM)."""
        total = 0.0

        if self.ncs2_available:
            # NCS2 has ~512MB inference memory per device (NOT the 16GB storage)
            total += (self.ncs2_inference_memory_mb * self.ncs2_device_count) / 1024.0

        if self.npu_available:
            total += self.npu_on_die_memory_mb / 1024.0

        if self.gna_available:
            total += self.gna_on_die_memory_mb / 1024.0

        if self.military_npu_available:
            total += self.military_npu_cache_mb / 1024.0

        return total

    def print_summary(self):
        """Print hardware profile summary."""
        print("\n" + "=" * 70)
        print(f"HARDWARE PROFILE: {self.profile_name}")
        print("=" * 70)

        print(f"\n{'SYSTEM MEMORY':.<50}")
        print(f"  Total RAM:           {self.system_ram_gb:>6.0f} GB")
        print(f"  Usable RAM:          {self.usable_ram_gb:>6.0f} GB")
        print(f"  Swap:                {self.swap_size_gb:>6.0f} GB {'✓' if self.swap_available else '✗'}")

        print(f"\n{'INTEL ARC GRAPHICS':.<50}")
        print(f"  Available:           {self.arc_gpu_available and '✓ YES' or '✗ NO'}")
        if self.arc_gpu_available:
            print(f"  Shared RAM:          {self.arc_gpu_usable_gb:>6.0f} GB (unified memory)")
            print(f"  Compute:             {self.arc_gpu_tops_int8:>6.0f} TOPS INT8")
            print(f"  Compute Units:       {self.arc_gpu_compute_units:>6d}")

        print(f"\n{'INTEL NPU (AI BOOST)':.<50}")
        print(f"  Available:           {self.npu_available and '✓ YES' or '✗ NO'}")
        if self.npu_available:
            print(f"  PCI ID:              {self.npu_pci_id}")
            print(f"  Cache:               {self.npu_on_die_memory_mb:>6.0f} MB (on-die)")
            print(f"  Baseline:            {self.npu_tops_int8:>6.0f} TOPS INT8")
            print(f"  Optimized:           {self.npu_tops_optimized:>6.0f} TOPS")
            print(f"  Tiles:               {self.npu_num_tiles:>6d} ({self.npu_streams_per_tile * self.npu_num_tiles} streams)")

        print(f"\n{'INTEL GNA':.<50}")
        print(f"  Available:           {self.gna_available and '✓ YES' or '✗ NO'}")
        if self.gna_available:
            print(f"  PCI ID:              {self.gna_pci_id}")
            print(f"  Memory:              {self.gna_on_die_memory_mb:>6.0f} MB (on-die)")
            print(f"  Power:               {self.gna_power_w:>6.1f} W")
            print(f"  Specialization:      PQC crypto, token validation")

        print(f"\n{'INTEL NCS2 (MOVIDIUS)':.<50}")
        print(f"  Available:           {self.ncs2_available and '✓ YES' or '✗ NO'}")
        if self.ncs2_available:
            print(f"  Device Count:        {self.ncs2_device_count:>6d}")
            print(f"  Storage/Device:      {self.ncs2_storage_per_device_gb:>6.0f} GB (model caching)")
            print(f"  Total Storage:       {self.ncs2_storage_per_device_gb * self.ncs2_device_count:>6.0f} GB ({self.ncs2_device_count} devices)")
            print(f"  Inference Mem:       {self.ncs2_inference_memory_mb:>6.0f} MB (per device)")
            print(f"  Total Inference Mem: {self.ncs2_inference_memory_mb * self.ncs2_device_count:>6.0f} MB")
            print(f"  TOPS/Device:         {self.ncs2_tops_per_device:>6.0f}")
            print(f"  Total TOPS:          {self.ncs2_total_tops:>6.0f}")

        if self.military_npu_available:
            print(f"\n{'MILITARY NPU':.<50}")
            print(f"  Available:           ✓ YES")
            print(f"  Cache:               {self.military_npu_cache_mb:>6.0f} MB")
            print(f"  TOPS:                {self.military_npu_tops:>6.0f}")

        print(f"\n{'CPU':.<50}")
        print(f"  Total Cores:         {self.cpu_cores:>6d}")
        print(f"  P-cores:             {self.cpu_p_cores:>6d} (CPUs 0-{self.cpu_p_cores-1})")
        print(f"  E-cores:             {self.cpu_e_cores:>6d} (CPUs {self.cpu_p_cores}-{self.cpu_cores-1})")
        print(f"  AVX-512:             {self.cpu_avx512_available and '✓ YES' or '✗ NO'} (P-cores only)")
        print(f"  AVX2:                {self.cpu_avx2_available and '✓ YES' or '✗ NO'} (all cores)")

        print(f"\n{'SYSTEM TOTALS':.<50}")
        print(f"  Total Memory:        {self.get_total_memory_gb():>6.0f} GB")
        print(f"  Accelerator Memory:  {self.get_total_accelerator_memory_gb():>6.1f} GB")
        print(f"  Total TOPS:          {self.total_system_tops:>6.0f}")

        print("\n" + "=" * 70 + "\n")

    def save(self, path: str):
        """Save profile to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Hardware profile saved: {path}")

    @staticmethod
    def load(path: str) -> 'HardwareProfile':
        """Load profile from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Hardware profile loaded: {path}")
        return HardwareProfile(**data)


# Default profile with CONFIRMED dynamic system specs
DEFAULT_PROFILE = HardwareProfile(
    profile_name="Dell Latitude 5450 - Dynamic Specs (64GB physical, 2GB DSMIL, 2 NCS2)",
    physical_ram_gb=64.0,  # Physical RAM installed
    dsmil_reserved_gb=2.0,  # Reserved by DSMIL firmware
    system_ram_gb=62.0,  # OS-visible RAM
    usable_ram_gb=55.8,  # 90% of 62GB
    arc_gpu_usable_gb=55.8,  # Can use 90% of OS RAM
    arc_gpu_tops_int8=40.0,  # BASE: 40 TOPS INT8 (scales to 72 TOPS classified)
    npu_on_die_memory_mb=128.0,  # 128MB BAR0 region
    npu_tops_optimized=26.4,  # MILITARY: 26.4 TOPS (2.4x boost)
    npu_tops_classified=35.0,  # CLASSIFIED: ~35 TOPS
    ncs2_storage_per_device_gb=16.0,  # 16GB on-stick storage per device
    ncs2_inference_memory_mb=512.0,  # ~512MB inference memory per device
    ncs2_device_count=2,  # CURRENT: 2 devices, 3rd in mail
    military_npu_cache_mb=128.0,
)


# Singleton instance
_profile: Optional[HardwareProfile] = None


def get_hardware_profile() -> HardwareProfile:
    """Get or create hardware profile."""
    global _profile

    if _profile is None:
        # Try to load from config file
        config_path = Path("/home/user/LAT5150DRVMIL/02-ai-engine/hardware_profile.json")

        if config_path.exists():
            try:
                _profile = HardwareProfile.load(str(config_path))
            except Exception as e:
                logger.warning(f"Failed to load hardware profile: {e}")
                _profile = DEFAULT_PROFILE
        else:
            _profile = DEFAULT_PROFILE
            # Save default profile
            try:
                _profile.save(str(config_path))
            except Exception as e:
                logger.warning(f"Failed to save hardware profile: {e}")

    return _profile


def set_hardware_profile(profile: HardwareProfile):
    """Set custom hardware profile."""
    global _profile
    _profile = profile
    logger.info(f"Hardware profile updated: {profile.profile_name}")


def update_hardware_specs(
    system_ram_gb: Optional[float] = None,
    ncs2_storage_per_device_gb: Optional[float] = None,
    ncs2_inference_memory_mb: Optional[float] = None,
    npu_cache_mb: Optional[float] = None,
    arc_gpu_tops: Optional[float] = None,
    **kwargs
):
    """
    Update hardware specifications dynamically.

    Args:
        system_ram_gb: System RAM in GB
        ncs2_storage_per_device_gb: NCS2 on-stick storage per device in GB (for model caching)
        ncs2_inference_memory_mb: NCS2 inference memory per device in MB (actual inference RAM)
        npu_cache_mb: NPU BAR0 memory-mapped region size in MB
        arc_gpu_tops: Arc GPU compute in TOPS INT8
        **kwargs: Other profile fields to update
    """
    profile = get_hardware_profile()

    if system_ram_gb is not None:
        profile.system_ram_gb = system_ram_gb
        profile.usable_ram_gb = system_ram_gb * 0.9
        profile.arc_gpu_usable_gb = system_ram_gb * 0.9

    if ncs2_storage_per_device_gb is not None:
        profile.ncs2_storage_per_device_gb = ncs2_storage_per_device_gb

    if ncs2_inference_memory_mb is not None:
        profile.ncs2_inference_memory_mb = ncs2_inference_memory_mb

    if npu_cache_mb is not None:
        profile.npu_on_die_memory_mb = npu_cache_mb

    if arc_gpu_tops is not None:
        profile.arc_gpu_tops_int8 = arc_gpu_tops

    # Update any other fields
    for key, value in kwargs.items():
        if hasattr(profile, key):
            setattr(profile, key, value)

    # Recalculate totals
    profile.__post_init__()

    logger.info("Hardware specifications updated")
