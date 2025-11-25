#!/usr/bin/env python3
"""
⚠️  DEPRECATED - This component will be removed in v3.0.0 (2026 Q3)
⚠️  Use dsmil_control_centre_104.py or unified entry point: python3 dsmil.py control
⚠️  See DEPRECATION_PLAN.md for migration guide

DSMIL Subsystem Controller - CSNA 2.0 Compliant
Comprehensive integration with all DSMIL platform capabilities

This module provides deep integration with:
- 84 DSMIL devices (79 safe + 5 quarantined)
- CSNA 2.0 quantum encryption
- TPM 2.0 attestation
- AVX-512 unlock status
- Thermal monitoring
- Security controls
- Mode 5 platform integrity
- GNA presence detection
- NPU military mode status

QUARANTINED DEVICES (NEVER ACTIVATE):
- 0x8009, 0x800A, 0x800B: Data destruction
- 0x8019: Network kill
- 0x8029: Communications blackout
"""

import os
import sys
import subprocess
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

# Import device database and quantum crypto (use extended 104+24 database)
try:
    from dsmil_device_database_extended import (
        ALL_DEVICES_EXTENDED as ALL_DEVICES,
        QUARANTINED_DEVICES_EXTENDED as QUARANTINED_DEVICES,
        SAFE_DEVICES_EXTENDED as SAFE_DEVICES,
        get_device_extended as get_device,
        get_devices_by_group_extended as get_devices_by_group,
        get_statistics_extended as get_device_stats,
        DeviceGroup, DSMILDevice
    )
    from dsmil_device_database_extended import DeviceStatus as DBDeviceStatus
    # Define helper functions for backward compat
    get_safe_devices = lambda: SAFE_DEVICES
    get_quarantined_devices = lambda: QUARANTINED_DEVICES
except ImportError as e:
    print(f"Warning: Could not import extended device database: {e}")
    ALL_DEVICES = {}
    QUARANTINED_DEVICES = []
    SAFE_DEVICES = []

try:
    from quantum_crypto_layer import (
        get_crypto_layer, initialize_crypto, SecurityLevel
    )
except ImportError as e:
    print(f"Warning: Could not import quantum crypto layer: {e}")


# ============================================================================
# DEPRECATION WARNING - REMOVE IN v3.0.0 (2026 Q3)
# ============================================================================
import warnings
warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATED: dsmil_subsystem_controller.py\n\n"
    "This component is deprecated and will be removed in v3.0.0 (2026 Q3).\n\n"
    "Migration:\n"
    "  - Use: python3 dsmil.py control\n"
    "  - Or: python3 02-ai-engine/dsmil_control_centre_104.py\n\n"
    "Benefits:\n"
    "  - 104 devices (vs 84)\n"
    "  - Direct driver IOCTL interface\n"
    "  - 4-phase cascading discovery\n"
    "  - Enhanced safety enforcement\n\n"
    "See DEPRECATION_PLAN.md for complete migration guide.\n"
    + "=" * 80,
    DeprecationWarning,
    stacklevel=2
)
# ============================================================================


class DeviceStatus(Enum):
    """Device operational status"""
    SAFE = "safe"
    MONITORED = "monitored"
    QUARANTINED = "quarantined"
    UNKNOWN = "unknown"
    ACTIVE = "active"
    INACTIVE = "inactive"


class SubsystemType(Enum):
    """DSMIL subsystem types"""
    DEVICE_CONTROL = "device_control"
    MONITORING = "monitoring"
    SECURITY = "security"
    THERMAL = "thermal"
    TPM_ATTESTATION = "tpm_attestation"
    AVX512_UNLOCK = "avx512_unlock"
    NPU_STATUS = "npu_status"
    GPU_STATUS = "gpu_status"  # Intel iGPU/Arc detection
    GNA_PRESENCE = "gna_presence"
    MODE5_INTEGRITY = "mode5_integrity"


@dataclass
class DSMILDevice:
    """DSMIL device information"""
    device_id: int
    name: str
    status: DeviceStatus
    description: str
    safe_to_activate: bool
    current_value: Optional[int] = None
    capabilities: List[str] = None


@dataclass
class SubsystemStatus:
    """Status of a DSMIL subsystem"""
    subsystem: SubsystemType
    operational: bool
    status_info: Dict[str, Any]
    last_check: float


class DSMILSubsystemController:
    """
    Comprehensive controller for all DSMIL subsystems
    Integrates platform capabilities with AI engine
    """

    # Quarantined devices - imported from device database
    # Will use QUARANTINED_DEVICES from dsmil_device_database module

    # Safe devices - imported from device database
    # Will use SAFE_DEVICES from dsmil_device_database module

    def __init__(self):
        """Initialize DSMIL subsystem controller"""
        self.devices: Dict[int, DSMILDevice] = {}
        self.subsystems: Dict[SubsystemType, SubsystemStatus] = {}
        self.base_path = Path("/home/user/LAT5150DRVMIL")

        # Intel platform information (discovered during initialization)
        self._intel_platform = None

        # Easy Win #3: Device status caching
        self.device_status_cache = {}
        self.cache_ttl_seconds = 5  # 5 second cache

        # Easy Win #4: Operation history logging
        from collections import deque
        from datetime import datetime
        self.operation_history = deque(maxlen=1000)  # Last 1000 operations

        # Audit Storage: Persistent audit logging
        try:
            from dsmil_audit_storage import DSMILAuditStorage, RiskLevel
            self.audit_storage = DSMILAuditStorage()
            self.RiskLevel = RiskLevel
            print("✓ Audit storage initialized")
        except ImportError as e:
            print(f"Warning: Could not initialize audit storage: {e}")
            self.audit_storage = None
            self.RiskLevel = None

        # Initialize subsystems
        self._detect_subsystems()
        self._load_device_database()

        print("✓ DSMIL Subsystem Controller initialized")
        print(f"  Safe devices: {len(SAFE_DEVICES)}")
        print(f"  Quarantined: {len(QUARANTINED_DEVICES)}")
        print(f"  Total devices: {len(self.devices)}")
        print(f"  Subsystems detected: {len(self.subsystems)}")

    def _detect_subsystems(self):
        """Detect available DSMIL subsystems"""

        # Check device control
        self._check_device_control()

        # Check monitoring
        self._check_monitoring_system()

        # Check TPM
        self._check_tpm_system()

        # Check AVX-512
        self._check_avx512_unlock()

        # Check NPU status (also discovers full Intel platform)
        self._check_npu_status()

        # Check iGPU status (uses discovered platform info)
        self._check_gpu_status()

        # Check GNA (uses discovered platform info)
        self._check_gna_system()

        # Check thermal monitoring
        self._check_thermal_system()

        # Check Mode 5
        self._check_mode5_system()

    def _check_device_control(self):
        """Check DSMIL device control availability"""
        operational = False
        status_info = {}

        # Check for kernel module
        try:
            result = subprocess.run(
                ["lsmod"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if "dell" in result.stdout.lower() or "dsmil" in result.stdout.lower():
                operational = True
                status_info["kernel_module"] = "loaded"
            else:
                status_info["kernel_module"] = "not_loaded"

        except Exception as e:
            status_info["error"] = str(e)

        # Check sysfs paths
        sysfs_paths = [
            "/sys/devices/platform/dell-smbios",
            "/sys/module/dell_smbios",
            "/proc/dsmil"
        ]

        available_paths = []
        for path in sysfs_paths:
            if os.path.exists(path):
                available_paths.append(path)
                operational = True

        status_info["sysfs_paths"] = available_paths

        self.subsystems[SubsystemType.DEVICE_CONTROL] = SubsystemStatus(
            subsystem=SubsystemType.DEVICE_CONTROL,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _check_monitoring_system(self):
        """Check monitoring system availability"""
        operational = False
        status_info = {}

        # Check for monitor script
        monitor_path = self.base_path / "01-source/monitor/dsmil-monitor.py"
        if monitor_path.exists():
            operational = True
            status_info["monitor_script"] = str(monitor_path)

        self.subsystems[SubsystemType.MONITORING] = SubsystemStatus(
            subsystem=SubsystemType.MONITORING,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _check_tpm_system(self):
        """Check TPM 2.0 attestation availability"""
        operational = False
        status_info = {}

        # Check for TPM device
        if os.path.exists("/dev/tpm0") or os.path.exists("/dev/tpmrm0"):
            operational = True
            status_info["tpm_device"] = "present"

            # Try to get TPM version
            try:
                result = subprocess.run(
                    ["tpm2_getcap", "properties-fixed"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    status_info["tpm_tools"] = "available"
                    if "TPM2" in result.stdout or "2.0" in result.stdout:
                        status_info["tpm_version"] = "2.0"

            except FileNotFoundError:
                status_info["tpm_tools"] = "not_installed"
            except Exception as e:
                status_info["tpm_error"] = str(e)

        self.subsystems[SubsystemType.TPM_ATTESTATION] = SubsystemStatus(
            subsystem=SubsystemType.TPM_ATTESTATION,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _check_avx512_unlock(self):
        """Check AVX-512 unlock status"""
        operational = False
        status_info = {}

        # Check /proc/dsmil_avx512
        if os.path.exists("/proc/dsmil_avx512"):
            try:
                with open("/proc/dsmil_avx512", 'r') as f:
                    content = f.read().strip()
                    if "unlocked" in content.lower():
                        operational = True
                        status_info["avx512_status"] = "unlocked"
                    else:
                        status_info["avx512_status"] = content

            except Exception as e:
                status_info["error"] = str(e)

        # Check CPU flags
        try:
            with open("/proc/cpuinfo", 'r') as f:
                for line in f:
                    if "flags" in line and "avx512" in line.lower():
                        status_info["cpu_support"] = "present"
                        break

        except:
            pass

        self.subsystems[SubsystemType.AVX512_UNLOCK] = SubsystemStatus(
            subsystem=SubsystemType.AVX512_UNLOCK,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _check_npu_status(self):
        """Check NPU with comprehensive Intel hardware detection"""
        operational = False
        status_info = {}

        try:
            # Use comprehensive Intel hardware discovery
            from intel_hardware_discovery import IntelHardwareDiscovery

            discovery = IntelHardwareDiscovery()
            platform = discovery.discover_complete_platform()

            if platform.npu and platform.npu.present:
                operational = True
                status_info["model"] = platform.npu.model
                status_info["generation"] = platform.npu.generation
                status_info["tops"] = platform.npu.tops
                status_info["status"] = platform.npu.status
                status_info["openvino_available"] = platform.npu.openvino_available

                if platform.npu.driver_version:
                    status_info["driver_version"] = platform.npu.driver_version
                if platform.npu.device_path:
                    status_info["device_path"] = platform.npu.device_path

                # Store full platform info for later use
                self._intel_platform = platform

                print(f"✓ Intel NPU detected: {platform.npu.model}")
                print(f"  AI Acceleration: {platform.npu.tops} TOPS")
                print(f"  Status: {platform.npu.status}")

                # Also check iGPU for total AI acceleration
                if platform.gpu and platform.gpu.present:
                    status_info["total_ai_tops"] = platform.total_ai_tops
                    print(f"  Total AI (NPU+iGPU): {platform.total_ai_tops} TOPS")

        except Exception as e:
            status_info["discovery_error"] = str(e)
            print(f"⚠ Intel NPU discovery error: {e}")

            # Fallback to legacy detection
            try:
                import openvino as ov
                core = ov.Core()
                devices = core.available_devices()

                if 'NPU' in devices:
                    operational = True
                    status_info["openvino_npu"] = "available"
                    status_info["legacy_detection"] = True

            except:
                status_info["openvino_npu"] = "not_available"

        self.subsystems[SubsystemType.NPU_STATUS] = SubsystemStatus(
            subsystem=SubsystemType.NPU_STATUS,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _check_gpu_status(self):
        """Check Intel iGPU/Arc with comprehensive hardware detection"""
        operational = False
        status_info = {}

        # Use cached Intel platform info from NPU check
        if self._intel_platform and self._intel_platform.gpu:
            gpu = self._intel_platform.gpu

            if gpu.present:
                operational = True
                status_info["model"] = gpu.model
                status_info["architecture"] = gpu.architecture
                status_info["xe_cores"] = gpu.xe_cores
                status_info["execution_units"] = gpu.execution_units
                status_info["ai_tops"] = gpu.tops
                status_info["clock_speed_mhz"] = gpu.clock_speed_mhz
                status_info["memory_mb"] = gpu.memory_mb

                if gpu.driver_version:
                    status_info["driver_version"] = gpu.driver_version
                if gpu.device_id:
                    status_info["device_id"] = gpu.device_id
                if gpu.pci_address:
                    status_info["pci_address"] = gpu.pci_address

                # API availability
                status_info["apis"] = {
                    "opencl": gpu.opencl_available,
                    "vulkan": gpu.vulkan_available,
                    "level_zero": gpu.level_zero_available,
                    "openvino": gpu.openvino_available
                }

                print(f"✓ Intel iGPU detected: {gpu.model}")
                print(f"  Architecture: {gpu.architecture} ({gpu.xe_cores} Xe cores)")
                print(f"  AI Acceleration: {gpu.tops} TOPS")
                print(f"  Memory: {gpu.memory_mb} MB (shared)")

        else:
            # Fallback: basic lspci detection
            try:
                import subprocess
                result = subprocess.run(
                    ["lspci", "-nn"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                for line in result.stdout.splitlines():
                    if "VGA compatible controller" in line and "Intel" in line:
                        operational = True
                        status_info["detected"] = "basic_lspci"
                        status_info["info"] = line
                        break

            except Exception as e:
                status_info["detection_error"] = str(e)

        self.subsystems[SubsystemType.GPU_STATUS] = SubsystemStatus(
            subsystem=SubsystemType.GPU_STATUS,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _check_gna_system(self):
        """Check GNA presence with Intel hardware detection"""
        operational = False
        status_info = {}

        # Use cached Intel platform info if available (from NPU check)
        if self._intel_platform and self._intel_platform.npu:
            if self._intel_platform.npu.present and "GNA" in self._intel_platform.npu.generation:
                operational = True
                status_info["gna_device"] = "integrated_with_npu"
                status_info["generation"] = self._intel_platform.npu.generation
                print(f"✓ Intel GNA detected (integrated with {self._intel_platform.npu.model})")
        else:
            # Fallback to OpenVINO detection
            try:
                import openvino as ov
                core = ov.Core()
                devices = core.available_devices()

                if 'GNA' in devices:
                    operational = True
                    status_info["gna_device"] = "available"

            except:
                status_info["gna_device"] = "not_available"

        # Check for GNA scripts
        gna_script = self.base_path / "01-source/scripts/gna_integration_demo.py"
        if gna_script.exists():
            status_info["gna_scripts"] = "present"

        self.subsystems[SubsystemType.GNA_PRESENCE] = SubsystemStatus(
            subsystem=SubsystemType.GNA_PRESENCE,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _check_thermal_system(self):
        """Check thermal monitoring system"""
        operational = False
        status_info = {}

        # Check thermal guardian script
        thermal_script = self.base_path / "01-source/scripts/thermal_guardian.py"
        if thermal_script.exists():
            operational = True
            status_info["thermal_guardian"] = str(thermal_script)

        # Check system thermal zones
        thermal_zones = list(Path("/sys/class/thermal").glob("thermal_zone*"))
        if thermal_zones:
            status_info["thermal_zones"] = len(thermal_zones)

        self.subsystems[SubsystemType.THERMAL] = SubsystemStatus(
            subsystem=SubsystemType.THERMAL,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _check_mode5_system(self):
        """Check Mode 5 platform integrity"""
        operational = False
        status_info = {}

        # Check for Mode 5 indicators in kernel
        try:
            result = subprocess.run(
                ["dmesg"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if "mode5" in result.stdout.lower() or "mode 5" in result.stdout.lower():
                operational = True
                status_info["mode5_kernel"] = "detected"

        except:
            pass

        self.subsystems[SubsystemType.MODE5_INTEGRITY] = SubsystemStatus(
            subsystem=SubsystemType.MODE5_INTEGRITY,
            operational=operational,
            status_info=status_info,
            last_check=time.time()
        )

    def _load_device_database(self):
        """Load DSMIL device database from comprehensive device database module"""
        # Load all 84 devices from device database
        if ALL_DEVICES:
            # Convert database devices to controller format
            for device_id, db_device in ALL_DEVICES.items():
                # Map database DeviceStatus to controller DeviceStatus
                status_map = {
                    DBDeviceStatus.SAFE: DeviceStatus.SAFE,
                    DBDeviceStatus.QUARANTINED: DeviceStatus.QUARANTINED,
                    DBDeviceStatus.RISKY: DeviceStatus.UNKNOWN,
                    DBDeviceStatus.UNKNOWN: DeviceStatus.UNKNOWN
                }

                status = status_map.get(db_device.status, DeviceStatus.UNKNOWN)

                # Create controller device object
                controller_device = DSMILDevice(
                    device_id=db_device.device_id,
                    name=db_device.name,
                    status=status,
                    description=db_device.description,
                    safe_to_activate=db_device.safe_to_activate
                )

                self.devices[device_id] = controller_device

            print(f"✓ Loaded {len(self.devices)} DSMIL devices from database")
            stats = get_device_stats()
            print(f"  Safe: {stats['safe']}, Quarantined: {stats['quarantined']}, Unknown: {stats['unknown']}")
        else:
            # Fallback if database not available
            print("Warning: Device database not available, using minimal device set")
            # Minimal fallback definitions
            device_definitions = {
                0x8003: DSMILDevice(0x8003, "Display", DeviceStatus.SAFE, "Display configuration", True),
                0x8004: DSMILDevice(0x8004, "Power", DeviceStatus.SAFE, "Power management", True),
                0x8005: DSMILDevice(0x8005, "Thermal", DeviceStatus.SAFE, "Thermal controls", True),
                0x8006: DSMILDevice(0x8006, "Security", DeviceStatus.SAFE, "Security settings", True),
                0x8007: DSMILDevice(0x8007, "Performance", DeviceStatus.SAFE, "Performance modes", True),
                0x802A: DSMILDevice(0x802A, "Connectivity", DeviceStatus.SAFE, "Wireless controls", True),
            }

            # Quarantined devices (NEVER activate)
            quarantined_defs = {
                0x8009: DSMILDevice(0x8009, "Data Wipe 1", DeviceStatus.QUARANTINED, "Data destruction - QUARANTINED", False),
                0x800A: DSMILDevice(0x800A, "Data Wipe 2", DeviceStatus.QUARANTINED, "Data destruction - QUARANTINED", False),
                0x800B: DSMILDevice(0x800B, "Data Wipe 3", DeviceStatus.QUARANTINED, "Data destruction - QUARANTINED", False),
                0x8019: DSMILDevice(0x8019, "Network Kill", DeviceStatus.QUARANTINED, "Network termination - QUARANTINED", False),
                0x8029: DSMILDevice(0x8029, "Comms Blackout", DeviceStatus.QUARANTINED, "Communications disable - QUARANTINED", False),
            }

            self.devices.update(device_definitions)
            self.devices.update(quarantined_defs)

    def get_subsystem_status(self, subsystem: SubsystemType) -> Optional[SubsystemStatus]:
        """Get status of specific subsystem"""
        return self.subsystems.get(subsystem)

    def get_all_subsystems_status(self) -> Dict[str, Any]:
        """Get status of all subsystems"""
        return {
            subsystem.value: {
                "operational": status.operational,
                "info": status.status_info,
                "last_check": status.last_check
            }
            for subsystem, status in self.subsystems.items()
        }

    def get_device_info(self, device_id: int) -> Optional[DSMILDevice]:
        """Get information about a specific device"""
        return self.devices.get(device_id)

    def list_safe_devices(self) -> List[DSMILDevice]:
        """List all safe devices (excluding quarantined)"""
        return [
            device for device in self.devices.values()
            if device.safe_to_activate
        ]

    def list_quarantined_devices(self) -> List[DSMILDevice]:
        """List quarantined devices"""
        return [
            device for device in self.devices.values()
            if device.status == DeviceStatus.QUARANTINED
        ]

    def is_device_safe(self, device_id: int) -> bool:
        """Check if device is safe to activate"""
        if device_id in QUARANTINED_DEVICES:
            return False

        device = self.devices.get(device_id)
        return device is not None and device.safe_to_activate

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        operational_count = sum(1 for s in self.subsystems.values() if s.operational)
        total_count = len(self.subsystems)

        return {
            "overall_status": "healthy" if operational_count >= total_count * 0.6 else "degraded",
            "subsystems_operational": operational_count,
            "subsystems_total": total_count,
            "operational_percentage": (operational_count / total_count * 100) if total_count > 0 else 0,
            "safe_devices": len(self.list_safe_devices()),
            "quarantined_devices": len(QUARANTINED_DEVICES),
            "total_devices": len(self.devices),
            "last_check": time.time()
        }

    def activate_device(self, device_id: int, value: int) -> Tuple[bool, str]:
        """
        Activate a DSMIL device (SAFETY ENFORCED)

        Args:
            device_id: Device ID to activate
            value: Value to set

        Returns:
            (success, message)
        """
        # SAFETY CHECK: Quarantine enforcement
        if device_id in QUARANTINED_DEVICES:
            return (False, f"SAFETY VIOLATION: Device 0x{device_id:04X} is QUARANTINED and cannot be activated")

        # Check if device is known and safe
        if not self.is_device_safe(device_id):
            return (False, f"Device 0x{device_id:04X} is not safe for activation")

        # Check if device control is operational
        device_control = self.get_subsystem_status(SubsystemType.DEVICE_CONTROL)
        if not device_control or not device_control.operational:
            return (False, "Device control subsystem not operational")

        # TODO: Implement actual device activation via sysfs/ioctl
        # For now, return success for safe devices
        device = self.devices.get(device_id)
        if device:
            return (True, f"Device {device.name} (0x{device_id:04X}) would be activated with value {value}")
        else:
            return (False, f"Device 0x{device_id:04X} not found in database")

    def get_tpm_quote(self) -> Optional[str]:
        """Get TPM 2.0 quote for attestation"""
        tpm_status = self.get_subsystem_status(SubsystemType.TPM_ATTESTATION)

        if not tpm_status or not tpm_status.operational:
            return None

        try:
            # Generate TPM quote
            result = subprocess.run(
                ["tpm2_quote", "-c", "0x81000001", "-l", "sha256:0,1,2"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return result.stdout

        except Exception as e:
            print(f"TPM quote error: {e}")

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "system_health": self.get_system_health(),
            "subsystems": self.get_all_subsystems_status(),
            "devices": {
                "safe": len(self.list_safe_devices()),
                "quarantined": len(self.list_quarantined_devices()),
                "total": len(self.devices)
            }
        }

    def get_intel_platform_info(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive Intel platform information (CPU, GPU, NPU)
        Returns None if not detected
        """
        if not self._intel_platform:
            return None

        try:
            from intel_hardware_discovery import IntelHardwareDiscovery
            discovery = IntelHardwareDiscovery()
            return discovery.export_to_dict(self._intel_platform)
        except Exception as e:
            return {"error": str(e)}

    def get_ai_acceleration_summary(self) -> Dict[str, Any]:
        """Get summary of total AI acceleration capabilities (NPU + iGPU + NCS2)"""
        summary = {
            "npu_available": False,
            "igpu_available": False,
            "ncs2_available": False,
            "npu_tops": 0.0,
            "igpu_tops": 0.0,
            "ncs2_tops": 0.0,
            "ncs2_count": 0,
            "ncs2_custom_drivers": False,
            "total_tops": 0.0
        }

        if self._intel_platform:
            # NPU information
            if self._intel_platform.npu and self._intel_platform.npu.present:
                summary["npu_available"] = True
                summary["npu_tops"] = self._intel_platform.npu.tops
                summary["npu_model"] = self._intel_platform.npu.model

            # iGPU information
            if self._intel_platform.gpu and self._intel_platform.gpu.present:
                summary["igpu_available"] = True
                summary["igpu_tops"] = self._intel_platform.gpu.tops
                summary["igpu_model"] = self._intel_platform.gpu.model

            # NCS2 (Neural Compute Stick 2) information
            if self._intel_platform.ncs2 and self._intel_platform.ncs2.count > 0:
                summary["ncs2_available"] = True
                summary["ncs2_tops"] = self._intel_platform.ncs2.total_tops
                summary["ncs2_count"] = self._intel_platform.ncs2.count
                summary["ncs2_tops_per_stick"] = self._intel_platform.ncs2.tops_per_stick
                summary["ncs2_custom_drivers"] = self._intel_platform.ncs2.custom_driver
                summary["ncs2_devices"] = self._intel_platform.ncs2.device_names

            # Total AI acceleration (NPU + iGPU + NCS2)
            summary["total_tops"] = self._intel_platform.total_ai_tops

        return summary

    # ========================================================================
    # EASY WINS - Quick Improvements for Better Monitoring & Performance
    # ========================================================================

    def get_thermal_status_enhanced(self):
        """
        Easy Win #1: Enhanced thermal monitoring with per-core readings
        Time: 1-2 hours | Risk: None (read-only) | Value: HIGH
        """
        temps = []
        thermal_path = Path('/sys/class/thermal/')

        if not thermal_path.exists():
            return {
                'zones': [],
                'max_temp': 0,
                'overall_status': 'unavailable',
                'error': 'Thermal sysfs not available'
            }

        try:
            for zone in thermal_path.glob('thermal_zone*'):
                temp_file = zone / 'temp'
                type_file = zone / 'type'

                if temp_file.exists() and type_file.exists():
                    temp = int(temp_file.read_text().strip()) / 1000.0
                    zone_type = type_file.read_text().strip()
                    temps.append({
                        'zone': zone.name,
                        'type': zone_type,
                        'temp_c': round(temp, 1),
                        'status': 'critical' if temp > 90 else 'warning' if temp > 80 else 'normal'
                    })

            max_temp = max([t['temp_c'] for t in temps]) if temps else 0
            overall_status = 'critical' if any(t['status'] == 'critical' for t in temps) else \
                           'warning' if any(t['status'] == 'warning' for t in temps) else 'normal'

            return {
                'zones': temps,
                'max_temp': max_temp,
                'overall_status': overall_status,
                'zone_count': len(temps)
            }
        except Exception as e:
            return {
                'zones': [],
                'max_temp': 0,
                'overall_status': 'error',
                'error': str(e)
            }

    def get_tpm_pcr_state(self):
        """
        Easy Win #2: TPM PCR state tracking
        Time: 1-2 hours | Risk: None (read-only) | Value: MEDIUM
        """
        try:
            result = subprocess.run(
                ['tpm2_pcrread', 'sha256:0,1,2,3,4,5,6,7,8'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                pcrs = {}
                for line in result.stdout.split('\n'):
                    if ':' in line and 'sha256' not in line.lower() and '0x' in line:
                        parts = line.strip().split(':')
                        if len(parts) == 2:
                            pcr_num = parts[0].strip()
                            pcr_val = parts[1].strip()
                            pcrs[pcr_num] = pcr_val

                return {
                    'success': True,
                    'pcrs': pcrs,
                    'pcr_count': len(pcrs),
                    'timestamp': time.time()
                }
        except FileNotFoundError:
            return {'success': False, 'error': 'tpm2_pcrread not installed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

        return {'success': False, 'error': 'TPM read failed'}

    def get_tpm_event_log(self):
        """Get TPM event log for measurement count"""
        event_log_path = Path('/sys/kernel/security/tpm0/binary_bios_measurements')

        if not event_log_path.exists():
            return {'event_count': 0, 'log_available': False}

        try:
            event_data = event_log_path.read_bytes()
            # Simplified event count (each event has specific structure)
            event_count = len(event_data) // 32  # Approximate based on typical event size

            return {
                'event_count': event_count,
                'log_size_bytes': len(event_data),
                'log_available': True
            }
        except Exception as e:
            return {'event_count': 0, 'error': str(e), 'log_available': False}

    def get_device_status_cached(self, device_id: int):
        """
        Easy Win #3: Device status with caching to reduce SMI calls
        Time: 30 minutes | Risk: None | Value: MEDIUM (performance)
        """
        cache_key = f"status_{device_id:04X}"
        now = datetime.now()

        # Check cache
        if cache_key in self.device_status_cache:
            cached_time, cached_value = self.device_status_cache[cache_key]
            if (now - cached_time).total_seconds() < self.cache_ttl_seconds:
                return cached_value

        # Cache miss or expired - fetch fresh status
        # For now, return device from database (TODO: implement actual SMI read)
        device = self.devices.get(device_id)
        status = {
            'device_id': f"0x{device_id:04X}",
            'exists': device is not None,
            'name': device.name if device else 'Unknown',
            'safe': device_id in SAFE_DEVICES,
            'quarantined': device_id in QUARANTINED_DEVICES,
            'cached': False
        }

        self.device_status_cache[cache_key] = (now, status)
        return status

    def clear_device_cache(self, device_id: Optional[int] = None):
        """Clear cache for specific device or all devices"""
        if device_id:
            cache_key = f"status_{device_id:04X}"
            self.device_status_cache.pop(cache_key, None)
        else:
            self.device_status_cache.clear()

    def log_operation(self, device_id: int, operation: str,
                     success: bool, details: str = "", value: Optional[int] = None,
                     risk_level: Optional[str] = None, session_id: Optional[str] = None,
                     thermal_impact: Optional[float] = None):
        """
        Easy Win #4: Log device operation for history and debugging
        Time: 1 hour | Risk: None | Value: HIGH (debugging)

        Now with persistent audit storage
        """
        device = self.devices.get(device_id)
        device_name = device.name if device else 'Unknown'

        # In-memory operation history (Easy Win #4)
        entry = {
            'timestamp': datetime.now().isoformat(),
            'device_id': f"0x{device_id:04X}",
            'device_name': device_name,
            'operation': operation,
            'success': success,
            'details': details,
            'value': value
        }
        self.operation_history.append(entry)

        # Persistent audit storage
        if self.audit_storage:
            # Determine risk level based on device and operation
            if risk_level is None:
                if device_id in QUARANTINED_DEVICES:
                    risk_level = self.RiskLevel.CRITICAL
                elif operation in ['activate', 'write', 'emergency']:
                    risk_level = self.RiskLevel.MEDIUM
                elif operation in ['read', 'status']:
                    risk_level = self.RiskLevel.LOW
                else:
                    risk_level = self.RiskLevel.LOW

            try:
                self.audit_storage.store_event(
                    device_id=device_id,
                    operation=operation,
                    success=success,
                    device_name=device_name,
                    details=details,
                    value=value,
                    risk_level=risk_level,
                    session_id=session_id,
                    thermal_impact=thermal_impact
                )
            except Exception as e:
                print(f"Warning: Could not store audit event: {e}")

    def get_operation_history(self, device_id: Optional[int] = None,
                             limit: int = 100,
                             operation_type: Optional[str] = None):
        """Get operation history with optional filtering"""
        history = list(self.operation_history)

        # Filter by device
        if device_id:
            history = [h for h in history if h['device_id'] == f"0x{device_id:04X}"]

        # Filter by operation type
        if operation_type:
            history = [h for h in history if h['operation'] == operation_type]

        return history[-limit:]

    def get_operation_stats(self):
        """Get statistics about operations"""
        if not self.operation_history:
            return {
                'total_operations': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0,
                'operations_by_device': {},
                'most_active_device': None
            }

        total = len(self.operation_history)
        success = sum(1 for op in self.operation_history if op['success'])

        # Count by device
        by_device = {}
        for op in self.operation_history:
            dev = op['device_id']
            by_device[dev] = by_device.get(dev, 0) + 1

        most_active = max(by_device.items(), key=lambda x: x[1])[0] if by_device else None

        return {
            'total_operations': total,
            'successful': success,
            'failed': total - success,
            'success_rate': round((success / total) * 100, 1) if total > 0 else 0,
            'operations_by_device': by_device,
            'most_active_device': most_active
        }

    def get_subsystem_health_score(self) -> Dict[str, Any]:
        """
        Easy Win #5: Calculate health score for each subsystem (0.0 - 1.0)
        Time: 30 minutes | Risk: None | Value: HIGH (quick overview)
        """
        scores = {}

        for subsystem_type in SubsystemType:
            status = self.get_subsystem_status(subsystem_type)

            if not status:
                scores[subsystem_type.value] = 0.0
                continue

            # Base score on operational status
            score = 1.0 if status.operational else 0.0

            # Subsystem-specific adjustments
            if subsystem_type == SubsystemType.THERMAL:
                temp = status.status_info.get('current_temp', 0)
                if temp > 90:
                    score *= 0.3  # Critical thermal
                elif temp > 85:
                    score *= 0.6
                elif temp > 80:
                    score *= 0.8  # Warning thermal

            elif subsystem_type == SubsystemType.TPM_ATTESTATION:
                tpm_avail = status.status_info.get('tpm_device') == 'present'
                if not tpm_avail:
                    score *= 0.5

            elif subsystem_type == SubsystemType.SECURITY:
                # Could check for quarantine violations here
                pass

            scores[subsystem_type.value] = round(score, 2)

        # Overall system health
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0

        return {
            'subsystem_scores': scores,
            'overall_health': round(avg_score, 2),
            'status': 'excellent' if avg_score > 0.9 else
                     'good' if avg_score > 0.7 else
                     'fair' if avg_score > 0.5 else
                     'poor',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstration of DSMIL subsystem controller"""
    print("=" * 70)
    print(" DSMIL Subsystem Controller Demo")
    print("=" * 70)
    print()

    # Initialize controller
    controller = DSMILSubsystemController()

    print("\n" + "=" * 70)
    print(" System Health")
    print("=" * 70)
    health = controller.get_system_health()
    for key, value in health.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print(" Subsystem Status")
    print("=" * 70)
    subsystems = controller.get_all_subsystems_status()
    for name, status in subsystems.items():
        operational = "✓" if status["operational"] else "✗"
        print(f"  {operational} {name}: {status['info']}")

    print("\n" + "=" * 70)
    print(" Safe Devices")
    print("=" * 70)
    for device in controller.list_safe_devices():
        print(f"  ✓ 0x{device.device_id:04X}: {device.name} - {device.description}")

    print("\n" + "=" * 70)
    print(" Quarantined Devices (NEVER ACTIVATE)")
    print("=" * 70)
    for device in controller.list_quarantined_devices():
        print(f"  ⚠ 0x{device.device_id:04X}: {device.name} - {device.description}")

    print("\n" + "=" * 70)
    print(" Safety Test")
    print("=" * 70)

    # Test safe device
    success, msg = controller.activate_device(0x8003, 1)
    print(f"  Safe device test: {msg}")

    # Test quarantined device (should fail)
    success, msg = controller.activate_device(0x8009, 1)
    print(f"  Quarantined device test: {msg}")

    print("\n✅ DSMIL Subsystem Controller operational")


if __name__ == "__main__":
    demo()
