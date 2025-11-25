#!/usr/bin/env python3
"""
⚠️  DEPRECATED - This component will be removed in v3.0.0 (2026 Q3)
⚠️  Use: dsmil_device_database_extended.py (supports 104 devices)
⚠️  See DEPRECATION_PLAN.md for migration guide

DSMIL Complete Device Database

Comprehensive database of all 84 DSMIL devices across 7 groups.
Based on FULL_DEVICE_COVERAGE_ANALYSIS.md

Device Groups:
- Group 0 (0x8000-0x800B): Core Security & Emergency
- Group 1 (0x8010-0x801B): Extended Security
- Group 2 (0x8020-0x802B): Network & Communications
- Group 3 (0x8030-0x803B): Data Processing
- Group 4 (0x8040-0x804B): Storage Control
- Group 5 (0x8050-0x805B): Peripheral Management
- Group 6 (0x8060-0x806B): Training Functions

Safety Status:
- SAFE: Verified safe for monitoring/control
- QUARANTINED: Absolutely prohibited (destructive capability)
- RISKY: Not safe for operations without further testing
- UNKNOWN: Assume dangerous until verified
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict


# ============================================================================
# DEPRECATION WARNING - REMOVE IN v3.0.0 (2026 Q3)
# ============================================================================
import warnings
warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATED: dsmil_device_database.py (84 devices)\n\n"
    "This component is deprecated and will be removed in v3.0.0 (2026 Q3).\n\n"
    "Migration:\n"
    "  from dsmil_device_database_extended import (\n"
    "      ALL_DEVICES_EXTENDED,\n"
    "      get_device_extended\n"
    "  )\n\n"
    "New database supports 104 devices (vs 84) with 2 new groups.\n\n"
    "See DEPRECATION_PLAN.md for complete migration guide.\n"
    + "=" * 80,
    DeprecationWarning,
    stacklevel=2
)
# ============================================================================


class DeviceStatus(Enum):
    """Device safety status"""
    SAFE = "safe"
    QUARANTINED = "quarantined"
    RISKY = "risky"
    UNKNOWN = "unknown"


class DeviceGroup(Enum):
    """Device groups"""
    GROUP_0_CORE_SECURITY = 0
    GROUP_1_EXTENDED_SECURITY = 1
    GROUP_2_NETWORK_COMM = 2
    GROUP_3_DATA_PROCESSING = 3
    GROUP_4_STORAGE_CONTROL = 4
    GROUP_5_PERIPHERAL_MGT = 5
    GROUP_6_TRAINING = 6


@dataclass
class DSMILDevice:
    """DSMIL device definition"""
    device_id: int                  # Device token (e.g., 0x8003)
    name: str                       # Device name
    status: DeviceStatus            # Safety status
    description: str                # Functionality description
    safe_to_activate: bool          # Can be safely activated
    group: DeviceGroup              # Device group
    read_safe: bool = False         # Safe for READ operations
    write_safe: bool = False        # Safe for WRITE operations
    monitored: bool = False         # Currently monitored in production


# ============================================================================
# GROUP 0: Core Security & Emergency (0x8000-0x800B)
# ============================================================================

GROUP_0_DEVICES = {
    0x8000: DSMILDevice(
        0x8000, "TPM Control", DeviceStatus.UNKNOWN,
        "Trusted Platform Module control interface",
        False, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8001: DSMILDevice(
        0x8001, "Boot Security", DeviceStatus.UNKNOWN,
        "Secure boot integrity verification",
        False, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8002: DSMILDevice(
        0x8002, "Credential Vault", DeviceStatus.UNKNOWN,
        "Secure credential storage and retrieval",
        False, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8003: DSMILDevice(
        0x8003, "Audit Log Controller", DeviceStatus.SAFE,
        "System audit trail management",
        True, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=True, monitored=True
    ),
    0x8004: DSMILDevice(
        0x8004, "Event Logger", DeviceStatus.SAFE,
        "Security event logging and correlation",
        True, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=True, monitored=True
    ),
    0x8005: DSMILDevice(
        0x8005, "Performance Monitor", DeviceStatus.SAFE,
        "System performance metrics collection",
        True, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=True, monitored=True
    ),
    0x8006: DSMILDevice(
        0x8006, "Thermal Sensor Hub", DeviceStatus.SAFE,
        "Temperature monitoring across all zones",
        True, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=True, monitored=True
    ),
    0x8007: DSMILDevice(
        0x8007, "Power State Controller", DeviceStatus.SAFE,
        "Power management and state transitions",
        True, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=True, monitored=True
    ),
    0x8008: DSMILDevice(
        0x8008, "Emergency Response Prep", DeviceStatus.RISKY,
        "Emergency response preparation (adjacent to wipe devices)",
        False, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8009: DSMILDevice(
        0x8009, "DATA DESTRUCTION", DeviceStatus.QUARANTINED,
        "DOD-standard data destruction - NEVER TOUCH",
        False, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=False, write_safe=False
    ),
    0x800A: DSMILDevice(
        0x800A, "CASCADE WIPE", DeviceStatus.QUARANTINED,
        "Secondary cascade wipe system - NEVER TOUCH",
        False, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=False, write_safe=False
    ),
    0x800B: DSMILDevice(
        0x800B, "HARDWARE SANITIZE", DeviceStatus.QUARANTINED,
        "Final hardware sanitization - NEVER TOUCH",
        False, DeviceGroup.GROUP_0_CORE_SECURITY,
        read_safe=False, write_safe=False
    ),
}

# ============================================================================
# GROUP 1: Extended Security (0x8010-0x801B)
# ============================================================================

GROUP_1_DEVICES = {
    0x8010: DSMILDevice(
        0x8010, "Intrusion Detection", DeviceStatus.UNKNOWN,
        "Hardware-level intrusion detection system",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8011: DSMILDevice(
        0x8011, "Access Control List", DeviceStatus.UNKNOWN,
        "Hardware ACL management",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8012: DSMILDevice(
        0x8012, "Secure Channel", DeviceStatus.UNKNOWN,
        "Encrypted communication channel controller",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8013: DSMILDevice(
        0x8013, "Key Management", DeviceStatus.RISKY,
        "Cryptographic key management (could affect encryption)",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8014: DSMILDevice(
        0x8014, "Certificate Store", DeviceStatus.UNKNOWN,
        "Hardware certificate storage and validation",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8015: DSMILDevice(
        0x8015, "Network Filter", DeviceStatus.UNKNOWN,
        "Hardware network packet filtering",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8016: DSMILDevice(
        0x8016, "VPN Controller", DeviceStatus.RISKY,
        "VPN connection management (could affect connectivity)",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8017: DSMILDevice(
        0x8017, "Remote Access", DeviceStatus.RISKY,
        "Remote access control system (security implications)",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8018: DSMILDevice(
        0x8018, "Pre-Isolation State", DeviceStatus.RISKY,
        "Pre-network-isolation state management (adjacent to network kill)",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x8019: DSMILDevice(
        0x8019, "NETWORK KILL", DeviceStatus.QUARANTINED,
        "Network destruction capability - NEVER TOUCH",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=False, write_safe=False
    ),
    0x801A: DSMILDevice(
        0x801A, "Port Security", DeviceStatus.UNKNOWN,
        "Physical port security management",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
    0x801B: DSMILDevice(
        0x801B, "Wireless Security", DeviceStatus.UNKNOWN,
        "Wireless interface security controls",
        False, DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        read_safe=True, write_safe=False
    ),
}

# ============================================================================
# GROUP 2: Network & Communications (0x8020-0x802B)
# ============================================================================

GROUP_2_DEVICES = {
    0x8020: DSMILDevice(
        0x8020, "Network Interface", DeviceStatus.UNKNOWN,
        "Primary network interface controller",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8021: DSMILDevice(
        0x8021, "Ethernet Controller", DeviceStatus.UNKNOWN,
        "Ethernet hardware controller",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8022: DSMILDevice(
        0x8022, "WiFi Controller", DeviceStatus.UNKNOWN,
        "WiFi hardware controller",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8023: DSMILDevice(
        0x8023, "Bluetooth Manager", DeviceStatus.UNKNOWN,
        "Bluetooth subsystem management",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8024: DSMILDevice(
        0x8024, "Cellular Modem", DeviceStatus.UNKNOWN,
        "Cellular modem controller (if present)",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8025: DSMILDevice(
        0x8025, "DNS Resolver", DeviceStatus.UNKNOWN,
        "Hardware DNS resolution",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8026: DSMILDevice(
        0x8026, "DHCP Client", DeviceStatus.UNKNOWN,
        "Hardware DHCP client",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8027: DSMILDevice(
        0x8027, "Routing Table", DeviceStatus.UNKNOWN,
        "Hardware routing table management",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8028: DSMILDevice(
        0x8028, "QoS Manager", DeviceStatus.UNKNOWN,
        "Quality of Service management",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
    0x8029: DSMILDevice(
        0x8029, "COMMS BLACKOUT", DeviceStatus.QUARANTINED,
        "Communications blackout capability - NEVER TOUCH",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=False, write_safe=False
    ),
    0x802A: DSMILDevice(
        0x802A, "Network Monitor", DeviceStatus.SAFE,
        "Network traffic monitoring and analysis",
        True, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=True, monitored=True
    ),
    0x802B: DSMILDevice(
        0x802B, "Packet Filter", DeviceStatus.UNKNOWN,
        "Hardware packet filtering engine",
        False, DeviceGroup.GROUP_2_NETWORK_COMM,
        read_safe=True, write_safe=False
    ),
}

# ============================================================================
# GROUP 3: Data Processing (0x8030-0x803B)
# ============================================================================

GROUP_3_DEVICES = {
    0x8030: DSMILDevice(
        0x8030, "Memory Manager", DeviceStatus.UNKNOWN,
        "Hardware memory management controller",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8031: DSMILDevice(
        0x8031, "Cache Controller", DeviceStatus.UNKNOWN,
        "Hardware cache management",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8032: DSMILDevice(
        0x8032, "DMA Engine", DeviceStatus.UNKNOWN,
        "Direct memory access controller",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8033: DSMILDevice(
        0x8033, "Encryption Engine", DeviceStatus.UNKNOWN,
        "Hardware data encryption/decryption",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8034: DSMILDevice(
        0x8034, "Compression Engine", DeviceStatus.UNKNOWN,
        "Hardware data compression",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8035: DSMILDevice(
        0x8035, "Checksum Generator", DeviceStatus.UNKNOWN,
        "Hardware checksum/hash computation",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8036: DSMILDevice(
        0x8036, "Buffer Manager", DeviceStatus.UNKNOWN,
        "Buffer allocation and management",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8037: DSMILDevice(
        0x8037, "Stream Processor", DeviceStatus.UNKNOWN,
        "Data stream processing engine",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8038: DSMILDevice(
        0x8038, "Pattern Matcher", DeviceStatus.UNKNOWN,
        "Hardware pattern matching engine",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x8039: DSMILDevice(
        0x8039, "Data Validator", DeviceStatus.UNKNOWN,
        "Data integrity validation",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x803A: DSMILDevice(
        0x803A, "Transform Engine", DeviceStatus.UNKNOWN,
        "Data transformation and conversion",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
    0x803B: DSMILDevice(
        0x803B, "Processing Monitor", DeviceStatus.UNKNOWN,
        "Data processing performance monitoring",
        False, DeviceGroup.GROUP_3_DATA_PROCESSING,
        read_safe=True, write_safe=False
    ),
}

# ============================================================================
# GROUP 4: Storage Control (0x8040-0x804B)
# ============================================================================

GROUP_4_DEVICES = {
    0x8040: DSMILDevice(
        0x8040, "Storage Controller", DeviceStatus.UNKNOWN,
        "Primary storage interface controller",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8041: DSMILDevice(
        0x8041, "Disk Encryption", DeviceStatus.UNKNOWN,
        "Full disk encryption controller",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8042: DSMILDevice(
        0x8042, "Secure Erase", DeviceStatus.RISKY,
        "Secure disk erase functionality",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8043: DSMILDevice(
        0x8043, "Backup Controller", DeviceStatus.UNKNOWN,
        "Backup and restore management",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8044: DSMILDevice(
        0x8044, "RAID Manager", DeviceStatus.UNKNOWN,
        "RAID configuration and management",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8045: DSMILDevice(
        0x8045, "Storage Monitor", DeviceStatus.UNKNOWN,
        "Storage performance and health monitoring",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8046: DSMILDevice(
        0x8046, "Volume Manager", DeviceStatus.UNKNOWN,
        "Logical volume management",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8047: DSMILDevice(
        0x8047, "Partition Controller", DeviceStatus.UNKNOWN,
        "Partition management and protection",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8048: DSMILDevice(
        0x8048, "Cache Manager", DeviceStatus.UNKNOWN,
        "Storage cache management",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x8049: DSMILDevice(
        0x8049, "SMART Monitor", DeviceStatus.UNKNOWN,
        "SMART disk health monitoring",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x804A: DSMILDevice(
        0x804A, "Firmware Controller", DeviceStatus.UNKNOWN,
        "Storage firmware management",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
    0x804B: DSMILDevice(
        0x804B, "Write Protection", DeviceStatus.UNKNOWN,
        "Hardware write protection control",
        False, DeviceGroup.GROUP_4_STORAGE_CONTROL,
        read_safe=True, write_safe=False
    ),
}

# ============================================================================
# GROUP 5: Peripheral Management (0x8050-0x805B)
# ============================================================================

GROUP_5_DEVICES = {
    0x8050: DSMILDevice(
        0x8050, "USB Controller", DeviceStatus.UNKNOWN,
        "USB port management and control",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8051: DSMILDevice(
        0x8051, "Display Controller", DeviceStatus.UNKNOWN,
        "Display output management",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8052: DSMILDevice(
        0x8052, "Audio Subsystem", DeviceStatus.UNKNOWN,
        "Audio input/output control",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8053: DSMILDevice(
        0x8053, "Keyboard Controller", DeviceStatus.UNKNOWN,
        "Keyboard security and management",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8054: DSMILDevice(
        0x8054, "Mouse Controller", DeviceStatus.UNKNOWN,
        "Mouse/touchpad management",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8055: DSMILDevice(
        0x8055, "Camera Controller", DeviceStatus.UNKNOWN,
        "Camera/webcam control and privacy",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8056: DSMILDevice(
        0x8056, "Microphone Controller", DeviceStatus.UNKNOWN,
        "Microphone control and privacy",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8057: DSMILDevice(
        0x8057, "Sensor Hub", DeviceStatus.UNKNOWN,
        "Environmental sensor management",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8058: DSMILDevice(
        0x8058, "Battery Controller", DeviceStatus.UNKNOWN,
        "Battery monitoring and management",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x8059: DSMILDevice(
        0x8059, "Fan Controller", DeviceStatus.UNKNOWN,
        "Cooling fan speed control",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x805A: DSMILDevice(
        0x805A, "LED Controller", DeviceStatus.UNKNOWN,
        "System LED indicator control",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
    0x805B: DSMILDevice(
        0x805B, "Fingerprint Scanner", DeviceStatus.UNKNOWN,
        "Biometric fingerprint authentication",
        False, DeviceGroup.GROUP_5_PERIPHERAL_MGT,
        read_safe=True, write_safe=False
    ),
}

# ============================================================================
# GROUP 6: Training Functions (0x8060-0x806B)
# ============================================================================

GROUP_6_DEVICES = {
    0x8060: DSMILDevice(
        0x8060, "Training Controller", DeviceStatus.UNKNOWN,
        "Training mode management",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8061: DSMILDevice(
        0x8061, "Simulation Engine", DeviceStatus.UNKNOWN,
        "Training simulation control",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8062: DSMILDevice(
        0x8062, "Scenario Manager", DeviceStatus.UNKNOWN,
        "Training scenario orchestration",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8063: DSMILDevice(
        0x8063, "Data Recorder", DeviceStatus.UNKNOWN,
        "Training session data recording",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8064: DSMILDevice(
        0x8064, "Assessment Tool", DeviceStatus.UNKNOWN,
        "Training performance assessment",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8065: DSMILDevice(
        0x8065, "Exercise Controller", DeviceStatus.UNKNOWN,
        "Training exercise management",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8066: DSMILDevice(
        0x8066, "Replay Engine", DeviceStatus.UNKNOWN,
        "Training session replay system",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8067: DSMILDevice(
        0x8067, "Metrics Collector", DeviceStatus.UNKNOWN,
        "Training metrics collection",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8068: DSMILDevice(
        0x8068, "Benchmark Tool", DeviceStatus.UNKNOWN,
        "Training benchmark execution",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x8069: DSMILDevice(
        0x8069, "Recovery Controller", DeviceStatus.UNKNOWN,
        "Training mode recovery and reset",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x806A: DSMILDevice(
        0x806A, "JRTC1 Interface", DeviceStatus.UNKNOWN,
        "Junior Reserve Officers' Training Corps interface",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
    0x806B: DSMILDevice(
        0x806B, "Training Status", DeviceStatus.UNKNOWN,
        "Training system status monitoring",
        False, DeviceGroup.GROUP_6_TRAINING,
        read_safe=True, write_safe=False
    ),
}

# ============================================================================
# Complete Database
# ============================================================================

ALL_DEVICES = {}
ALL_DEVICES.update(GROUP_0_DEVICES)
ALL_DEVICES.update(GROUP_1_DEVICES)
ALL_DEVICES.update(GROUP_2_DEVICES)
ALL_DEVICES.update(GROUP_3_DEVICES)
ALL_DEVICES.update(GROUP_4_DEVICES)
ALL_DEVICES.update(GROUP_5_DEVICES)
ALL_DEVICES.update(GROUP_6_DEVICES)

# Quarantined devices list
QUARANTINED_DEVICES = [0x8009, 0x800A, 0x800B, 0x8019, 0x8029]

# Safe devices list
SAFE_DEVICES = [
    0x8003, 0x8004, 0x8005, 0x8006, 0x8007, 0x802A
]

def get_device(device_id: int) -> DSMILDevice:
    """Get device by ID"""
    return ALL_DEVICES.get(device_id)

def get_devices_by_group(group: DeviceGroup) -> Dict[int, DSMILDevice]:
    """Get all devices in a group"""
    return {k: v for k, v in ALL_DEVICES.items() if v.group == group}

def get_devices_by_status(status: DeviceStatus) -> Dict[int, DSMILDevice]:
    """Get all devices with a specific status"""
    return {k: v for k, v in ALL_DEVICES.items() if v.status == status}

def get_safe_devices() -> Dict[int, DSMILDevice]:
    """Get all safe devices"""
    return get_devices_by_status(DeviceStatus.SAFE)

def get_quarantined_devices() -> Dict[int, DSMILDevice]:
    """Get all quarantined devices"""
    return get_devices_by_status(DeviceStatus.QUARANTINED)

def get_monitored_devices() -> Dict[int, DSMILDevice]:
    """Get all currently monitored devices"""
    return {k: v for k, v in ALL_DEVICES.items() if v.monitored}

def get_read_safe_devices() -> Dict[int, DSMILDevice]:
    """Get all devices safe for READ operations"""
    return {k: v for k, v in ALL_DEVICES.items() if v.read_safe}

def get_statistics() -> Dict:
    """Get device database statistics"""
    return {
        "total_devices": len(ALL_DEVICES),
        "safe": len(get_safe_devices()),
        "quarantined": len(get_quarantined_devices()),
        "risky": len(get_devices_by_status(DeviceStatus.RISKY)),
        "unknown": len(get_devices_by_status(DeviceStatus.UNKNOWN)),
        "monitored": len(get_monitored_devices()),
        "read_safe": len(get_read_safe_devices()),
        "groups": {
            "group_0": len(get_devices_by_group(DeviceGroup.GROUP_0_CORE_SECURITY)),
            "group_1": len(get_devices_by_group(DeviceGroup.GROUP_1_EXTENDED_SECURITY)),
            "group_2": len(get_devices_by_group(DeviceGroup.GROUP_2_NETWORK_COMM)),
            "group_3": len(get_devices_by_group(DeviceGroup.GROUP_3_DATA_PROCESSING)),
            "group_4": len(get_devices_by_group(DeviceGroup.GROUP_4_STORAGE_CONTROL)),
            "group_5": len(get_devices_by_group(DeviceGroup.GROUP_5_PERIPHERAL_MGT)),
            "group_6": len(get_devices_by_group(DeviceGroup.GROUP_6_TRAINING)),
        }
    }


if __name__ == "__main__":
    """Print database statistics"""
    stats = get_statistics()
    print("="*70)
    print(" DSMIL DEVICE DATABASE STATISTICS")
    print("="*70)
    print(f"\nTotal Devices: {stats['total_devices']}")
    print(f"Safe:          {stats['safe']} ({stats['safe']/stats['total_devices']*100:.1f}%)")
    print(f"Quarantined:   {stats['quarantined']} ({stats['quarantined']/stats['total_devices']*100:.1f}%)")
    print(f"Risky:         {stats['risky']} ({stats['risky']/stats['total_devices']*100:.1f}%)")
    print(f"Unknown:       {stats['unknown']} ({stats['unknown']/stats['total_devices']*100:.1f}%)")
    print(f"Monitored:     {stats['monitored']} ({stats['monitored']/stats['total_devices']*100:.1f}%)")
    print(f"Read Safe:     {stats['read_safe']} ({stats['read_safe']/stats['total_devices']*100:.1f}%)")

    print("\nDevices by Group:")
    for group_name, count in stats['groups'].items():
        print(f"  {group_name}: {count} devices")

    print("\n" + "="*70)
    print(f" 79 USABLE DEVICES (84 total - 5 quarantined)")
    print("="*70)
