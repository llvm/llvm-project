#!/usr/bin/env python3
"""
DSMIL Driver Interface - v5.2.0 Compatible
==========================================
Python interface for the DSMIL 104-device kernel driver v5.2.0

Provides IOCTL bindings, token operations, TPM authentication,
BIOS management, and device discovery for the new driver architecture.

Author: LAT5150DRVMIL Integration Team
Version: 1.0.0
Driver Compatibility: dsmil-104dev v5.2.0
"""

import os
import sys
import struct
import fcntl
import ctypes
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DRIVER CONSTANTS
# ============================================================================

DRIVER_VERSION = "5.2.0"

# Preferred and fallback device paths. Some kernels expose the node as
# /dev/dsmil-104dev rather than the canonical /dev/dsmil0.
PRIMARY_DEVICE_PATH = "/dev/dsmil0"
FALLBACK_DEVICE_PATHS = [
    "/dev/dsmil-104dev",
    "/dev/dsmil",
]

DEVICE_PATH = PRIMARY_DEVICE_PATH

# Device architecture
DSMIL_DEVICE_COUNT = 104
DSMIL_GROUP_COUNT = 9  # 104 devices / 12 devices per group (with rounding)
DSMIL_BIOS_COUNT = 3

# Token ranges
TOKEN_DEVICE_BASE = 0x8000
TOKEN_DEVICE_END = 0x8137  # 104 devices × 3 tokens per device

# IOCTL magic number
DSMIL_IOC_MAGIC = ord('D')

# IOCTL commands (from updated driver)
DSMIL_IOC_GET_VERSION = 0x80044401        # _IOR('D', 1, u32)
DSMIL_IOC_GET_STATUS = 0x80444402         # _IOR('D', 2, struct)
DSMIL_IOC_READ_TOKEN = 0xC0084403         # _IOWR('D', 3, struct)
DSMIL_IOC_WRITE_TOKEN = 0xC0084404        # _IOWR('D', 4, struct)
DSMIL_IOC_DISCOVER_TOKENS = 0x80284405    # _IOR('D', 5, struct)
DSMIL_IOC_GET_DEVICE_INFO = 0xC0104406    # _IOWR('D', 6, struct)
DSMIL_IOC_GET_BIOS_STATUS = 0x80184407    # _IOR('D', 7, struct)
DSMIL_IOC_BIOS_FAILOVER = 0x40044408      # _IOW('D', 8, enum)
DSMIL_IOC_BIOS_SYNC = 0x40084409          # _IOW('D', 9, struct)
DSMIL_IOC_AUTHENTICATE = 0x4010440A       # _IOW('D', 10, struct)
DSMIL_IOC_TPM_GET_CHALLENGE = 0x8024440B  # _IOR('D', 11, struct)
DSMIL_IOC_TPM_INVALIDATE = 0x0000440C     # _IO('D', 12)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class BiosID(Enum):
    """BIOS system identifiers"""
    BIOS_A = 0
    BIOS_B = 1
    BIOS_C = 2


class ActivationMethod(Enum):
    """Device activation methods"""
    IOCTL = "ioctl"
    SYSFS = "sysfs"
    SMI = "smi"


# ============================================================================
# DATA STRUCTURES (matching kernel driver)
# ============================================================================

@dataclass
class SystemStatus:
    """System status (DSMIL_IOC_GET_STATUS)"""
    driver_version: int
    device_count: int
    group_count: int
    active_bios: int
    bios_health_a: int
    bios_health_b: int
    bios_health_c: int
    thermal_celsius: int
    authenticated: int
    token_reads: int
    token_writes: int
    failover_count: int


@dataclass
class TokenOp:
    """Token operation (READ/WRITE)"""
    token_id: int
    value: int = 0


@dataclass
class DeviceInfo:
    """Device information"""
    device_id: int
    status: int = 0
    config: int = 0
    data: int = 0


@dataclass
class BiosStatus:
    """BIOS status for all 3 systems"""
    active_bios: int
    bios_a_health: int
    bios_b_health: int
    bios_c_health: int
    bios_a_errors: int
    bios_b_errors: int
    bios_c_errors: int


@dataclass
class BiosSyncRequest:
    """BIOS sync request"""
    source: int
    target: int


@dataclass
class AuthRequest:
    """Authentication request"""
    auth_method: int
    auth_data: bytes
    auth_data_len: int


@dataclass
class TPMChallenge:
    """TPM challenge data"""
    challenge: bytes  # 32 bytes
    challenge_id: int
    tpm_available: int


# ============================================================================
# CTYPES STRUCTURES (for IOCTL)
# ============================================================================

class CSystemStatus(ctypes.Structure):
    """C structure for system status"""
    _fields_ = [
        ("driver_version", ctypes.c_uint32),
        ("device_count", ctypes.c_uint32),
        ("group_count", ctypes.c_uint32),
        ("active_bios", ctypes.c_uint32),
        ("bios_health_a", ctypes.c_uint32),
        ("bios_health_b", ctypes.c_uint32),
        ("bios_health_c", ctypes.c_uint32),
        ("thermal_celsius", ctypes.c_int32),
        ("authenticated", ctypes.c_uint32),
        ("token_reads", ctypes.c_uint32),
        ("token_writes", ctypes.c_uint32),
        ("failover_count", ctypes.c_uint32),
    ]


class CTokenOp(ctypes.Structure):
    """C structure for token operations"""
    _fields_ = [
        ("token_id", ctypes.c_uint16),
        ("value", ctypes.c_uint32),
    ]


class CTPMChallenge(ctypes.Structure):
    """C structure for TPM challenge"""
    _fields_ = [
        ("challenge", ctypes.c_uint8 * 32),
        ("challenge_id", ctypes.c_uint32),
        ("tpm_available", ctypes.c_uint8),
    ]


# ============================================================================
# DRIVER INTERFACE CLASS
# ============================================================================

class DSMILDriverInterface:
    """
    Python interface to DSMIL 104-device kernel driver v5.2.0

    Provides high-level API for:
    - Token read/write operations
    - Device discovery and enumeration
    - TPM authentication
    - BIOS management
    - System monitoring
    """

    def __init__(self, device_path: Optional[str] = None):
        """Initialize driver interface"""
        # Resolve device path: prefer explicit, then primary, then fallbacks
        if device_path is not None:
            self.device_path = device_path
        else:
            self.device_path = resolve_device_path()
        self.fd = None
        self._authenticated = False

    def open(self) -> bool:
        """Open driver device"""
        try:
            self.fd = os.open(self.device_path, os.O_RDWR)
            logger.info(f"Opened driver device: {self.device_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to open {self.device_path}: {e}")
            self.fd = None
            return False

    def close(self):
        """Close driver device"""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            logger.info("Closed driver device")

    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def is_open(self) -> bool:
        """Check if driver is open"""
        return self.fd is not None

    # ========================================================================
    # VERSION AND STATUS
    # ========================================================================

    def get_version(self) -> Optional[str]:
        """Get driver version"""
        if not self.is_open():
            return None

        try:
            version_buf = ctypes.c_uint32()
            fcntl.ioctl(self.fd, DSMIL_IOC_GET_VERSION, version_buf)

            major = (version_buf.value >> 16) & 0xFF
            minor = (version_buf.value >> 8) & 0xFF
            patch = version_buf.value & 0xFF

            return f"{major}.{minor}.{patch}"
        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            return None

    def get_status(self) -> Optional[SystemStatus]:
        """Get system status"""
        if not self.is_open():
            return None

        try:
            status = CSystemStatus()
            fcntl.ioctl(self.fd, DSMIL_IOC_GET_STATUS, status)

            return SystemStatus(
                driver_version=status.driver_version,
                device_count=status.device_count,
                group_count=status.group_count,
                active_bios=status.active_bios,
                bios_health_a=status.bios_health_a,
                bios_health_b=status.bios_health_b,
                bios_health_c=status.bios_health_c,
                thermal_celsius=status.thermal_celsius,
                authenticated=status.authenticated,
                token_reads=status.token_reads,
                token_writes=status.token_writes,
                failover_count=status.failover_count,
            )
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return None

    # ========================================================================
    # TOKEN OPERATIONS
    # ========================================================================

    def read_token(self, token_id: int) -> Optional[int]:
        """Read token value"""
        if not self.is_open():
            return None

        try:
            token_op = CTokenOp()
            token_op.token_id = token_id
            token_op.value = 0

            fcntl.ioctl(self.fd, DSMIL_IOC_READ_TOKEN, token_op)

            logger.debug(f"Read token 0x{token_id:04X} = 0x{token_op.value:08X}")
            return token_op.value
        except Exception as e:
            logger.error(f"Failed to read token 0x{token_id:04X}: {e}")
            return None

    def write_token(self, token_id: int, value: int) -> bool:
        """Write token value"""
        if not self.is_open():
            return False

        try:
            token_op = CTokenOp()
            token_op.token_id = token_id
            token_op.value = value

            fcntl.ioctl(self.fd, DSMIL_IOC_WRITE_TOKEN, token_op)

            logger.debug(f"Wrote token 0x{token_id:04X} = 0x{value:08X}")
            return True
        except Exception as e:
            logger.error(f"Failed to write token 0x{token_id:04X}: {e}")
            return False

    # ========================================================================
    # DEVICE OPERATIONS
    # ========================================================================

    def get_device_info(self, device_id: int) -> Optional[DeviceInfo]:
        """Get device information (status, config, data tokens)"""
        if not self.is_open():
            return None

        try:
            # Calculate token IDs for this device
            base_token = TOKEN_DEVICE_BASE + (device_id * 3)

            status = self.read_token(base_token + 0)  # Status token
            config = self.read_token(base_token + 1)  # Config token
            data = self.read_token(base_token + 2)    # Data token

            if status is None:
                return None

            return DeviceInfo(
                device_id=device_id,
                status=status,
                config=config if config is not None else 0,
                data=data if data is not None else 0,
            )
        except Exception as e:
            logger.error(f"Failed to get device info for device {device_id}: {e}")
            return None

    def discover_devices(self) -> List[int]:
        """
        Discover all 104 devices by attempting to read their status tokens

        Returns list of device IDs that respond successfully
        """
        discovered = []

        for device_id in range(DSMIL_DEVICE_COUNT):
            token_id = TOKEN_DEVICE_BASE + (device_id * 3)

            # Try to read status token
            value = self.read_token(token_id)
            if value is not None:
                discovered.append(device_id)
                logger.debug(f"Discovered device {device_id} (0x{token_id:04X})")

        logger.info(f"Discovered {len(discovered)}/{DSMIL_DEVICE_COUNT} devices")
        return discovered

    def activate_device(self, device_id: int) -> bool:
        """
        Activate a device by writing to its config token

        Sets bit 0 (enable) in the device's config token
        """
        if not self.is_open():
            return False

        try:
            config_token = TOKEN_DEVICE_BASE + (device_id * 3) + 1

            # Read current config
            current_config = self.read_token(config_token)
            if current_config is None:
                return False

            # Set enable bit
            new_config = current_config | 0x00000001

            # Write new config
            success = self.write_token(config_token, new_config)

            if success:
                logger.info(f"Activated device {device_id}")

            return success
        except Exception as e:
            logger.error(f"Failed to activate device {device_id}: {e}")
            return False

    # ========================================================================
    # TPM AUTHENTICATION
    # ========================================================================

    def tpm_get_challenge(self) -> Optional[TPMChallenge]:
        """Get TPM authentication challenge"""
        if not self.is_open():
            return None

        try:
            chal = CTPMChallenge()
            fcntl.ioctl(self.fd, DSMIL_IOC_TPM_GET_CHALLENGE, chal)

            challenge_bytes = bytes(chal.challenge)

            logger.info(f"TPM challenge: ID=0x{chal.challenge_id:08X}, " +
                       f"TPM available={bool(chal.tpm_available)}")

            return TPMChallenge(
                challenge=challenge_bytes,
                challenge_id=chal.challenge_id,
                tpm_available=chal.tpm_available,
            )
        except Exception as e:
            logger.error(f"Failed to get TPM challenge: {e}")
            return None

    def authenticate(self, auth_method: int, auth_data: bytes) -> bool:
        """Submit authentication credentials"""
        if not self.is_open():
            return False

        try:
            # Prepare auth request structure
            # Note: Actual structure packing depends on kernel driver definition
            auth_buf = struct.pack("I", auth_method) + auth_data[:256]
            auth_len = len(auth_data)

            # This is a simplified version - actual implementation needs proper struct
            fcntl.ioctl(self.fd, DSMIL_IOC_AUTHENTICATE, auth_buf)

            self._authenticated = True
            logger.info("Authentication successful")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def tpm_invalidate(self) -> bool:
        """Invalidate TPM session"""
        if not self.is_open():
            return False

        try:
            fcntl.ioctl(self.fd, DSMIL_IOC_TPM_INVALIDATE)
            self._authenticated = False
            logger.info("TPM session invalidated")
            return True
        except Exception as e:
            logger.error(f"Failed to invalidate TPM session: {e}")
            return False

    # ========================================================================
    # BIOS MANAGEMENT
    # ========================================================================

    def get_bios_status(self) -> Optional[BiosStatus]:
        """Get BIOS status for all 3 systems"""
        if not self.is_open():
            return None

        try:
            # Read via status structure
            status = self.get_status()
            if not status:
                return None

            return BiosStatus(
                active_bios=status.active_bios,
                bios_a_health=status.bios_health_a,
                bios_b_health=status.bios_health_b,
                bios_c_health=status.bios_health_c,
                bios_a_errors=0,  # These would need separate token reads
                bios_b_errors=0,
                bios_c_errors=0,
            )
        except Exception as e:
            logger.error(f"Failed to get BIOS status: {e}")
            return None

    def bios_failover(self, target_bios: BiosID) -> bool:
        """Trigger manual BIOS failover"""
        if not self.is_open():
            return False

        try:
            bios_id = ctypes.c_uint32(target_bios.value)
            fcntl.ioctl(self.fd, DSMIL_IOC_BIOS_FAILOVER, bios_id)

            logger.info(f"BIOS failover to {target_bios.name} successful")
            return True
        except Exception as e:
            logger.error(f"BIOS failover failed: {e}")
            return False

    # ========================================================================
    # SYSFS INTERFACE (alternative to IOCTL)
    # ========================================================================

    @staticmethod
    def sysfs_read(attribute: str) -> Optional[str]:
        """Read from sysfs attribute"""
        sysfs_path = f"/sys/class/dsmil/dsmil0/{attribute}"
        try:
            with open(sysfs_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read sysfs {attribute}: {e}")
            return None

    @staticmethod
    def sysfs_write(attribute: str, value: str) -> bool:
        """Write to sysfs attribute"""
        sysfs_path = f"/sys/class/dsmil/dsmil0/{attribute}"
        try:
            with open(sysfs_path, 'w') as f:
                f.write(value)
            return True
        except Exception as e:
            logger.error(f"Failed to write sysfs {attribute}: {e}")
            return False

    def get_sysfs_device_count(self) -> Optional[int]:
        """Get device count from sysfs"""
        count_str = self.sysfs_read("device_count")
        return int(count_str) if count_str else None

    def get_sysfs_driver_version(self) -> Optional[str]:
        """Get driver version from sysfs"""
        return self.sysfs_read("driver_version")

    def get_sysfs_tpm_status(self) -> Optional[str]:
        """Get TPM status from sysfs"""
        return self.sysfs_read("tpm_status")

    def get_sysfs_error_stats(self) -> Optional[str]:
        """Get error statistics from sysfs"""
        return self.sysfs_read("error_stats")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_driver_loaded() -> bool:
    """Check if DSMIL driver is loaded"""
    return resolve_device_path() is not None


def get_driver_version() -> Optional[str]:
    """Get driver version (convenience function)"""
    try:
        with DSMILDriverInterface() as driver:
            return driver.get_version()
    except:
        return None


def discover_all_devices() -> List[int]:
    """Discover all devices (convenience function)"""
    try:
        with DSMILDriverInterface() as driver:
            return driver.discover_devices()
    except:
        return []


def quick_device_scan() -> Dict[str, any]:
    """Quick device scan returning summary"""
    resolved_path = resolve_device_path()

    result = {
        'driver_loaded': resolved_path is not None,
        'driver_version': None,
        'device_count': 0,
        'devices_discovered': [],
        'system_status': None,
        'backend_name': None,
    }

    if not result['driver_loaded']:
        return result

    try:
        with DSMILDriverInterface(device_path=resolved_path) as driver:
            result['driver_version'] = driver.get_version()
            result['devices_discovered'] = driver.discover_devices()
            result['device_count'] = len(result['devices_discovered'])
            result['system_status'] = driver.get_status()
            # Try to read backend name from sysfs (best-effort)
            backend = get_backend_name()
            result['backend_name'] = backend
    except Exception as e:
        logger.error(f"Quick scan failed: {e}")

    return result


def resolve_device_path() -> Optional[str]:
    """
    Resolve the best available DSMIL device path.

    Preference order:
    1. /dev/dsmil0
    2. /dev/dsmil-104dev
    3. /dev/dsmil
    """
    import os

    if os.path.exists(PRIMARY_DEVICE_PATH):
        return PRIMARY_DEVICE_PATH

    for path in FALLBACK_DEVICE_PATHS:
        if os.path.exists(path):
            return path

    return None

def get_backend_name() -> Optional[str]:
    """
    Best-effort detection of SMBIOS backend name via sysfs.

    Returns backend string (e.g. 'simulated (database-aware)' or
    'dell-smbios (kernel subsystem)') or None if unavailable.
    """
    import os

    candidates = [
        "/sys/class/dsmil-104dev/dsmil0/smbios_backend",
        "/sys/class/dsmil-104dev-override/dsmil0/smbios_backend",
    ]

    for path in candidates:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read().strip()
        except Exception:
            continue

    return None


# ============================================================================
# MAIN - DIAGNOSTICS
# ============================================================================

def main():
    """Diagnostic tool"""
    print("=" * 70)
    print("DSMIL Driver Interface - Diagnostic Tool")
    print("=" * 70)
    print()

    # Check driver
    print("[1/5] Checking driver...")
    if not check_driver_loaded():
        print(f"  ✗ Driver not loaded ({DEVICE_PATH} not found)")
        print(f"  Run: sudo insmod dsmil-104dev.ko")
        return 1

    print(f"  ✓ Driver loaded ({DEVICE_PATH} exists)")

    # Get version
    print("\n[2/5] Getting driver version...")
    with DSMILDriverInterface() as driver:
        version = driver.get_version()
        if version:
            print(f"  ✓ Driver version: {version}")
        else:
            print(f"  ✗ Failed to get version")

        # Get status
        print("\n[3/5] Getting system status...")
        status = driver.get_status()
        if status:
            print(f"  ✓ Device count: {status.device_count}")
            print(f"  ✓ Group count: {status.group_count}")
            print(f"  ✓ Active BIOS: {chr(ord('A') + status.active_bios)}")
            print(f"  ✓ BIOS health: A={status.bios_health_a} " +
                  f"B={status.bios_health_b} C={status.bios_health_c}")
            print(f"  ✓ Thermal: {status.thermal_celsius}°C")
            print(f"  ✓ Token ops: {status.token_reads} reads, " +
                  f"{status.token_writes} writes")
        else:
            print(f"  ✗ Failed to get status")

        # Discover devices
        print("\n[4/5] Discovering devices...")
        discovered = driver.discover_devices()
        print(f"  ✓ Discovered {len(discovered)}/{DSMIL_DEVICE_COUNT} devices")

        if discovered:
            print(f"  First 5 devices: ", end="")
            for i in range(min(5, len(discovered))):
                print(f"{discovered[i]} ", end="")
            print()

        # Test token read
        print("\n[5/5] Testing token read...")
        if discovered:
            test_device = discovered[0]
            test_token = TOKEN_DEVICE_BASE + (test_device * 3)
            value = driver.read_token(test_token)

            if value is not None:
                print(f"  ✓ Read device {test_device} status token 0x{test_token:04X} = 0x{value:08X}")
            else:
                print(f"  ✗ Failed to read token")

    print()
    print("=" * 70)
    print("Diagnostic complete")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
