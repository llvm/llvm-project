#!/usr/bin/env python3
"""
DSMIL Hardware Access Module
Provides low-level hardware access including MSR and SMM operations

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Requires: Root privileges, kernel modules (msr, mem)
"""

import os
import struct
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum


class AccessLevel(Enum):
    """Hardware access levels"""
    USERSPACE = "userspace"
    KERNEL = "kernel"
    MSR = "msr"              # Model-Specific Registers
    SMM = "smm"              # System Management Mode
    FIRMWARE = "firmware"
    MEMORY_MAPPED = "mmio"   # Memory-Mapped I/O


class MSRAccess:
    """Model-Specific Register (MSR) access"""

    MSR_DEVICE = "/dev/cpu/{cpu}/msr"

    # Common MSR addresses
    MSR_IA32_FEATURE_CONTROL = 0x3A
    MSR_IA32_BIOS_SIGN_ID = 0x8B
    MSR_IA32_SMM_MONITOR_CTL = 0x9B
    MSR_IA32_MTRRCAP = 0xFE
    MSR_IA32_SYSENTER_CS = 0x174
    MSR_IA32_SYSENTER_ESP = 0x175
    MSR_IA32_SYSENTER_EIP = 0x176
    MSR_IA32_MISC_ENABLE = 0x1A0
    MSR_IA32_TSC = 0x10

    @staticmethod
    def is_available() -> bool:
        """Check if MSR access is available"""
        # Check if msr module is loaded
        msr_loaded = os.path.exists("/dev/cpu/0/msr")
        if not msr_loaded:
            # Try to load msr module
            try:
                subprocess.run(['modprobe', 'msr'], check=False,
                             capture_output=True, timeout=2)
                msr_loaded = os.path.exists("/dev/cpu/0/msr")
            except:
                pass

        return msr_loaded and os.geteuid() == 0

    @staticmethod
    def read_msr(msr_address: int, cpu: int = 0) -> Optional[int]:
        """
        Read a Model-Specific Register

        Args:
            msr_address: MSR register address (e.g., 0x3A)
            cpu: CPU number to read from

        Returns:
            64-bit MSR value or None on error
        """
        if not MSRAccess.is_available():
            return None

        msr_path = f"/dev/cpu/{cpu}/msr"
        try:
            with open(msr_path, 'rb') as f:
                f.seek(msr_address)
                data = f.read(8)  # MSRs are 64-bit
                if len(data) == 8:
                    return struct.unpack('<Q', data)[0]
        except Exception as e:
            print(f"Error reading MSR 0x{msr_address:X}: {e}")
            return None

    @staticmethod
    def write_msr(msr_address: int, value: int, cpu: int = 0) -> bool:
        """
        Write to a Model-Specific Register (DANGEROUS - requires extreme caution)

        Args:
            msr_address: MSR register address
            value: 64-bit value to write
            cpu: CPU number to write to

        Returns:
            True on success, False on error
        """
        if not MSRAccess.is_available():
            return False

        msr_path = f"/dev/cpu/{cpu}/msr"
        try:
            with open(msr_path, 'wb') as f:
                f.seek(msr_address)
                data = struct.pack('<Q', value)
                f.write(data)
                return True
        except Exception as e:
            print(f"Error writing MSR 0x{msr_address:X}: {e}")
            return False

    @staticmethod
    def get_all_cpus() -> List[int]:
        """Get list of available CPU numbers"""
        cpu_dirs = Path("/dev/cpu").glob("*/msr")
        return sorted([int(p.parent.name) for p in cpu_dirs])

    @staticmethod
    def read_feature_control() -> Optional[Dict[str, Any]]:
        """Read IA32_FEATURE_CONTROL MSR and decode"""
        value = MSRAccess.read_msr(MSRAccess.MSR_IA32_FEATURE_CONTROL)
        if value is None:
            return None

        return {
            'raw_value': value,
            'locked': bool(value & 0x1),
            'vmx_inside_smx': bool(value & 0x2),
            'vmx_outside_smx': bool(value & 0x4),
            'smx_enabled': bool(value & (0x2 | 0x20)),
        }

    @staticmethod
    def read_smm_monitor_ctl() -> Optional[Dict[str, Any]]:
        """Read IA32_SMM_MONITOR_CTL MSR for SMM configuration"""
        value = MSRAccess.read_msr(MSRAccess.MSR_IA32_SMM_MONITOR_CTL)
        if value is None:
            return None

        return {
            'raw_value': value,
            'valid': bool(value & 0x1),
            'mseg_base': (value >> 12) << 12,  # Bits 12-31
        }


class SMMAccess:
    """System Management Mode (SMM) access"""

    SMM_PORT = 0xB2  # APMC (Advanced Power Management Control) port

    @staticmethod
    def is_available() -> bool:
        """Check if SMM trigger is available"""
        # SMM requires root and specific hardware support
        if os.geteuid() != 0:
            return False

        # Check if we have I/O port access
        try:
            # Try to access /dev/port (requires root)
            return os.path.exists("/dev/port")
        except:
            return False

    @staticmethod
    def trigger_smi(command: int) -> bool:
        """
        Trigger a System Management Interrupt (DANGEROUS - firmware level)

        Args:
            command: SMI command byte to send

        Returns:
            True if command was sent (not if it succeeded)
        """
        if not SMMAccess.is_available():
            return False

        # EXTREME CAUTION: This triggers firmware-level SMI
        # Only specific command codes are safe
        # Unknown codes can brick the system

        print(f"WARNING: SMI trigger requested (0x{command:02X})")
        print("This is a firmware-level operation and can be dangerous!")

        # For safety, we only allow read-only SMI commands
        SAFE_SMI_COMMANDS = [
            0x00,  # NOP/Query
            0x01,  # Get SMI version
            # Add other safe read-only commands here
        ]

        if command not in SAFE_SMI_COMMANDS:
            print(f"Unsafe SMI command 0x{command:02X} blocked for safety")
            return False

        try:
            # Write to APMC port to trigger SMI
            # This requires direct port I/O access
            import io_port  # Would need ctypes or python-ioport package
            io_port.outb(command, SMMAccess.SMM_PORT)
            return True
        except ImportError:
            print("SMI trigger requires io_port module (not installed)")
            return False
        except Exception as e:
            print(f"Error triggering SMI: {e}")
            return False

    @staticmethod
    def get_smm_info() -> Dict[str, Any]:
        """Get SMM configuration information (read-only)"""
        info = {
            'available': SMMAccess.is_available(),
            'apmc_port': hex(SMMAccess.SMM_PORT),
        }

        # Try to read SMM control MSR
        smm_ctl = MSRAccess.read_smm_monitor_ctl()
        if smm_ctl:
            info['monitor_control'] = smm_ctl

        return info


class MemoryMappedIO:
    """Memory-Mapped I/O access"""

    @staticmethod
    def read_mmio(address: int, size: int = 4) -> Optional[int]:
        """
        Read from memory-mapped I/O region

        Args:
            address: Physical memory address
            size: Number of bytes to read (1, 2, 4, or 8)

        Returns:
            Value read or None on error
        """
        if not os.path.exists("/dev/mem"):
            return None

        if os.geteuid() != 0:
            return None

        try:
            with open("/dev/mem", "rb") as f:
                f.seek(address)
                data = f.read(size)

                if len(data) != size:
                    return None

                # Unpack based on size
                fmt = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}[size]
                return struct.unpack(f'<{fmt}', data)[0]
        except Exception as e:
            print(f"Error reading MMIO 0x{address:X}: {e}")
            return None

    @staticmethod
    def write_mmio(address: int, value: int, size: int = 4) -> bool:
        """
        Write to memory-mapped I/O region (DANGEROUS)

        Args:
            address: Physical memory address
            value: Value to write
            size: Number of bytes to write (1, 2, 4, or 8)

        Returns:
            True on success, False on error
        """
        if not os.path.exists("/dev/mem"):
            return False

        if os.geteuid() != 0:
            return False

        try:
            fmt = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}[size]
            data = struct.pack(f'<{fmt}', value)

            with open("/dev/mem", "r+b") as f:
                f.seek(address)
                f.write(data)
                return True
        except Exception as e:
            print(f"Error writing MMIO 0x{address:X}: {e}")
            return False


class HardwareAccess:
    """Unified hardware access interface"""

    @staticmethod
    def check_capabilities() -> Dict[str, bool]:
        """Check what hardware access levels are available"""
        return {
            'userspace': True,
            'root': os.geteuid() == 0,
            'msr': MSRAccess.is_available(),
            'smm': SMMAccess.is_available(),
            'mmio': os.path.exists("/dev/mem") and os.geteuid() == 0,
        }

    @staticmethod
    def get_access_level() -> AccessLevel:
        """Determine current access level"""
        if os.geteuid() != 0:
            return AccessLevel.USERSPACE

        if MSRAccess.is_available():
            # If we have MSR access, we likely have kernel-level access
            return AccessLevel.MSR

        return AccessLevel.KERNEL

    @staticmethod
    def elevate_to_msr() -> bool:
        """Attempt to elevate to MSR access level"""
        if MSRAccess.is_available():
            return True

        # Try to load MSR module
        try:
            subprocess.run(['modprobe', 'msr'], check=True,
                         capture_output=True, timeout=5)
            return MSRAccess.is_available()
        except:
            return False

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Get comprehensive hardware access information"""
        caps = HardwareAccess.check_capabilities()

        info = {
            'capabilities': caps,
            'current_level': HardwareAccess.get_access_level().value,
            'root': os.geteuid() == 0,
            'uid': os.geteuid(),
        }

        if caps['msr']:
            info['msr'] = {
                'available_cpus': MSRAccess.get_all_cpus(),
                'feature_control': MSRAccess.read_feature_control(),
            }

        if caps['smm']:
            info['smm'] = SMMAccess.get_smm_info()

        return info


# Safety warnings
if __name__ == "__main__":
    print("=" * 80)
    print("DSMIL HARDWARE ACCESS MODULE")
    print("=" * 80)
    print()
    print("⚠️  WARNING: This module provides low-level hardware access")
    print("⚠️  MSR writes and SMM triggers can damage hardware or brick the system")
    print("⚠️  Only use with extreme caution and proper knowledge")
    print()

    info = HardwareAccess.get_hardware_info()

    print(f"Current Access Level: {info['current_level'].upper()}")
    print(f"Root Access: {info['root']}")
    print(f"UID: {info['uid']}")
    print()

    print("Capabilities:")
    for cap, available in info['capabilities'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {cap.upper()}")
    print()

    if info.get('msr'):
        print("MSR Access Details:")
        print(f"  CPUs: {info['msr']['available_cpus']}")
        if info['msr']['feature_control']:
            fc = info['msr']['feature_control']
            print(f"  Feature Control: 0x{fc['raw_value']:016X}")
            print(f"    Locked: {fc['locked']}")
            print(f"    VMX in SMX: {fc['vmx_inside_smx']}")
            print(f"    VMX out SMX: {fc['vmx_outside_smx']}")
        print()

    if info.get('smm'):
        print("SMM Access Details:")
        print(f"  Available: {info['smm']['available']}")
        print(f"  APMC Port: {info['smm']['apmc_port']}")
        if 'monitor_control' in info['smm']:
            smc = info['smm']['monitor_control']
            print(f"  Monitor Valid: {smc['valid']}")
            print(f"  MSEG Base: 0x{smc['mseg_base']:X}")
