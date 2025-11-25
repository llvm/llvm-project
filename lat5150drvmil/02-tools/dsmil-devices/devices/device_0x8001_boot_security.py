#!/usr/bin/env python3
"""
Device 0x8001: Boot Security Controller

Manages secure boot validation, boot chain integrity, and boot policy
enforcement for the MIL-SPEC platform.

Device ID: 0x8001
Group: 0 (Core Security)
Risk Level: MONITORED (80% safe for READ operations)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from typing import Dict, List, Optional, Any


class BootStage(object):
    """Boot stage identifiers"""
    BIOS = 0
    BOOTLOADER = 1
    KERNEL = 2
    INITRAMFS = 3
    USERSPACE = 4


class BootPolicy(object):
    """Boot policy flags"""
    SECURE_BOOT_ENABLED = 0x0001
    MEASURED_BOOT = 0x0002
    VERIFIED_BOOT = 0x0004
    ANTI_ROLLBACK = 0x0008
    RECOVERY_MODE = 0x0010
    DEBUG_DISABLED = 0x0020
    SIGNATURE_REQUIRED = 0x0040
    TPM_REQUIRED = 0x0080


class BootSecurityDevice(DSMILDeviceBase):
    """Boot Security Controller (0x8001)"""

    # Register map
    REG_BOOT_STATUS = 0x00
    REG_BOOT_POLICY = 0x04
    REG_BOOT_STAGE = 0x08
    REG_BOOT_MEASUREMENTS = 0x0C
    REG_SIGNATURE_STATUS = 0x10
    REG_ROLLBACK_INDEX = 0x14
    REG_RECOVERY_STATUS = 0x18
    REG_BOOT_FLAGS = 0x1C

    # Status bits
    STATUS_SECURE_BOOT_ACTIVE = 0x01
    STATUS_BOOT_VERIFIED = 0x02
    STATUS_MEASUREMENTS_VALID = 0x04
    STATUS_SIGNATURE_VALID = 0x08
    STATUS_ROLLBACK_PROTECTED = 0x10
    STATUS_RECOVERY_MODE = 0x20
    STATUS_TPM_AVAILABLE = 0x40
    STATUS_BOOT_LOCKED = 0x80

    def __init__(self, device_id: int = 0x8001,
                 name: str = "Boot Security",
                 description: str = "Secure Boot and Boot Chain Integrity"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.boot_policy = 0
        self.current_boot_stage = BootStage.USERSPACE
        self.boot_measurements = {}
        self.signature_status = {}
        self.rollback_index = 0

        # Register map
        self.register_map = {
            "BOOT_STATUS": {
                "offset": self.REG_BOOT_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Boot security status"
            },
            "BOOT_POLICY": {
                "offset": self.REG_BOOT_POLICY,
                "size": 4,
                "access": "RO",
                "description": "Active boot policy flags"
            },
            "BOOT_STAGE": {
                "offset": self.REG_BOOT_STAGE,
                "size": 4,
                "access": "RO",
                "description": "Current boot stage"
            },
            "BOOT_MEASUREMENTS": {
                "offset": self.REG_BOOT_MEASUREMENTS,
                "size": 4,
                "access": "RO",
                "description": "Boot measurements count"
            },
            "SIGNATURE_STATUS": {
                "offset": self.REG_SIGNATURE_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Signature verification status"
            },
            "ROLLBACK_INDEX": {
                "offset": self.REG_ROLLBACK_INDEX,
                "size": 4,
                "access": "RO",
                "description": "Current rollback protection index"
            },
            "RECOVERY_STATUS": {
                "offset": self.REG_RECOVERY_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Recovery mode status"
            },
            "BOOT_FLAGS": {
                "offset": self.REG_BOOT_FLAGS,
                "size": 4,
                "access": "RO",
                "description": "Boot configuration flags"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Boot Security device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Read boot policy
            self.boot_policy = self._read_boot_policy()

            # Read current boot stage
            self.current_boot_stage = BootStage.USERSPACE

            # Initialize boot measurements
            self.boot_measurements = {
                BootStage.BIOS: {"measured": True, "valid": True},
                BootStage.BOOTLOADER: {"measured": True, "valid": True},
                BootStage.KERNEL: {"measured": True, "valid": True},
                BootStage.INITRAMFS: {"measured": True, "valid": True},
            }

            # Initialize signature status
            self.signature_status = {
                "bios": True,
                "bootloader": True,
                "kernel": True,
            }

            # Read rollback index
            self.rollback_index = 1

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "boot_policy": self.boot_policy,
                "boot_stage": self.current_boot_stage,
                "measurements": len(self.boot_measurements),
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_ONLY,
            DeviceCapability.STATUS_REPORTING,
            DeviceCapability.EVENT_MONITORING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        status_reg = self._read_status_register()

        return {
            "secure_boot_active": bool(status_reg & self.STATUS_SECURE_BOOT_ACTIVE),
            "boot_verified": bool(status_reg & self.STATUS_BOOT_VERIFIED),
            "measurements_valid": bool(status_reg & self.STATUS_MEASUREMENTS_VALID),
            "signature_valid": bool(status_reg & self.STATUS_SIGNATURE_VALID),
            "rollback_protected": bool(status_reg & self.STATUS_ROLLBACK_PROTECTED),
            "recovery_mode": bool(status_reg & self.STATUS_RECOVERY_MODE),
            "tpm_available": bool(status_reg & self.STATUS_TPM_AVAILABLE),
            "boot_locked": bool(status_reg & self.STATUS_BOOT_LOCKED),
            "boot_stage": self.current_boot_stage,
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            # Simulated register reads
            if register == "BOOT_STATUS":
                value = self._read_status_register()
            elif register == "BOOT_POLICY":
                value = self.boot_policy
            elif register == "BOOT_STAGE":
                value = self.current_boot_stage
            elif register == "BOOT_MEASUREMENTS":
                value = len(self.boot_measurements)
            elif register == "SIGNATURE_STATUS":
                value = sum(1 for v in self.signature_status.values() if v)
            elif register == "ROLLBACK_INDEX":
                value = self.rollback_index
            elif register == "RECOVERY_STATUS":
                value = 0  # Not in recovery
            elif register == "BOOT_FLAGS":
                value = self.boot_policy
            else:
                value = 0

            self._record_operation(True)
            return OperationResult(True, data={
                "register": register,
                "value": value,
                "hex": f"0x{value:08X}",
            })

        except Exception as e:
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    # Boot Security specific operations

    def get_boot_policy(self) -> OperationResult:
        """Get active boot policy"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        policy_flags = []
        if self.boot_policy & BootPolicy.SECURE_BOOT_ENABLED:
            policy_flags.append("SECURE_BOOT")
        if self.boot_policy & BootPolicy.MEASURED_BOOT:
            policy_flags.append("MEASURED_BOOT")
        if self.boot_policy & BootPolicy.VERIFIED_BOOT:
            policy_flags.append("VERIFIED_BOOT")
        if self.boot_policy & BootPolicy.ANTI_ROLLBACK:
            policy_flags.append("ANTI_ROLLBACK")
        if self.boot_policy & BootPolicy.DEBUG_DISABLED:
            policy_flags.append("DEBUG_DISABLED")
        if self.boot_policy & BootPolicy.SIGNATURE_REQUIRED:
            policy_flags.append("SIGNATURE_REQUIRED")
        if self.boot_policy & BootPolicy.TPM_REQUIRED:
            policy_flags.append("TPM_REQUIRED")

        self._record_operation(True)
        return OperationResult(True, data={
            "policy": self.boot_policy,
            "flags": policy_flags,
        })

    def get_boot_measurements(self) -> OperationResult:
        """Get boot chain measurements"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        measurements = []
        for stage, data in self.boot_measurements.items():
            measurements.append({
                "stage": stage,
                "stage_name": self._get_stage_name(stage),
                "measured": data["measured"],
                "valid": data["valid"],
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "measurements": measurements,
            "total": len(measurements),
            "all_valid": all(m["valid"] for m in measurements),
        })

    def verify_boot_stage(self, stage: int) -> OperationResult:
        """
        Verify a specific boot stage

        Args:
            stage: Boot stage (BootStage)

        Returns:
            OperationResult with verification status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if stage not in self.boot_measurements:
            return OperationResult(False, error="Invalid boot stage")

        measurement = self.boot_measurements[stage]

        self._record_operation(True)
        return OperationResult(True, data={
            "stage": stage,
            "stage_name": self._get_stage_name(stage),
            "verified": measurement["valid"],
            "measured": measurement["measured"],
        })

    def get_signature_status(self) -> OperationResult:
        """Get signature verification status for boot components"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        signatures = []
        for component, valid in self.signature_status.items():
            signatures.append({
                "component": component,
                "signature_valid": valid,
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "signatures": signatures,
            "all_valid": all(self.signature_status.values()),
        })

    def get_rollback_index(self) -> OperationResult:
        """Get current rollback protection index"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "rollback_index": self.rollback_index,
            "protected": bool(self.boot_policy & BootPolicy.ANTI_ROLLBACK),
        })

    def check_recovery_status(self) -> OperationResult:
        """Check if system is in recovery mode"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        status = self._read_status_register()
        in_recovery = bool(status & self.STATUS_RECOVERY_MODE)

        self._record_operation(True)
        return OperationResult(True, data={
            "recovery_mode": in_recovery,
            "boot_verified": bool(status & self.STATUS_BOOT_VERIFIED),
        })

    def get_boot_chain_summary(self) -> OperationResult:
        """Get comprehensive boot chain summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        status = self._read_status_register()

        summary = {
            "secure_boot": bool(status & self.STATUS_SECURE_BOOT_ACTIVE),
            "boot_verified": bool(status & self.STATUS_BOOT_VERIFIED),
            "measurements_valid": bool(status & self.STATUS_MEASUREMENTS_VALID),
            "signatures_valid": all(self.signature_status.values()),
            "rollback_protected": bool(status & self.STATUS_ROLLBACK_PROTECTED),
            "rollback_index": self.rollback_index,
            "recovery_mode": bool(status & self.STATUS_RECOVERY_MODE),
            "tpm_integrated": bool(status & self.STATUS_TPM_AVAILABLE),
            "boot_locked": bool(status & self.STATUS_BOOT_LOCKED),
            "current_stage": self._get_stage_name(self.current_boot_stage),
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read boot status register (simulated)"""
        status = (
            self.STATUS_SECURE_BOOT_ACTIVE |
            self.STATUS_BOOT_VERIFIED |
            self.STATUS_MEASUREMENTS_VALID |
            self.STATUS_SIGNATURE_VALID |
            self.STATUS_ROLLBACK_PROTECTED |
            self.STATUS_TPM_AVAILABLE |
            self.STATUS_BOOT_LOCKED
        )
        return status

    def _read_boot_policy(self) -> int:
        """Read boot policy (simulated)"""
        policy = (
            BootPolicy.SECURE_BOOT_ENABLED |
            BootPolicy.MEASURED_BOOT |
            BootPolicy.VERIFIED_BOOT |
            BootPolicy.ANTI_ROLLBACK |
            BootPolicy.DEBUG_DISABLED |
            BootPolicy.SIGNATURE_REQUIRED |
            BootPolicy.TPM_REQUIRED
        )
        return policy

    def _get_stage_name(self, stage: int) -> str:
        """Get boot stage name"""
        stage_names = {
            BootStage.BIOS: "BIOS",
            BootStage.BOOTLOADER: "Bootloader",
            BootStage.KERNEL: "Kernel",
            BootStage.INITRAMFS: "InitramFS",
            BootStage.USERSPACE: "Userspace",
        }
        return stage_names.get(stage, "Unknown")
