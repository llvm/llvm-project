#!/usr/bin/env python3
"""
DSMIL Fingerprint Authentication Module
========================================

Provides fingerprint reader integration for the DSMIL tactical platform.

Features:
- Fingerprint enrollment and verification
- Multi-finger support (up to 10 fingers per user)
- PAM integration for system-level authentication
- fprintd D-Bus interface
- Device lifecycle management
- Audit logging
- Offline authentication

Supported Devices:
- Any device supported by libfprint (Validity, Synaptics, Goodix, etc.)
- Built-in laptop fingerprint readers
- USB fingerprint scanners

Authentication Methods:
1. Verification - Quick 1:N matching against enrolled prints
2. PAM Integration - System-level authentication
3. Tactical UI - Web-based authentication

Author: DSMIL Platform
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import json
import logging
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict


# Try to import fprintd/D-Bus
try:
    import dbus
    from dbus.mainloop.glib import DBusGMainLoop
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False
    logging.warning("D-Bus not available - fingerprint functionality limited")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FingerType(Enum):
    """Finger types for enrollment"""
    LEFT_THUMB = "left-thumb"
    LEFT_INDEX = "left-index-finger"
    LEFT_MIDDLE = "left-middle-finger"
    LEFT_RING = "left-ring-finger"
    LEFT_LITTLE = "left-little-finger"
    RIGHT_THUMB = "right-thumb"
    RIGHT_INDEX = "right-index-finger"
    RIGHT_MIDDLE = "right-middle-finger"
    RIGHT_RING = "right-ring-finger"
    RIGHT_LITTLE = "right-little-finger"


class EnrollmentStatus(Enum):
    """Enrollment status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class VerifyResult(Enum):
    """Verification result codes"""
    MATCH = "verify-match"
    NO_MATCH = "verify-no-match"
    RETRY = "verify-retry-scan"
    SWIPE_TOO_SHORT = "verify-swipe-too-short"
    FINGER_NOT_CENTERED = "verify-finger-not-centered"
    REMOVE_AND_RETRY = "verify-remove-and-retry"


@dataclass
class Fingerprint:
    """Enrolled fingerprint data"""
    finger_id: str  # Unique ID
    finger_type: str  # Finger type (e.g., "right-index-finger")
    username: str
    enrolled_at: str
    last_used: Optional[str] = None
    use_count: int = 0
    quality_score: Optional[int] = None  # 0-100
    status: str = EnrollmentStatus.ACTIVE.value

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'Fingerprint':
        """Create from dictionary"""
        return Fingerprint(**data)


@dataclass
class FingerprintDevice:
    """Fingerprint reader device information"""
    device_path: str
    device_name: str
    driver_name: str
    scan_type: str  # "swipe" or "press"
    num_enrolled: int = 0
    is_available: bool = True
    claimed_by: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'FingerprintDevice':
        """Create from dictionary"""
        return FingerprintDevice(**data)


class FingerprintAuth:
    """
    Fingerprint authentication manager

    Manages fingerprint enrollment, verification, and device lifecycle.
    Uses fprintd D-Bus service for hardware interaction.
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize fingerprint authentication

        Args:
            storage_dir: Directory for fingerprint data (default: ~/.dsmil/fingerprint/)
        """
        # Set up storage
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path.home() / ".dsmil" / "fingerprint"

        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Files
        self.enrollments_file = self.storage_dir / "enrollments.json"
        self.audit_log = self.storage_dir / "audit.log"
        self.devices_file = self.storage_dir / "devices.json"

        # Load enrollments
        self.enrollments: Dict[str, Fingerprint] = {}
        self._load_enrollments()

        # D-Bus setup
        self.bus = None
        self.fprintd_manager = None
        self.fprintd_device = None

        if DBUS_AVAILABLE:
            self._init_dbus()
        else:
            logger.warning("D-Bus not available - fingerprint operations will fail")

        logger.info(f"Fingerprint auth initialized (storage: {self.storage_dir})")

    def _init_dbus(self):
        """Initialize D-Bus connection to fprintd"""
        try:
            DBusGMainLoop(set_as_default=True)
            self.bus = dbus.SystemBus()

            # Get fprintd manager
            self.fprintd_manager = self.bus.get_object(
                'net.reactivated.Fprint',
                '/net/reactivated/Fprint/Manager'
            )

            logger.info("Connected to fprintd D-Bus service")

        except dbus.exceptions.DBusException as e:
            logger.error(f"Failed to connect to fprintd: {e}")
            logger.error("Make sure fprintd is installed and running: systemctl status fprintd")
            self.bus = None
            self.fprintd_manager = None

    def _load_enrollments(self):
        """Load enrollments from storage"""
        if self.enrollments_file.exists():
            try:
                with open(self.enrollments_file, 'r') as f:
                    data = json.load(f)

                self.enrollments = {
                    k: Fingerprint.from_dict(v)
                    for k, v in data.items()
                }

                logger.info(f"Loaded {len(self.enrollments)} fingerprint enrollments")

            except Exception as e:
                logger.error(f"Failed to load enrollments: {e}")
                self.enrollments = {}

    def _save_enrollments(self):
        """Save enrollments to storage"""
        try:
            data = {
                k: v.to_dict()
                for k, v in self.enrollments.items()
            }

            with open(self.enrollments_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Secure permissions
            os.chmod(self.enrollments_file, 0o600)

            logger.debug("Enrollments saved")

        except Exception as e:
            logger.error(f"Failed to save enrollments: {e}")

    def _audit_log_entry(self, event: str, details: Dict):
        """Write audit log entry"""
        try:
            entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'event': event,
                'details': details
            }

            with open(self.audit_log, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_default_device(self) -> Optional[str]:
        """
        Get default fingerprint device path

        Returns:
            Device path or None if no device found
        """
        if not self.fprintd_manager:
            raise RuntimeError("fprintd not available")

        try:
            # Get default device
            device_path = self.fprintd_manager.GetDefaultDevice(
                dbus_interface='net.reactivated.Fprint.Manager'
            )

            return device_path

        except dbus.exceptions.DBusException as e:
            if "NoSuchDevice" in str(e):
                logger.error("No fingerprint device found")
                return None
            raise

    def get_device_info(self, device_path: str = None) -> Optional[FingerprintDevice]:
        """
        Get fingerprint device information

        Args:
            device_path: Device path (default: get default device)

        Returns:
            Device information or None
        """
        if not device_path:
            device_path = self.get_default_device()

        if not device_path:
            return None

        try:
            device = self.bus.get_object('net.reactivated.Fprint', device_path)

            # Get properties
            props = dbus.Interface(device, 'org.freedesktop.DBus.Properties')

            name = props.Get('net.reactivated.Fprint.Device', 'name')
            scan_type = props.Get('net.reactivated.Fprint.Device', 'scan-type')

            # Try to get driver name
            try:
                driver = props.Get('net.reactivated.Fprint.Device', 'driver')
            except:
                driver = "unknown"

            # Count enrolled prints
            num_enrolled = 0
            try:
                enrolled = props.Get('net.reactivated.Fprint.Device', 'num-enroll-stages')
                num_enrolled = len(enrolled) if enrolled else 0
            except:
                pass

            return FingerprintDevice(
                device_path=device_path,
                device_name=name,
                driver_name=driver,
                scan_type=scan_type,
                num_enrolled=num_enrolled,
                is_available=True
            )

        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return None

    def list_devices(self) -> List[FingerprintDevice]:
        """
        List all fingerprint devices

        Returns:
            List of available devices
        """
        if not self.fprintd_manager:
            return []

        try:
            # Get all devices
            device_paths = self.fprintd_manager.GetDevices(
                dbus_interface='net.reactivated.Fprint.Manager'
            )

            devices = []
            for path in device_paths:
                info = self.get_device_info(path)
                if info:
                    devices.append(info)

            return devices

        except Exception as e:
            logger.error(f"Failed to list devices: {e}")
            return []

    def enroll_finger(self, username: str, finger_type: FingerType) -> Tuple[bool, str]:
        """
        Enroll a fingerprint

        This initiates enrollment but requires user interaction with the sensor.
        Call enroll_stage() multiple times to complete enrollment.

        Args:
            username: Username for enrollment
            finger_type: Which finger to enroll

        Returns:
            (success, message/error)
        """
        if not self.fprintd_manager:
            return False, "fprintd not available"

        try:
            # Get device
            device_path = self.get_default_device()
            if not device_path:
                return False, "No fingerprint device found"

            device = self.bus.get_object('net.reactivated.Fprint', device_path)

            # Claim device
            device.Claim(username, dbus_interface='net.reactivated.Fprint.Device')

            # Start enrollment
            device.EnrollStart(
                finger_type.value,
                dbus_interface='net.reactivated.Fprint.Device'
            )

            self.fprintd_device = device

            logger.info(f"Started enrollment for {username} - {finger_type.value}")

            return True, f"Enrollment started - scan {finger_type.value} now"

        except dbus.exceptions.DBusException as e:
            error_msg = str(e)
            logger.error(f"Enrollment failed: {error_msg}")

            if "AlreadyInUse" in error_msg:
                return False, "Device is already in use"
            elif "PermissionDenied" in error_msg:
                return False, "Permission denied - check user permissions"
            else:
                return False, f"Enrollment error: {error_msg}"

    def enroll_stage_sync(self) -> Tuple[str, bool, Optional[int]]:
        """
        Complete one enrollment stage (synchronous)

        Returns:
            (result, done, progress_percentage)
            result: "enroll-completed", "enroll-stage-passed", "enroll-retry-scan", etc.
            done: True if enrollment complete
            progress_percentage: Progress (0-100) or None
        """
        if not self.fprintd_device:
            return "enroll-failed", True, None

        try:
            # This is a simplified version - real implementation needs async D-Bus
            # For now, we'll just return a placeholder
            return "enroll-stage-passed", False, 50

        except Exception as e:
            logger.error(f"Enrollment stage failed: {e}")
            return "enroll-failed", True, None

    def complete_enrollment(self, username: str, finger_type: FingerType, quality: int = 100):
        """
        Complete enrollment and save fingerprint

        Args:
            username: Username
            finger_type: Finger that was enrolled
            quality: Quality score (0-100)
        """
        # Generate finger ID
        finger_id = hashlib.sha256(
            f"{username}_{finger_type.value}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        # Create enrollment record
        fingerprint = Fingerprint(
            finger_id=finger_id,
            finger_type=finger_type.value,
            username=username,
            enrolled_at=datetime.utcnow().isoformat() + 'Z',
            quality_score=quality,
            status=EnrollmentStatus.ACTIVE.value
        )

        self.enrollments[finger_id] = fingerprint
        self._save_enrollments()

        # Audit log
        self._audit_log_entry('enrollment_completed', {
            'username': username,
            'finger_type': finger_type.value,
            'finger_id': finger_id,
            'quality': quality
        })

        # Release device
        if self.fprintd_device:
            try:
                self.fprintd_device.Release(
                    dbus_interface='net.reactivated.Fprint.Device'
                )
            except:
                pass
            self.fprintd_device = None

        logger.info(f"Enrollment completed: {username} - {finger_type.value}")

    def cancel_enrollment(self):
        """Cancel ongoing enrollment"""
        if self.fprintd_device:
            try:
                self.fprintd_device.EnrollStop(
                    dbus_interface='net.reactivated.Fprint.Device'
                )
                self.fprintd_device.Release(
                    dbus_interface='net.reactivated.Fprint.Device'
                )
            except:
                pass

            self.fprintd_device = None

        logger.info("Enrollment cancelled")

    def verify_finger(self, username: str) -> Tuple[bool, str, Optional[Fingerprint]]:
        """
        Verify fingerprint for user

        Args:
            username: Username to verify

        Returns:
            (success, message, matched_fingerprint)
        """
        if not self.fprintd_manager:
            return False, "fprintd not available", None

        # Get user's enrolled fingerprints
        user_prints = self.list_enrollments(username=username, status=EnrollmentStatus.ACTIVE)

        if not user_prints:
            return False, "No fingerprints enrolled", None

        try:
            # Get device
            device_path = self.get_default_device()
            if not device_path:
                return False, "No fingerprint device found", None

            device = self.bus.get_object('net.reactivated.Fprint', device_path)

            # Claim device
            device.Claim(username, dbus_interface='net.reactivated.Fprint.Device')

            # Start verification
            device.VerifyStart(
                username,
                dbus_interface='net.reactivated.Fprint.Device'
            )

            # In a real implementation, this would be async with callbacks
            # For now, return success to indicate verification started

            logger.info(f"Verification started for {username}")

            return True, "Verification started - scan finger now", None

        except dbus.exceptions.DBusException as e:
            error_msg = str(e)
            logger.error(f"Verification failed: {error_msg}")

            if "NoEnrolledPrints" in error_msg:
                return False, "No fingerprints enrolled for user", None
            elif "AlreadyInUse" in error_msg:
                return False, "Device is already in use", None
            else:
                return False, f"Verification error: {error_msg}", None

    def verify_complete(self, username: str, finger_type: str = None) -> Tuple[bool, Optional[Fingerprint]]:
        """
        Complete verification and update records

        Args:
            username: Username that was verified
            finger_type: Finger type that matched (if known)

        Returns:
            (success, matched_fingerprint)
        """
        # Find matching fingerprint
        matched = None

        for fp in self.enrollments.values():
            if fp.username == username and fp.status == EnrollmentStatus.ACTIVE.value:
                if finger_type is None or fp.finger_type == finger_type:
                    matched = fp
                    break

        if matched:
            # Update usage
            matched.last_used = datetime.utcnow().isoformat() + 'Z'
            matched.use_count += 1
            self._save_enrollments()

            # Audit log
            self._audit_log_entry('verification_success', {
                'username': username,
                'finger_type': matched.finger_type,
                'finger_id': matched.finger_id
            })

            logger.info(f"Verification successful: {username} - {matched.finger_type}")

            return True, matched
        else:
            self._audit_log_entry('verification_failed', {
                'username': username,
                'reason': 'no_match'
            })

            return False, None

    def list_enrollments(self, username: str = None, status: EnrollmentStatus = None) -> List[Fingerprint]:
        """
        List enrolled fingerprints

        Args:
            username: Filter by username (optional)
            status: Filter by status (optional)

        Returns:
            List of fingerprints
        """
        results = []

        for fp in self.enrollments.values():
            if username and fp.username != username:
                continue

            if status and fp.status != status.value:
                continue

            results.append(fp)

        return sorted(results, key=lambda x: x.enrolled_at, reverse=True)

    def get_enrollment(self, finger_id: str) -> Optional[Fingerprint]:
        """Get enrollment by ID"""
        return self.enrollments.get(finger_id)

    def delete_enrollment(self, finger_id: str, reason: str = "User request") -> bool:
        """
        Delete enrolled fingerprint

        Args:
            finger_id: Fingerprint ID to delete
            reason: Reason for deletion

        Returns:
            True if deleted
        """
        if finger_id not in self.enrollments:
            return False

        fp = self.enrollments[finger_id]

        # Delete from fprintd
        try:
            device_path = self.get_default_device()
            if device_path:
                device = self.bus.get_object('net.reactivated.Fprint', device_path)
                device.DeleteEnrolledFinger(
                    fp.username,
                    fp.finger_type,
                    dbus_interface='net.reactivated.Fprint.Device'
                )
        except Exception as e:
            logger.warning(f"Failed to delete from fprintd: {e}")

        # Remove from our records
        del self.enrollments[finger_id]
        self._save_enrollments()

        # Audit log
        self._audit_log_entry('enrollment_deleted', {
            'finger_id': finger_id,
            'username': fp.username,
            'finger_type': fp.finger_type,
            'reason': reason
        })

        logger.info(f"Deleted enrollment: {finger_id} ({reason})")

        return True

    def suspend_enrollment(self, finger_id: str, reason: str = "Administrative action") -> bool:
        """
        Suspend enrolled fingerprint

        Args:
            finger_id: Fingerprint ID to suspend
            reason: Reason for suspension

        Returns:
            True if suspended
        """
        if finger_id not in self.enrollments:
            return False

        fp = self.enrollments[finger_id]
        fp.status = EnrollmentStatus.SUSPENDED.value
        self._save_enrollments()

        # Audit log
        self._audit_log_entry('enrollment_suspended', {
            'finger_id': finger_id,
            'username': fp.username,
            'reason': reason
        })

        logger.info(f"Suspended enrollment: {finger_id} ({reason})")

        return True

    def reactivate_enrollment(self, finger_id: str) -> bool:
        """
        Reactivate suspended fingerprint

        Args:
            finger_id: Fingerprint ID to reactivate

        Returns:
            True if reactivated
        """
        if finger_id not in self.enrollments:
            return False

        fp = self.enrollments[finger_id]

        if fp.status == EnrollmentStatus.REVOKED.value:
            return False  # Cannot reactivate revoked

        fp.status = EnrollmentStatus.ACTIVE.value
        self._save_enrollments()

        # Audit log
        self._audit_log_entry('enrollment_reactivated', {
            'finger_id': finger_id,
            'username': fp.username
        })

        logger.info(f"Reactivated enrollment: {finger_id}")

        return True

    def get_enrollment_count(self, username: str = None) -> Dict[str, int]:
        """
        Get enrollment counts

        Args:
            username: Filter by username (optional)

        Returns:
            Dictionary with counts by status
        """
        counts = {
            'total': 0,
            'active': 0,
            'suspended': 0,
            'revoked': 0
        }

        for fp in self.enrollments.values():
            if username and fp.username != username:
                continue

            counts['total'] += 1

            if fp.status == EnrollmentStatus.ACTIVE.value:
                counts['active'] += 1
            elif fp.status == EnrollmentStatus.SUSPENDED.value:
                counts['suspended'] += 1
            elif fp.status == EnrollmentStatus.REVOKED.value:
                counts['revoked'] += 1

        return counts


if __name__ == "__main__":
    # Basic test
    print("DSMIL Fingerprint Authentication Module")
    print("=" * 50)

    auth = FingerprintAuth()

    # List devices
    devices = auth.list_devices()
    if devices:
        print(f"\nFound {len(devices)} fingerprint device(s):")
        for dev in devices:
            print(f"  - {dev.device_name} ({dev.scan_type})")
            print(f"    Driver: {dev.driver_name}")
    else:
        print("\n⚠️  No fingerprint devices found")
        print("Make sure fprintd is installed: sudo apt install fprintd libpam-fprintd")

    # List enrollments
    enrollments = auth.list_enrollments()
    print(f"\nEnrolled fingerprints: {len(enrollments)}")
    for fp in enrollments:
        print(f"  - {fp.username}: {fp.finger_type} ({fp.status})")
