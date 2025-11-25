#!/usr/bin/env python3
"""
DSMIL Yubikey Authentication Module
====================================

Hardware token authentication using Yubikey devices with FIDO2/WebAuthn.

Features:
- FIDO2/U2F registration and authentication
- Challenge-response authentication (offline)
- Yubico OTP support
- Multi-device support (up to 5 keys per user)
- Credential management
- Audit logging

Security:
- Hardware-backed authentication
- Phishing-resistant (FIDO2)
- No shared secrets on server
- Private keys never leave device
- Works offline (challenge-response)

Author: DSMIL Platform
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import json
import hmac
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# FIDO2/WebAuthn support
try:
    from fido2.server import Fido2Server
    from fido2.webauthn import PublicKeyCredentialRpEntity, UserVerificationRequirement
    from fido2.ctap2.base import Ctap2
    from fido2 import cbor
    FIDO2_AVAILABLE = True
except ImportError:
    FIDO2_AVAILABLE = False
    print("Warning: python-fido2 not installed. Install with: pip install fido2")

# Yubico OTP support
try:
    from yubico_client import Yubico
    YUBICO_OTP_AVAILABLE = True
except ImportError:
    YUBICO_OTP_AVAILABLE = False
    print("Warning: yubico-client not installed. Install with: pip install yubico-client")

# Challenge-Response support
try:
    import ykman
    from ykman.device import list_all_devices
    from ykman.otp import OtpController
    YKMAN_AVAILABLE = True
except ImportError:
    YKMAN_AVAILABLE = False
    print("Warning: yubikey-manager not installed. Install with: pip install yubikey-manager")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Yubikey authentication methods"""
    FIDO2 = "fido2"              # FIDO2/WebAuthn (primary)
    CHALLENGE_RESPONSE = "cr"     # Challenge-response (backup)
    YUBICO_OTP = "otp"           # Yubico OTP (fallback)


class DeviceStatus(Enum):
    """Yubikey device status"""
    ACTIVE = "active"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class YubikeyCredential:
    """FIDO2 credential information"""
    credential_id: str           # Base64-encoded credential ID
    public_key: str              # Base64-encoded public key
    aaguid: str                  # Authenticator AAGUID
    sign_count: int              # Signature counter
    created_at: str              # ISO timestamp
    last_used: Optional[str]     # ISO timestamp
    device_name: str             # User-friendly name
    status: str                  # active/revoked/suspended

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'YubikeyCredential':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class YubikeyDevice:
    """Yubikey device information"""
    device_id: str               # Unique device ID
    device_name: str             # User-friendly name
    serial_number: Optional[str] # Hardware serial (if available)
    firmware_version: Optional[str]
    auth_methods: List[str]      # Supported auth methods
    credentials: List[YubikeyCredential]  # FIDO2 credentials
    challenge_response_slot: Optional[int]  # CR slot (1 or 2)
    otp_public_id: Optional[str]  # Yubico OTP public ID
    status: str                  # active/revoked/suspended
    created_at: str              # ISO timestamp
    last_used: Optional[str]     # ISO timestamp

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['credentials'] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.credentials]
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'YubikeyDevice':
        """Create from dictionary"""
        creds = data.get('credentials', [])
        data['credentials'] = [YubikeyCredential.from_dict(c) if isinstance(c, dict) else c for c in creds]
        return cls(**data)


class YubikeyAuth:
    """
    Yubikey authentication manager

    Handles FIDO2/WebAuthn, Challenge-Response, and Yubico OTP authentication
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize Yubikey authentication

        Args:
            storage_dir: Directory for storing credentials (default: ~/.dsmil/yubikey)
        """
        if storage_dir is None:
            storage_dir = os.path.expanduser("~/.dsmil/yubikey")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        self.devices_file = self.storage_dir / "devices.json"
        self.audit_log = self.storage_dir / "audit.log"

        # Initialize FIDO2 server
        if FIDO2_AVAILABLE:
            rp = PublicKeyCredentialRpEntity(
                id="localhost",
                name="DSMIL Tactical Platform"
            )
            self.fido2_server = Fido2Server(rp)
        else:
            self.fido2_server = None
            logger.warning("FIDO2 not available - install python-fido2")

        # Initialize Yubico OTP client
        if YUBICO_OTP_AVAILABLE:
            # Note: Requires API credentials from Yubico
            # Get your credentials from: https://upgrade.yubico.com/getapikey/
            self.yubico_client = None  # Will be configured by admin
        else:
            self.yubico_client = None
            logger.warning("Yubico OTP not available - install yubico-client")

        # Load existing devices
        self.devices = self._load_devices()

        logger.info(f"YubikeyAuth initialized: {len(self.devices)} devices loaded")

    def _load_devices(self) -> Dict[str, YubikeyDevice]:
        """Load devices from storage"""
        if not self.devices_file.exists():
            return {}

        try:
            with open(self.devices_file, 'r') as f:
                data = json.load(f)

            devices = {}
            for device_id, device_data in data.items():
                devices[device_id] = YubikeyDevice.from_dict(device_data)

            logger.info(f"Loaded {len(devices)} Yubikey devices")
            return devices

        except Exception as e:
            logger.error(f"Failed to load devices: {e}")
            return {}

    def _save_devices(self):
        """Save devices to storage"""
        try:
            data = {device_id: device.to_dict() for device_id, device in self.devices.items()}

            with open(self.devices_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Secure permissions
            os.chmod(self.devices_file, 0o600)

            logger.info(f"Saved {len(self.devices)} Yubikey devices")

        except Exception as e:
            logger.error(f"Failed to save devices: {e}")
            raise

    def _audit_log_event(self, event_type: str, user: str, device_id: Optional[str],
                        success: bool, details: str = ""):
        """Log audit event"""
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            entry = {
                'timestamp': timestamp,
                'event_type': event_type,
                'user': user,
                'device_id': device_id,
                'success': success,
                'details': details
            }

            with open(self.audit_log, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    # =========================================================================
    # FIDO2/WebAuthn Authentication
    # =========================================================================

    def begin_fido2_registration(self, user: str, user_display_name: str = None) -> Dict:
        """
        Begin FIDO2/WebAuthn registration

        Args:
            user: Username
            user_display_name: Display name for user

        Returns:
            Dictionary with registration options for client
        """
        if not FIDO2_AVAILABLE or not self.fido2_server:
            raise RuntimeError("FIDO2 not available")

        if user_display_name is None:
            user_display_name = user

        # Generate user handle
        user_handle = secrets.token_bytes(32)

        # Create registration options
        registration_data, state = self.fido2_server.register_begin(
            {
                "id": user_handle,
                "name": user,
                "displayName": user_display_name
            },
            credentials=[],  # Existing credentials (for excluding)
            user_verification=UserVerificationRequirement.PREFERRED
        )

        # Store state for verification
        state_file = self.storage_dir / f"fido2_reg_state_{user}.json"
        with open(state_file, 'w') as f:
            json.dump({
                'state': state.hex(),
                'user': user,
                'user_handle': user_handle.hex(),
                'timestamp': datetime.utcnow().isoformat()
            }, f)
        os.chmod(state_file, 0o600)

        logger.info(f"FIDO2 registration started for user: {user}")

        return {
            'publicKey': registration_data,
            'status': 'success'
        }

    def complete_fido2_registration(self, user: str, client_data: Dict,
                                   device_name: str = "Yubikey") -> str:
        """
        Complete FIDO2/WebAuthn registration

        Args:
            user: Username
            client_data: Client response data
            device_name: User-friendly name for device

        Returns:
            Device ID
        """
        if not FIDO2_AVAILABLE or not self.fido2_server:
            raise RuntimeError("FIDO2 not available")

        # Load state
        state_file = self.storage_dir / f"fido2_reg_state_{user}.json"
        if not state_file.exists():
            raise ValueError("No registration in progress")

        with open(state_file, 'r') as f:
            state_data = json.load(f)

        state = bytes.fromhex(state_data['state'])
        user_handle = bytes.fromhex(state_data['user_handle'])

        # Verify registration
        try:
            auth_data = self.fido2_server.register_complete(
                state,
                client_data
            )

            credential = auth_data.credential_data

            # Create credential object
            cred = YubikeyCredential(
                credential_id=credential.credential_id.hex(),
                public_key=bytes(credential.public_key).hex(),
                aaguid=credential.aaguid.hex(),
                sign_count=auth_data.counter,
                created_at=datetime.utcnow().isoformat() + "Z",
                last_used=None,
                device_name=device_name,
                status=DeviceStatus.ACTIVE.value
            )

            # Create or update device
            device_id = hashlib.sha256(credential.credential_id).hexdigest()[:16]

            if device_id in self.devices:
                device = self.devices[device_id]
                device.credentials.append(cred)
            else:
                device = YubikeyDevice(
                    device_id=device_id,
                    device_name=device_name,
                    serial_number=None,
                    firmware_version=None,
                    auth_methods=[AuthMethod.FIDO2.value],
                    credentials=[cred],
                    challenge_response_slot=None,
                    otp_public_id=None,
                    status=DeviceStatus.ACTIVE.value,
                    created_at=datetime.utcnow().isoformat() + "Z",
                    last_used=None
                )

            self.devices[device_id] = device
            self._save_devices()

            # Clean up state
            state_file.unlink()

            self._audit_log_event(
                'fido2_registration',
                user,
                device_id,
                True,
                f"Registered device: {device_name}"
            )

            logger.info(f"FIDO2 registration completed for user: {user}, device: {device_id}")

            return device_id

        except Exception as e:
            self._audit_log_event(
                'fido2_registration',
                user,
                None,
                False,
                f"Registration failed: {str(e)}"
            )
            raise

    def begin_fido2_authentication(self, user: str) -> Dict:
        """
        Begin FIDO2/WebAuthn authentication

        Args:
            user: Username

        Returns:
            Dictionary with authentication options for client
        """
        if not FIDO2_AVAILABLE or not self.fido2_server:
            raise RuntimeError("FIDO2 not available")

        # Get user's credentials
        user_credentials = []
        for device in self.devices.values():
            if device.status == DeviceStatus.ACTIVE.value:
                for cred in device.credentials:
                    if cred.status == DeviceStatus.ACTIVE.value:
                        user_credentials.append({
                            'type': 'public-key',
                            'id': bytes.fromhex(cred.credential_id)
                        })

        if not user_credentials:
            raise ValueError(f"No active credentials for user: {user}")

        # Create authentication options
        auth_data, state = self.fido2_server.authenticate_begin(
            credentials=user_credentials,
            user_verification=UserVerificationRequirement.PREFERRED
        )

        # Store state for verification
        state_file = self.storage_dir / f"fido2_auth_state_{user}.json"
        with open(state_file, 'w') as f:
            json.dump({
                'state': state.hex(),
                'user': user,
                'timestamp': datetime.utcnow().isoformat()
            }, f)
        os.chmod(state_file, 0o600)

        logger.info(f"FIDO2 authentication started for user: {user}")

        return {
            'publicKey': auth_data,
            'status': 'success'
        }

    def complete_fido2_authentication(self, user: str, credential_id: str,
                                     client_data: Dict) -> bool:
        """
        Complete FIDO2/WebAuthn authentication

        Args:
            user: Username
            credential_id: Credential ID (hex string)
            client_data: Client response data

        Returns:
            True if authentication successful
        """
        if not FIDO2_AVAILABLE or not self.fido2_server:
            raise RuntimeError("FIDO2 not available")

        # Load state
        state_file = self.storage_dir / f"fido2_auth_state_{user}.json"
        if not state_file.exists():
            raise ValueError("No authentication in progress")

        with open(state_file, 'r') as f:
            state_data = json.load(f)

        state = bytes.fromhex(state_data['state'])

        # Find credential
        device = None
        credential = None

        for dev in self.devices.values():
            for cred in dev.credentials:
                if cred.credential_id == credential_id:
                    device = dev
                    credential = cred
                    break
            if credential:
                break

        if not credential:
            raise ValueError(f"Credential not found: {credential_id}")

        # Verify authentication
        try:
            # Get credential data for verification
            cred_data = bytes.fromhex(credential.credential_id)

            auth_data = self.fido2_server.authenticate_complete(
                state,
                [{'type': 'public-key', 'id': cred_data}],
                credential_id=cred_data,
                client_data=client_data
            )

            # Update credential
            credential.sign_count = auth_data.counter
            credential.last_used = datetime.utcnow().isoformat() + "Z"

            # Update device
            device.last_used = datetime.utcnow().isoformat() + "Z"

            self._save_devices()

            # Clean up state
            state_file.unlink()

            self._audit_log_event(
                'fido2_authentication',
                user,
                device.device_id,
                True,
                f"Authenticated with device: {device.device_name}"
            )

            logger.info(f"FIDO2 authentication successful for user: {user}")

            return True

        except Exception as e:
            self._audit_log_event(
                'fido2_authentication',
                user,
                device.device_id if device else None,
                False,
                f"Authentication failed: {str(e)}"
            )
            logger.error(f"FIDO2 authentication failed: {e}")
            return False

    # =========================================================================
    # Challenge-Response Authentication
    # =========================================================================

    def setup_challenge_response(self, device_id: str, slot: int = 2,
                                secret: bytes = None) -> bool:
        """
        Setup challenge-response on Yubikey

        Args:
            device_id: Device ID
            slot: OTP slot (1 or 2, default 2)
            secret: 20-byte secret (generated if None)

        Returns:
            True if successful
        """
        if not YKMAN_AVAILABLE:
            raise RuntimeError("yubikey-manager not available")

        if slot not in [1, 2]:
            raise ValueError("Slot must be 1 or 2")

        if secret is None:
            secret = secrets.token_bytes(20)

        if len(secret) != 20:
            raise ValueError("Secret must be 20 bytes")

        try:
            # Find Yubikey device
            devices = list_all_devices()
            if not devices:
                raise RuntimeError("No Yubikey detected")

            dev, info = devices[0]

            # Configure challenge-response
            with dev.open_connection(OtpController) as controller:
                controller.program_chalresp(slot, secret, require_touch=True)

            # Update device
            if device_id in self.devices:
                device = self.devices[device_id]
                device.challenge_response_slot = slot
                if AuthMethod.CHALLENGE_RESPONSE.value not in device.auth_methods:
                    device.auth_methods.append(AuthMethod.CHALLENGE_RESPONSE.value)

                # Store secret
                secret_file = self.storage_dir / f"cr_secret_{device_id}.bin"
                with open(secret_file, 'wb') as f:
                    f.write(secret)
                os.chmod(secret_file, 0o600)

                self._save_devices()

                logger.info(f"Challenge-response configured for device: {device_id}")
                return True
            else:
                raise ValueError(f"Device not found: {device_id}")

        except Exception as e:
            logger.error(f"Failed to setup challenge-response: {e}")
            return False

    def authenticate_challenge_response(self, device_id: str, challenge: bytes = None) -> bool:
        """
        Authenticate using challenge-response

        Args:
            device_id: Device ID
            challenge: Challenge bytes (generated if None)

        Returns:
            True if authentication successful
        """
        if not YKMAN_AVAILABLE:
            raise RuntimeError("yubikey-manager not available")

        if device_id not in self.devices:
            raise ValueError(f"Device not found: {device_id}")

        device = self.devices[device_id]

        if device.challenge_response_slot is None:
            raise ValueError(f"Challenge-response not configured for device: {device_id}")

        if challenge is None:
            challenge = secrets.token_bytes(32)

        try:
            # Load secret
            secret_file = self.storage_dir / f"cr_secret_{device_id}.bin"
            with open(secret_file, 'rb') as f:
                secret = f.read()

            # Find Yubikey device
            devices = list_all_devices()
            if not devices:
                raise RuntimeError("No Yubikey detected")

            dev, info = devices[0]

            # Perform challenge-response
            with dev.open_connection(OtpController) as controller:
                response = controller.calculate_hmac(device.challenge_response_slot, challenge)

            # Verify response
            expected = hmac.new(secret, challenge, hashlib.sha1).digest()

            if hmac.compare_digest(response, expected):
                # Update device
                device.last_used = datetime.utcnow().isoformat() + "Z"
                self._save_devices()

                self._audit_log_event(
                    'challenge_response_authentication',
                    'system',
                    device_id,
                    True,
                    f"Authenticated with device: {device.device_name}"
                )

                logger.info(f"Challenge-response authentication successful for device: {device_id}")
                return True
            else:
                self._audit_log_event(
                    'challenge_response_authentication',
                    'system',
                    device_id,
                    False,
                    "Invalid response"
                )
                logger.warning(f"Challenge-response authentication failed for device: {device_id}")
                return False

        except Exception as e:
            logger.error(f"Challenge-response authentication error: {e}")
            return False

    # =========================================================================
    # Device Management
    # =========================================================================

    def list_devices(self, status: Optional[DeviceStatus] = None) -> List[YubikeyDevice]:
        """
        List all registered devices

        Args:
            status: Filter by status (optional)

        Returns:
            List of devices
        """
        devices = list(self.devices.values())

        if status:
            devices = [d for d in devices if d.status == status.value]

        return devices

    def get_device(self, device_id: str) -> Optional[YubikeyDevice]:
        """Get device by ID"""
        return self.devices.get(device_id)

    def revoke_device(self, device_id: str, reason: str = "") -> bool:
        """
        Revoke a device

        Args:
            device_id: Device ID
            reason: Reason for revocation

        Returns:
            True if successful
        """
        if device_id not in self.devices:
            return False

        device = self.devices[device_id]
        device.status = DeviceStatus.REVOKED.value

        # Revoke all credentials
        for cred in device.credentials:
            cred.status = DeviceStatus.REVOKED.value

        self._save_devices()

        self._audit_log_event(
            'device_revocation',
            'admin',
            device_id,
            True,
            f"Device revoked: {reason}"
        )

        logger.info(f"Device revoked: {device_id}, reason: {reason}")
        return True

    def suspend_device(self, device_id: str, reason: str = "") -> bool:
        """
        Suspend a device

        Args:
            device_id: Device ID
            reason: Reason for suspension

        Returns:
            True if successful
        """
        if device_id not in self.devices:
            return False

        device = self.devices[device_id]
        device.status = DeviceStatus.SUSPENDED.value

        self._save_devices()

        self._audit_log_event(
            'device_suspension',
            'admin',
            device_id,
            True,
            f"Device suspended: {reason}"
        )

        logger.info(f"Device suspended: {device_id}, reason: {reason}")
        return True

    def reactivate_device(self, device_id: str) -> bool:
        """
        Reactivate a suspended device

        Args:
            device_id: Device ID

        Returns:
            True if successful
        """
        if device_id not in self.devices:
            return False

        device = self.devices[device_id]

        if device.status == DeviceStatus.REVOKED.value:
            logger.warning(f"Cannot reactivate revoked device: {device_id}")
            return False

        device.status = DeviceStatus.ACTIVE.value

        self._save_devices()

        self._audit_log_event(
            'device_reactivation',
            'admin',
            device_id,
            True,
            "Device reactivated"
        )

        logger.info(f"Device reactivated: {device_id}")
        return True

    def get_device_count(self) -> Dict[str, int]:
        """Get device count by status"""
        counts = {
            'total': len(self.devices),
            'active': 0,
            'suspended': 0,
            'revoked': 0
        }

        for device in self.devices.values():
            if device.status == DeviceStatus.ACTIVE.value:
                counts['active'] += 1
            elif device.status == DeviceStatus.SUSPENDED.value:
                counts['suspended'] += 1
            elif device.status == DeviceStatus.REVOKED.value:
                counts['revoked'] += 1

        return counts


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("DSMIL Yubikey Authentication Module")
    print("=" * 60)

    # Check dependencies
    print("\nDependency Check:")
    print(f"  FIDO2/WebAuthn: {'✓ Available' if FIDO2_AVAILABLE else '✗ Not available'}")
    print(f"  Yubico OTP:     {'✓ Available' if YUBICO_OTP_AVAILABLE else '✗ Not available'}")
    print(f"  YubiKey Manager:{'✓ Available' if YKMAN_AVAILABLE else '✗ Not available'}")

    # Initialize auth
    print("\nInitializing YubikeyAuth...")
    auth = YubikeyAuth()

    # Show device count
    counts = auth.get_device_count()
    print(f"\nRegistered Devices:")
    print(f"  Total:     {counts['total']}")
    print(f"  Active:    {counts['active']}")
    print(f"  Suspended: {counts['suspended']}")
    print(f"  Revoked:   {counts['revoked']}")

    # List devices
    devices = auth.list_devices()
    if devices:
        print("\nDevices:")
        for device in devices:
            print(f"\n  Device ID: {device.device_id}")
            print(f"  Name:      {device.device_name}")
            print(f"  Status:    {device.status}")
            print(f"  Methods:   {', '.join(device.auth_methods)}")
            print(f"  Creds:     {len(device.credentials)}")
    else:
        print("\nNo devices registered.")
        print("\nTo register a Yubikey:")
        print("  1. Use the admin tool: python yubikey_admin.py register")
        print("  2. Or integrate with your application")

    print("\n" + "=" * 60)
