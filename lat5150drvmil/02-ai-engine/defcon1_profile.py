#!/usr/bin/env python3
"""
DSMIL DEFCON1 Security Profile
===============================

Maximum security profile requiring dual YubiKey authentication.

Classification: TOP SECRET // FOR OFFICIAL USE ONLY
Threat Level: DEFCON 1 (Maximum Readiness)

Features:
- Dual YubiKey authentication (both keys must pass)
- 4-person authorization required
- Executive approval mandatory
- Emergency-only access
- 1-hour maximum session duration
- Continuous authentication monitoring
- Comprehensive audit logging

Author: DSMIL Platform
Date: 2025-11-25
"""

import os
import sys
import json
import secrets
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import YubiKey authentication
try:
    from yubikey_auth import YubikeyAuth, DeviceStatus, AuthMethod
    YUBIKEY_AVAILABLE = True
except ImportError:
    YUBIKEY_AVAILABLE = False
    print("ERROR: yubikey_auth module not found")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """System threat levels"""
    GREEN = 1      # Peacetime
    YELLOW = 2     # Elevated
    ORANGE = 3     # High
    RED = 4        # Imminent
    DEFCON_1 = 5   # Maximum readiness


class AuthorizationLevel(Enum):
    """Authorization levels for DEFCON1"""
    STANDARD = 1          # Standard operator
    SUPERVISOR = 2        # Supervisor
    COMMANDER = 3         # Commander
    EXECUTIVE = 4         # Executive/Presidential


@dataclass
class Authorizer:
    """Individual authorizer information"""
    user_id: str
    name: str
    role: str
    level: AuthorizationLevel
    yubikey_device_id: str
    authorized_at: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class DEFCON1Session:
    """DEFCON1 authentication session"""
    session_id: str
    user_id: str
    threat_level: str
    primary_yubikey_id: str
    secondary_yubikey_id: str
    authorizers: List[Authorizer]
    created_at: str
    expires_at: str
    last_auth_check: str
    is_active: bool
    access_restrictions: List[str]
    audit_trail: List[Dict]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['authorizers'] = [asdict(a) for a in self.authorizers]
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'DEFCON1Session':
        """Create from dictionary"""
        authorizers = [Authorizer(**a) for a in data.get('authorizers', [])]
        data['authorizers'] = authorizers
        return cls(**data)


class DEFCON1Profile:
    """
    DEFCON1 Security Profile Manager

    Implements maximum security authentication requiring:
    - Two YubiKeys (both must authenticate)
    - 4-person authorization
    - Executive approval
    - 1-hour session duration
    - Continuous authentication
    """

    # DEFCON1 Requirements
    REQUIRED_YUBIKEYS = 2
    REQUIRED_AUTHORIZERS = 4
    SESSION_DURATION_HOURS = 1
    CONTINUOUS_AUTH_INTERVAL_MINUTES = 5
    MIN_EXECUTIVE_AUTHORIZERS = 1

    def __init__(self, storage_dir: str = None):
        """
        Initialize DEFCON1 profile

        Args:
            storage_dir: Directory for storing profile data (default: ~/.dsmil/defcon1)
        """
        if not YUBIKEY_AVAILABLE:
            raise RuntimeError("YubiKey authentication not available")

        if storage_dir is None:
            storage_dir = os.path.expanduser("~/.dsmil/defcon1")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        self.sessions_file = self.storage_dir / "sessions.json"
        self.audit_log = self.storage_dir / "defcon1_audit.log"
        self.config_file = self.storage_dir / "defcon1_config.json"

        # Initialize YubiKey authentication
        self.yubikey_auth = YubikeyAuth()

        # Load sessions
        self.sessions = self._load_sessions()

        # Load configuration
        self.config = self._load_config()

        logger.info("DEFCON1 Profile initialized")

    def _load_sessions(self) -> Dict[str, DEFCON1Session]:
        """Load active sessions"""
        if not self.sessions_file.exists():
            return {}

        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)

            sessions = {}
            for session_id, session_data in data.items():
                sessions[session_id] = DEFCON1Session.from_dict(session_data)

            logger.info(f"Loaded {len(sessions)} DEFCON1 sessions")
            return sessions

        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            return {}

    def _save_sessions(self):
        """Save active sessions"""
        try:
            data = {sid: session.to_dict() for sid, session in self.sessions.items()}

            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2)

            os.chmod(self.sessions_file, 0o600)
            logger.info(f"Saved {len(self.sessions)} DEFCON1 sessions")

        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            raise

    def _load_config(self) -> Dict:
        """Load DEFCON1 configuration"""
        default_config = {
            'threat_level': ThreatLevel.DEFCON_1.name,
            'required_yubikeys': self.REQUIRED_YUBIKEYS,
            'required_authorizers': self.REQUIRED_AUTHORIZERS,
            'session_duration_hours': self.SESSION_DURATION_HOURS,
            'continuous_auth_interval_minutes': self.CONTINUOUS_AUTH_INTERVAL_MINUTES,
            'access_restrictions': [
                'EMERGENCY_ONLY',
                'EXECUTIVE_AUTHORIZATION_REQUIRED',
                'DUAL_YUBIKEY_MANDATORY',
                'CONTINUOUS_MONITORING',
                'FULL_AUDIT_TRAIL'
            ],
            'authorized_executives': []
        }

        if not self.config_file.exists():
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            os.chmod(self.config_file, 0o600)
            return default_config

        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return default_config

    def _audit_log_event(self, event_type: str, session_id: Optional[str],
                        success: bool, details: str = ""):
        """Log audit event"""
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            entry = {
                'timestamp': timestamp,
                'event_type': event_type,
                'session_id': session_id,
                'threat_level': ThreatLevel.DEFCON_1.name,
                'success': success,
                'details': details
            }

            with open(self.audit_log, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def begin_defcon1_authentication(self, user_id: str, user_display_name: str = None) -> Dict:
        """
        Begin DEFCON1 authentication process

        Requires user to present two YubiKeys for registration

        Args:
            user_id: User identifier
            user_display_name: Display name for user

        Returns:
            Dictionary with authentication state and next steps
        """
        if user_display_name is None:
            user_display_name = user_id

        session_id = secrets.token_hex(16)

        logger.info(f"DEFCON1 authentication initiated for user: {user_id}")

        self._audit_log_event(
            'defcon1_auth_begin',
            session_id,
            True,
            f"User: {user_id} initiated DEFCON1 authentication"
        )

        return {
            'status': 'success',
            'session_id': session_id,
            'user_id': user_id,
            'threat_level': ThreatLevel.DEFCON_1.name,
            'requirements': {
                'yubikeys_required': self.REQUIRED_YUBIKEYS,
                'authorizers_required': self.REQUIRED_AUTHORIZERS,
                'executive_authorizers_required': self.MIN_EXECUTIVE_AUTHORIZERS,
                'session_duration_hours': self.SESSION_DURATION_HOURS
            },
            'next_step': 'authenticate_primary_yubikey',
            'message': 'Insert PRIMARY YubiKey and complete authentication'
        }

    def authenticate_dual_yubikey(self, session_id: str, user_id: str,
                                  primary_device_id: str, secondary_device_id: str,
                                  primary_credential: Dict, secondary_credential: Dict) -> bool:
        """
        Authenticate using two YubiKeys

        Both keys must successfully authenticate via FIDO2 challenge-response

        Args:
            session_id: Session identifier
            user_id: User identifier
            primary_device_id: Primary YubiKey device ID
            secondary_device_id: Secondary YubiKey device ID
            primary_credential: Primary YubiKey authentication credential
            secondary_credential: Secondary YubiKey authentication credential

        Returns:
            True if both YubiKeys authenticate successfully
        """
        logger.info(f"Dual YubiKey authentication for session: {session_id}")

        # Verify devices are different
        if primary_device_id == secondary_device_id:
            logger.error("Primary and secondary YubiKeys must be different devices")
            self._audit_log_event(
                'dual_yubikey_auth',
                session_id,
                False,
                "Same device used for both YubiKeys"
            )
            return False

        # Authenticate primary YubiKey
        try:
            primary_auth = self.yubikey_auth.complete_fido2_authentication(
                user_id,
                primary_device_id,
                primary_credential
            )

            if not primary_auth:
                logger.error("Primary YubiKey authentication failed")
                self._audit_log_event(
                    'dual_yubikey_auth',
                    session_id,
                    False,
                    "Primary YubiKey authentication failed"
                )
                return False

        except Exception as e:
            logger.error(f"Primary YubiKey authentication error: {e}")
            self._audit_log_event(
                'dual_yubikey_auth',
                session_id,
                False,
                f"Primary YubiKey error: {str(e)}"
            )
            return False

        # Authenticate secondary YubiKey
        try:
            secondary_auth = self.yubikey_auth.complete_fido2_authentication(
                user_id,
                secondary_device_id,
                secondary_credential
            )

            if not secondary_auth:
                logger.error("Secondary YubiKey authentication failed")
                self._audit_log_event(
                    'dual_yubikey_auth',
                    session_id,
                    False,
                    "Secondary YubiKey authentication failed"
                )
                return False

        except Exception as e:
            logger.error(f"Secondary YubiKey authentication error: {e}")
            self._audit_log_event(
                'dual_yubikey_auth',
                session_id,
                False,
                f"Secondary YubiKey error: {str(e)}"
            )
            return False

        # Both YubiKeys authenticated successfully
        logger.info(f"Dual YubiKey authentication successful for session: {session_id}")
        self._audit_log_event(
            'dual_yubikey_auth',
            session_id,
            True,
            f"Primary: {primary_device_id}, Secondary: {secondary_device_id}"
        )

        return True

    def add_authorizer(self, session_id: str, authorizer: Authorizer,
                      yubikey_credential: Dict) -> bool:
        """
        Add an authorizer to the DEFCON1 session

        Each authorizer must authenticate with their YubiKey

        Args:
            session_id: Session identifier
            authorizer: Authorizer information
            yubikey_credential: YubiKey authentication credential

        Returns:
            True if authorizer added successfully
        """
        logger.info(f"Adding authorizer: {authorizer.name} ({authorizer.role}) to session: {session_id}")

        # Verify authorizer's YubiKey
        try:
            auth_result = self.yubikey_auth.complete_fido2_authentication(
                authorizer.user_id,
                authorizer.yubikey_device_id,
                yubikey_credential
            )

            if not auth_result:
                logger.error(f"Authorizer YubiKey authentication failed: {authorizer.name}")
                self._audit_log_event(
                    'add_authorizer',
                    session_id,
                    False,
                    f"Authorizer {authorizer.name} authentication failed"
                )
                return False

        except Exception as e:
            logger.error(f"Authorizer authentication error: {e}")
            self._audit_log_event(
                'add_authorizer',
                session_id,
                False,
                f"Authorizer {authorizer.name} error: {str(e)}"
            )
            return False

        # Record authorization
        authorizer.authorized_at = datetime.utcnow().isoformat() + "Z"
        authorizer.signature = secrets.token_hex(32)

        logger.info(f"Authorizer {authorizer.name} authenticated successfully")
        self._audit_log_event(
            'add_authorizer',
            session_id,
            True,
            f"Authorizer: {authorizer.name}, Role: {authorizer.role}, Level: {authorizer.level.name}"
        )

        return True

    def create_defcon1_session(self, session_id: str, user_id: str,
                               primary_yubikey_id: str, secondary_yubikey_id: str,
                               authorizers: List[Authorizer]) -> DEFCON1Session:
        """
        Create a DEFCON1 session after all authentication requirements are met

        Args:
            session_id: Session identifier
            user_id: User identifier
            primary_yubikey_id: Primary YubiKey device ID
            secondary_yubikey_id: Secondary YubiKey device ID
            authorizers: List of authenticated authorizers

        Returns:
            DEFCON1Session object

        Raises:
            ValueError: If requirements not met
        """
        # Validate requirements
        if len(authorizers) < self.REQUIRED_AUTHORIZERS:
            raise ValueError(f"Insufficient authorizers: {len(authorizers)} < {self.REQUIRED_AUTHORIZERS}")

        # Verify executive authorization
        executive_count = sum(1 for a in authorizers if a.level == AuthorizationLevel.EXECUTIVE)
        if executive_count < self.MIN_EXECUTIVE_AUTHORIZERS:
            raise ValueError(f"Insufficient executive authorizers: {executive_count} < {self.MIN_EXECUTIVE_AUTHORIZERS}")

        # Create session
        now = datetime.utcnow()
        expires = now + timedelta(hours=self.SESSION_DURATION_HOURS)

        session = DEFCON1Session(
            session_id=session_id,
            user_id=user_id,
            threat_level=ThreatLevel.DEFCON_1.name,
            primary_yubikey_id=primary_yubikey_id,
            secondary_yubikey_id=secondary_yubikey_id,
            authorizers=authorizers,
            created_at=now.isoformat() + "Z",
            expires_at=expires.isoformat() + "Z",
            last_auth_check=now.isoformat() + "Z",
            is_active=True,
            access_restrictions=self.config['access_restrictions'],
            audit_trail=[]
        )

        # Save session
        self.sessions[session_id] = session
        self._save_sessions()

        logger.info(f"DEFCON1 session created: {session_id}, expires: {expires.isoformat()}")
        self._audit_log_event(
            'session_created',
            session_id,
            True,
            f"User: {user_id}, Authorizers: {len(authorizers)}, Expires: {expires.isoformat()}"
        )

        return session

    def verify_continuous_auth(self, session_id: str, user_id: str,
                              primary_credential: Dict, secondary_credential: Dict) -> bool:
        """
        Verify continuous authentication for active session

        Required every 5 minutes during DEFCON1 session

        Args:
            session_id: Session identifier
            user_id: User identifier
            primary_credential: Primary YubiKey credential
            secondary_credential: Secondary YubiKey credential

        Returns:
            True if continuous authentication successful
        """
        if session_id not in self.sessions:
            logger.error(f"Session not found: {session_id}")
            return False

        session = self.sessions[session_id]

        # Check if session expired
        expires = datetime.fromisoformat(session.expires_at.replace('Z', '+00:00'))
        if datetime.utcnow().replace(tzinfo=expires.tzinfo) > expires:
            logger.error(f"Session expired: {session_id}")
            session.is_active = False
            self._save_sessions()
            return False

        # Verify dual YubiKey authentication
        auth_result = self.authenticate_dual_yubikey(
            session_id,
            user_id,
            session.primary_yubikey_id,
            session.secondary_yubikey_id,
            primary_credential,
            secondary_credential
        )

        if auth_result:
            session.last_auth_check = datetime.utcnow().isoformat() + "Z"
            self._save_sessions()

            logger.info(f"Continuous authentication successful: {session_id}")
            self._audit_log_event(
                'continuous_auth',
                session_id,
                True,
                "Continuous authentication verified"
            )
            return True
        else:
            logger.error(f"Continuous authentication failed: {session_id}")
            session.is_active = False
            self._save_sessions()

            self._audit_log_event(
                'continuous_auth',
                session_id,
                False,
                "Continuous authentication failed - session terminated"
            )
            return False

    def terminate_session(self, session_id: str, reason: str = "Manual termination"):
        """
        Terminate a DEFCON1 session

        Args:
            session_id: Session identifier
            reason: Reason for termination
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
            self._save_sessions()

            logger.info(f"Session terminated: {session_id}, reason: {reason}")
            self._audit_log_event(
                'session_terminated',
                session_id,
                True,
                reason
            )

    def get_session(self, session_id: str) -> Optional[DEFCON1Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def list_active_sessions(self) -> List[DEFCON1Session]:
        """List all active DEFCON1 sessions"""
        return [s for s in self.sessions.values() if s.is_active]

    def get_session_status(self, session_id: str) -> Dict:
        """Get detailed session status"""
        if session_id not in self.sessions:
            return {'status': 'error', 'message': 'Session not found'}

        session = self.sessions[session_id]

        # Check expiration
        expires = datetime.fromisoformat(session.expires_at.replace('Z', '+00:00'))
        now = datetime.utcnow().replace(tzinfo=expires.tzinfo)
        time_remaining = (expires - now).total_seconds()

        return {
            'status': 'active' if session.is_active else 'inactive',
            'session_id': session.session_id,
            'user_id': session.user_id,
            'threat_level': session.threat_level,
            'created_at': session.created_at,
            'expires_at': session.expires_at,
            'time_remaining_seconds': max(0, time_remaining),
            'last_auth_check': session.last_auth_check,
            'authorizers': [
                {
                    'name': a.name,
                    'role': a.role,
                    'level': a.level.name,
                    'authorized_at': a.authorized_at
                }
                for a in session.authorizers
            ],
            'access_restrictions': session.access_restrictions
        }


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command line interface for DEFCON1 profile"""
    print("=" * 80)
    print("DSMIL DEFCON1 Security Profile")
    print("=" * 80)
    print()
    print("Classification: TOP SECRET // FOR OFFICIAL USE ONLY")
    print("Threat Level: DEFCON 1 (Maximum Readiness)")
    print()
    print("=" * 80)
    print()

    # Initialize profile
    print("Initializing DEFCON1 profile...")
    profile = DEFCON1Profile()

    print()
    print("✅ DEFCON1 Profile initialized")
    print()
    print("Requirements:")
    print(f"  - YubiKeys Required: {profile.REQUIRED_YUBIKEYS}")
    print(f"  - Authorizers Required: {profile.REQUIRED_AUTHORIZERS}")
    print(f"  - Executive Authorizers: {profile.MIN_EXECUTIVE_AUTHORIZERS}")
    print(f"  - Session Duration: {profile.SESSION_DURATION_HOURS} hour(s)")
    print(f"  - Continuous Auth Interval: {profile.CONTINUOUS_AUTH_INTERVAL_MINUTES} minutes")
    print()
    print("Access Restrictions:")
    for restriction in profile.config['access_restrictions']:
        print(f"  - {restriction}")
    print()

    # List active sessions
    active_sessions = profile.list_active_sessions()
    print(f"Active DEFCON1 Sessions: {len(active_sessions)}")

    if active_sessions:
        print()
        for session in active_sessions:
            print(f"  Session ID: {session.session_id}")
            print(f"  User: {session.user_id}")
            print(f"  Created: {session.created_at}")
            print(f"  Expires: {session.expires_at}")
            print(f"  Authorizers: {len(session.authorizers)}")
            print()

    print("=" * 80)


if __name__ == "__main__":
    main()
