#!/usr/bin/env python3
"""
DSMIL DEFCON1 Administration Tool
==================================

Command-line tool for managing DEFCON1 authentication sessions

Classification: TOP SECRET // FOR OFFICIAL USE ONLY

Usage:
    ./defcon1_admin.py init-session <user-id>
    ./defcon1_admin.py list-sessions
    ./defcon1_admin.py session-status <session-id>
    ./defcon1_admin.py terminate-session <session-id>
    ./defcon1_admin.py test-dual-auth <user-id>

Author: DSMIL Platform
Date: 2025-11-25
"""

import sys
import argparse
import json
from datetime import datetime
from defcon1_profile import (
    DEFCON1Profile,
    Authorizer,
    AuthorizationLevel,
    ThreatLevel
)
from yubikey_auth import YubikeyAuth


class DEFCON1Admin:
    """DEFCON1 administration interface"""

    def __init__(self):
        self.profile = DEFCON1Profile()
        self.yubikey_auth = YubikeyAuth()

    def init_session(self, user_id: str):
        """Initialize a DEFCON1 authentication session"""
        print("=" * 80)
        print("DEFCON1 Session Initialization")
        print("=" * 80)
        print()
        print(f"User: {user_id}")
        print(f"Threat Level: {ThreatLevel.DEFCON_1.name}")
        print()

        # Begin authentication
        result = self.profile.begin_defcon1_authentication(user_id, user_id)

        print("✅ DEFCON1 authentication session initiated")
        print()
        print(f"Session ID: {result['session_id']}")
        print()
        print("Requirements:")
        print(f"  - YubiKeys: {result['requirements']['yubikeys_required']}")
        print(f"  - Authorizers: {result['requirements']['authorizers_required']}")
        print(f"  - Executive Authorizers: {result['requirements']['executive_authorizers_required']}")
        print(f"  - Session Duration: {result['requirements']['session_duration_hours']} hour(s)")
        print()
        print("Next Steps:")
        print("  1. Insert PRIMARY YubiKey")
        print("  2. Complete FIDO2 authentication")
        print("  3. Insert SECONDARY YubiKey")
        print("  4. Complete FIDO2 authentication")
        print("  5. Gather 4 authorizers (including 1 executive)")
        print("  6. Each authorizer authenticates with their YubiKey")
        print()
        print(f"Message: {result['message']}")
        print()
        print("=" * 80)

        return result

    def list_sessions(self):
        """List all active DEFCON1 sessions"""
        print("=" * 80)
        print("Active DEFCON1 Sessions")
        print("=" * 80)
        print()

        sessions = self.profile.list_active_sessions()

        if not sessions:
            print("No active DEFCON1 sessions")
            print()
            return

        for i, session in enumerate(sessions, 1):
            print(f"[{i}] Session ID: {session.session_id}")
            print(f"    User: {session.user_id}")
            print(f"    Threat Level: {session.threat_level}")
            print(f"    Created: {session.created_at}")
            print(f"    Expires: {session.expires_at}")
            print(f"    Primary YubiKey: {session.primary_yubikey_id}")
            print(f"    Secondary YubiKey: {session.secondary_yubikey_id}")
            print(f"    Authorizers: {len(session.authorizers)}")
            print(f"    Active: {session.is_active}")
            print()

            # List authorizers
            if session.authorizers:
                print("    Authorizers:")
                for auth in session.authorizers:
                    print(f"      - {auth.name} ({auth.role}) - Level: {auth.level.name}")
                    print(f"        Authorized: {auth.authorized_at}")
                print()

        print("=" * 80)

    def session_status(self, session_id: str):
        """Get detailed session status"""
        print("=" * 80)
        print("DEFCON1 Session Status")
        print("=" * 80)
        print()

        status = self.profile.get_session_status(session_id)

        if status.get('status') == 'error':
            print(f"❌ {status['message']}")
            print()
            return

        print(f"Session ID: {status['session_id']}")
        print(f"Status: {status['status'].upper()}")
        print(f"User: {status['user_id']}")
        print(f"Threat Level: {status['threat_level']}")
        print()
        print(f"Created: {status['created_at']}")
        print(f"Expires: {status['expires_at']}")
        print(f"Time Remaining: {int(status['time_remaining_seconds'])} seconds ({int(status['time_remaining_seconds'] / 60)} minutes)")
        print(f"Last Auth Check: {status['last_auth_check']}")
        print()

        # Authorizers
        print(f"Authorizers ({len(status['authorizers'])}):")
        for auth in status['authorizers']:
            print(f"  - {auth['name']} ({auth['role']})")
            print(f"    Level: {auth['level']}")
            print(f"    Authorized: {auth['authorized_at']}")
        print()

        # Access restrictions
        print("Access Restrictions:")
        for restriction in status['access_restrictions']:
            print(f"  - {restriction}")
        print()

        print("=" * 80)

    def terminate_session(self, session_id: str, reason: str = "Manual termination"):
        """Terminate a DEFCON1 session"""
        print("=" * 80)
        print("Terminate DEFCON1 Session")
        print("=" * 80)
        print()

        session = self.profile.get_session(session_id)
        if not session:
            print(f"❌ Session not found: {session_id}")
            print()
            return

        print(f"Session ID: {session_id}")
        print(f"User: {session.user_id}")
        print(f"Created: {session.created_at}")
        print(f"Reason: {reason}")
        print()

        confirm = input("Terminate session? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled")
            print()
            return

        self.profile.terminate_session(session_id, reason)

        print()
        print("✅ Session terminated")
        print()
        print("=" * 80)

    def test_dual_auth(self, user_id: str):
        """Test dual YubiKey authentication (simulation)"""
        print("=" * 80)
        print("Test Dual YubiKey Authentication")
        print("=" * 80)
        print()
        print("⚠️  This is a simulation - actual YubiKey authentication requires")
        print("    browser WebAuthn integration for FIDO2 challenges")
        print()

        # List registered YubiKeys
        devices = self.yubikey_auth.list_devices()
        if len(devices) < 2:
            print(f"❌ Insufficient YubiKeys registered: {len(devices)} < 2")
            print()
            print("Please register at least 2 YubiKeys:")
            print("  python3 yubikey_admin.py register --user", user_id)
            print()
            return

        print(f"Registered YubiKeys ({len(devices)}):")
        for i, device in enumerate(devices, 1):
            print(f"  [{i}] {device.device_name} (ID: {device.device_id})")
            print(f"      Status: {device.status}")
            print(f"      Methods: {', '.join(device.auth_methods)}")
            print()

        if len(devices) >= 2:
            primary = devices[0]
            secondary = devices[1]

            print("Dual Authentication Configuration:")
            print(f"  Primary YubiKey: {primary.device_name} ({primary.device_id})")
            print(f"  Secondary YubiKey: {secondary.device_name} ({secondary.device_id})")
            print()
            print("✅ Dual YubiKey configuration valid")
            print()
            print("Next Steps:")
            print("  1. Use init-session command to create DEFCON1 session")
            print("  2. Integrate with web interface for FIDO2 authentication")
            print("  3. Both YubiKeys must complete WebAuthn challenges")
            print()

        print("=" * 80)

    def demo_workflow(self):
        """Demonstrate complete DEFCON1 workflow"""
        print("=" * 80)
        print("DEFCON1 Authentication Workflow Demo")
        print("=" * 80)
        print()
        print("This demonstration shows the complete DEFCON1 authentication process.")
        print()
        print("=" * 80)
        print()

        # Check YubiKey availability
        devices = self.yubikey_auth.list_devices()
        print(f"Step 1: Check YubiKey Availability")
        print(f"  Registered YubiKeys: {len(devices)}")

        if len(devices) < 2:
            print(f"  ❌ Need at least 2 YubiKeys (have {len(devices)})")
            print()
            print("  Register additional YubiKeys:")
            print("    python3 yubikey_admin.py register")
            print()
            return

        print(f"  ✅ Sufficient YubiKeys available")
        print()

        # Show workflow steps
        print("Step 2: DEFCON1 Authentication Process")
        print()
        print("  2.1 User initiates DEFCON1 session")
        print("      → System generates session ID")
        print("      → Records threat level: DEFCON 1")
        print()
        print("  2.2 Primary YubiKey Authentication")
        print("      → User inserts primary YubiKey")
        print("      → Browser prompts for WebAuthn")
        print("      → User touches YubiKey sensor")
        print("      → FIDO2 challenge-response validates")
        print()
        print("  2.3 Secondary YubiKey Authentication")
        print("      → User removes primary, inserts secondary YubiKey")
        print("      → Browser prompts for WebAuthn again")
        print("      → User touches second YubiKey sensor")
        print("      → FIDO2 challenge-response validates")
        print()
        print("  2.4 Gather Authorizers (4 required)")
        print()

        # Example authorizers
        example_authorizers = [
            ("Authorizer 1", "Operator", AuthorizationLevel.STANDARD),
            ("Authorizer 2", "Supervisor", AuthorizationLevel.SUPERVISOR),
            ("Authorizer 3", "Commander", AuthorizationLevel.COMMANDER),
            ("Authorizer 4", "Executive", AuthorizationLevel.EXECUTIVE)
        ]

        for name, role, level in example_authorizers:
            print(f"      → {name} ({role})")
            print(f"        - Level: {level.name}")
            print(f"        - Authenticates with personal YubiKey")
            print(f"        - Provides digital signature")
            print()

        print("  2.5 Session Creation")
        print("      → Verify all requirements met")
        print("      → 2 YubiKeys authenticated ✅")
        print("      → 4 authorizers validated ✅")
        print("      → 1 executive authorizer ✅")
        print("      → Create session (1 hour duration)")
        print("      → Enable continuous monitoring")
        print()

        print("  2.6 Continuous Authentication (every 5 minutes)")
        print("      → User re-authenticates with both YubiKeys")
        print("      → Session remains active")
        print("      → If authentication fails → session terminates")
        print()

        print("  2.7 Session Expiration (after 1 hour)")
        print("      → Session automatically terminates")
        print("      → Full audit trail preserved")
        print("      → User must re-authenticate for new session")
        print()

        print("=" * 80)
        print()
        print("Access Restrictions During DEFCON1:")
        restrictions = [
            "EMERGENCY_ONLY - Only emergency operations permitted",
            "EXECUTIVE_AUTHORIZATION_REQUIRED - Executive approval mandatory",
            "DUAL_YUBIKEY_MANDATORY - Both YubiKeys must authenticate",
            "CONTINUOUS_MONITORING - Real-time authentication checks",
            "FULL_AUDIT_TRAIL - All actions logged and auditable"
        ]
        for restriction in restrictions:
            print(f"  • {restriction}")
        print()

        print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='DSMIL DEFCON1 Administration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  init-session <user-id>        Initialize DEFCON1 session
  list-sessions                 List active sessions
  session-status <session-id>   Get session status
  terminate-session <session-id> Terminate session
  test-dual-auth <user-id>      Test dual YubiKey configuration
  demo                          Show complete workflow demo

Examples:
  # Initialize DEFCON1 session
  ./defcon1_admin.py init-session tactical_user

  # List all active sessions
  ./defcon1_admin.py list-sessions

  # Check session status
  ./defcon1_admin.py session-status abc123def456

  # Terminate session
  ./defcon1_admin.py terminate-session abc123def456

  # Test dual YubiKey setup
  ./defcon1_admin.py test-dual-auth tactical_user

  # View workflow demonstration
  ./defcon1_admin.py demo
        """
    )

    parser.add_argument('command', help='Command to execute')
    parser.add_argument('args', nargs='*', help='Command arguments')

    args = parser.parse_args()

    admin = DEFCON1Admin()

    try:
        if args.command == 'init-session':
            if not args.args:
                print("Error: user-id required")
                print("Usage: defcon1_admin.py init-session <user-id>")
                sys.exit(1)
            admin.init_session(args.args[0])

        elif args.command == 'list-sessions':
            admin.list_sessions()

        elif args.command == 'session-status':
            if not args.args:
                print("Error: session-id required")
                print("Usage: defcon1_admin.py session-status <session-id>")
                sys.exit(1)
            admin.session_status(args.args[0])

        elif args.command == 'terminate-session':
            if not args.args:
                print("Error: session-id required")
                print("Usage: defcon1_admin.py terminate-session <session-id>")
                sys.exit(1)
            reason = args.args[1] if len(args.args) > 1 else "Manual termination"
            admin.terminate_session(args.args[0], reason)

        elif args.command == 'test-dual-auth':
            if not args.args:
                print("Error: user-id required")
                print("Usage: defcon1_admin.py test-dual-auth <user-id>")
                sys.exit(1)
            admin.test_dual_auth(args.args[0])

        elif args.command == 'demo':
            admin.demo_workflow()

        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
