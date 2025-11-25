#!/usr/bin/env python3
"""
DEFCON1 YubiKey Registration Guide
===================================

Interactive guide for registering YubiKeys and setting up authorizers
for DEFCON1 authentication.

Classification: TOP SECRET // FOR OFFICIAL USE ONLY

Usage:
    ./defcon1_registration_guide.py

Author: DSMIL Platform
Date: 2025-11-25
"""

import sys
import os
from pathlib import Path
from yubikey_auth import YubikeyAuth, DeviceStatus


class DEFCON1RegistrationGuide:
    """Interactive registration guide for DEFCON1 setup"""

    def __init__(self):
        self.yubikey_auth = YubikeyAuth()

    def print_header(self, title):
        """Print section header"""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80 + "\n")

    def print_step(self, step_num, title):
        """Print step header"""
        print(f"\n{'─' * 80}")
        print(f"STEP {step_num}: {title}")
        print(f"{'─' * 80}\n")

    def check_prerequisites(self):
        """Check system prerequisites"""
        self.print_header("DEFCON1 REGISTRATION GUIDE")

        print("Classification: TOP SECRET // FOR OFFICIAL USE ONLY")
        print("Threat Level: DEFCON 1 (Maximum Readiness)")
        print()

        print("This guide will help you:")
        print("  1. Register TWO YubiKeys for the primary user (dual authentication)")
        print("  2. Register YubiKeys for FOUR authorizers (approval chain)")
        print("  3. Configure authorization levels")
        print("  4. Test the complete DEFCON1 authentication flow")
        print()

        input("Press ENTER to begin setup...")

    def explain_architecture(self):
        """Explain the DEFCON1 authentication architecture"""
        self.print_header("DEFCON1 AUTHENTICATION ARCHITECTURE")

        print("DEFCON1 requires MULTIPLE YubiKeys from MULTIPLE people:")
        print()

        print("┌─────────────────────────────────────────────────────────────┐")
        print("│ PRIMARY USER (Person accessing the system)                 │")
        print("├─────────────────────────────────────────────────────────────┤")
        print("│ • Primary YubiKey   → FIDO2 authentication                 │")
        print("│ • Secondary YubiKey → FIDO2 authentication (different key) │")
        print("└─────────────────────────────────────────────────────────────┘")
        print()

        print("┌─────────────────────────────────────────────────────────────┐")
        print("│ AUTHORIZERS (4 separate people approving access)           │")
        print("├─────────────────────────────────────────────────────────────┤")
        print("│ Authorizer #1 (Standard/Supervisor)                        │")
        print("│   • Their personal YubiKey → FIDO2 authentication          │")
        print("│                                                             │")
        print("│ Authorizer #2 (Supervisor/Commander)                       │")
        print("│   • Their personal YubiKey → FIDO2 authentication          │")
        print("│                                                             │")
        print("│ Authorizer #3 (Commander)                                  │")
        print("│   • Their personal YubiKey → FIDO2 authentication          │")
        print("│                                                             │")
        print("│ Authorizer #4 (EXECUTIVE - REQUIRED)                       │")
        print("│   • Their personal YubiKey → FIDO2 authentication          │")
        print("│   • Executive-level authorization (Presidential/SECDEF)    │")
        print("└─────────────────────────────────────────────────────────────┘")
        print()

        print("TOTAL YUBIKEYS NEEDED: 6 minimum")
        print("  • 2 for primary user (dual auth)")
        print("  • 4 for authorizers (one per person)")
        print()

        input("Press ENTER to continue...")

    def register_primary_user_yubikeys(self):
        """Guide for registering primary user's two YubiKeys"""
        self.print_step(1, "Register Primary User's YubiKeys (2 required)")

        print("The primary user needs TWO different YubiKeys:")
        print("  1. Primary YubiKey (everyday use)")
        print("  2. Secondary YubiKey (backup/redundancy)")
        print()

        print("⚠️  IMPORTANT: These must be PHYSICALLY DIFFERENT YubiKeys!")
        print("   You cannot use the same YubiKey twice.")
        print()

        user_id = input("Enter primary user ID (e.g., 'tactical_user'): ").strip()
        if not user_id:
            print("❌ User ID required")
            return False

        print()
        print("=" * 80)
        print("REGISTERING PRIMARY YUBIKEY")
        print("=" * 80)
        print()
        print("Steps:")
        print("  1. Insert your PRIMARY YubiKey (the one you'll use most often)")
        print("  2. Open browser to: http://localhost:5001/tactical_yubikey_ui.html")
        print("  3. Click 'REGISTER NEW KEY'")
        print("  4. Enter name: 'Primary YubiKey'")
        print("  5. Touch YubiKey when prompted")
        print()

        input("Press ENTER after registering PRIMARY YubiKey...")

        # Check if primary YubiKey registered
        devices = self.yubikey_auth.list_devices()
        if len(devices) < 1:
            print("❌ No YubiKeys detected. Please register via web interface:")
            print("   python3 yubikey_admin.py register --name 'Primary YubiKey' --user", user_id)
            return False

        print("✅ Primary YubiKey registration detected")
        print()

        print("=" * 80)
        print("REGISTERING SECONDARY YUBIKEY")
        print("=" * 80)
        print()
        print("Steps:")
        print("  1. REMOVE your primary YubiKey")
        print("  2. Insert your SECONDARY YubiKey (different physical device)")
        print("  3. In browser: Click 'REGISTER NEW KEY' again")
        print("  4. Enter name: 'Secondary YubiKey'")
        print("  5. Touch YubiKey when prompted")
        print()

        input("Press ENTER after registering SECONDARY YubiKey...")

        # Check if both YubiKeys registered
        devices = self.yubikey_auth.list_devices()
        if len(devices) < 2:
            print("❌ Only 1 YubiKey detected. Need 2 for DEFCON1.")
            print("   Please register second YubiKey via web interface")
            return False

        print("✅ Secondary YubiKey registration detected")
        print()
        print(f"✅ Primary user '{user_id}' has 2 YubiKeys registered!")
        print()

        # Show registered YubiKeys
        print("Registered YubiKeys for", user_id + ":")
        for i, device in enumerate(devices[:2], 1):
            print(f"  [{i}] {device.device_name}")
            print(f"      ID: {device.device_id}")
            print(f"      Status: {device.status}")
            print(f"      Serial: {device.serial_number or 'N/A'}")
            print()

        return True

    def register_authorizers(self):
        """Guide for registering authorizer YubiKeys"""
        self.print_step(2, "Register Authorizers' YubiKeys (4 required)")

        print("Each authorizer is a SEPARATE PERSON with their OWN YubiKey.")
        print()
        print("Required Authorizers:")
        print("  • Authorizer #1: Standard Operator or Supervisor")
        print("  • Authorizer #2: Supervisor or Commander")
        print("  • Authorizer #3: Commander")
        print("  • Authorizer #4: EXECUTIVE (Presidential/SECDEF level) ⚠️ REQUIRED")
        print()

        authorizers = []

        for i in range(1, 5):
            print(f"\n{'─' * 40}")
            print(f"AUTHORIZER #{i}")
            print(f"{'─' * 40}\n")

            if i == 4:
                print("⚠️  EXECUTIVE-LEVEL AUTHORIZER (REQUIRED)")
                print("   This must be Presidential/SECDEF/Executive level")
                print()

            user_id = input(f"  User ID for Authorizer #{i}: ").strip()
            if not user_id:
                print("  ⚠️  Skipping (user ID required)")
                continue

            name = input(f"  Full name: ").strip()
            role = input(f"  Role (e.g., 'Operator', 'Commander', 'Executive'): ").strip()

            if i == 4:
                level = "EXECUTIVE"
            elif i == 3:
                level = "COMMANDER"
            elif i == 2:
                level = "SUPERVISOR"
            else:
                level = "STANDARD"

            print()
            print(f"  Authorization Level: {level}")
            print()
            print("  Steps to register:")
            print(f"  1. {name} inserts their personal YubiKey")
            print("  2. Open browser: http://localhost:5001/tactical_yubikey_ui.html")
            print("  3. Click 'REGISTER NEW KEY'")
            print(f"  4. Enter name: '{name} - {role}'")
            print("  5. Touch YubiKey when prompted")
            print()

            input(f"  Press ENTER after {name} registers their YubiKey...")

            authorizers.append({
                'number': i,
                'user_id': user_id,
                'name': name,
                'role': role,
                'level': level
            })

            print(f"  ✅ Authorizer #{i} recorded: {name} ({role})")

        print()
        print("=" * 80)
        print("AUTHORIZER SUMMARY")
        print("=" * 80)
        print()

        if len(authorizers) < 4:
            print(f"⚠️  Warning: Only {len(authorizers)} authorizers configured")
            print("   DEFCON1 requires 4 authorizers including 1 executive")
            print()

        for auth in authorizers:
            print(f"Authorizer #{auth['number']}: {auth['name']}")
            print(f"  User ID: {auth['user_id']}")
            print(f"  Role: {auth['role']}")
            print(f"  Level: {auth['level']}")
            print()

        # Check for executive
        has_executive = any(a['level'] == 'EXECUTIVE' for a in authorizers)
        if not has_executive:
            print("❌ No EXECUTIVE-level authorizer configured!")
            print("   DEFCON1 requires at least 1 executive authorizer")
            print()
        else:
            print("✅ Executive authorizer configured")

        return len(authorizers) >= 4 and has_executive

    def show_authentication_workflow(self):
        """Show the complete authentication workflow"""
        self.print_step(3, "DEFCON1 Authentication Workflow")

        print("When a DEFCON1 session is initiated, here's what happens:")
        print()

        print("PHASE 1: PRIMARY USER DUAL YUBIKEY AUTHENTICATION")
        print("  1. User initiates DEFCON1 session")
        print("     → System generates session ID")
        print()
        print("  2. User inserts PRIMARY YubiKey")
        print("     → Browser shows WebAuthn prompt")
        print("     → User touches YubiKey sensor")
        print("     → FIDO2 challenge-response validates")
        print("     → ✅ Primary YubiKey authenticated")
        print()
        print("  3. User REMOVES primary, inserts SECONDARY YubiKey")
        print("     → Browser shows NEW WebAuthn prompt (different challenge)")
        print("     → User touches YubiKey sensor")
        print("     → FIDO2 challenge-response validates")
        print("     → ✅ Secondary YubiKey authenticated")
        print()

        print("PHASE 2: AUTHORIZER CHAIN (4 people)")
        print("  4. Authorizer #1 arrives")
        print("     → Inserts their personal YubiKey")
        print("     → WebAuthn authentication")
        print("     → Digital signature recorded")
        print("     → ✅ Authorizer #1 approved")
        print()
        print("  5. Authorizer #2 arrives")
        print("     → Inserts their personal YubiKey")
        print("     → WebAuthn authentication")
        print("     → Digital signature recorded")
        print("     → ✅ Authorizer #2 approved")
        print()
        print("  6. Authorizer #3 arrives")
        print("     → Inserts their personal YubiKey")
        print("     → WebAuthn authentication")
        print("     → Digital signature recorded")
        print("     → ✅ Authorizer #3 approved")
        print()
        print("  7. Authorizer #4 (EXECUTIVE) arrives")
        print("     → Inserts their personal YubiKey")
        print("     → WebAuthn authentication")
        print("     → Digital signature recorded")
        print("     → ✅ EXECUTIVE authorization granted")
        print()

        print("PHASE 3: SESSION ACTIVATION")
        print("  8. System validates all requirements:")
        print("     ✓ Primary user authenticated with 2 YubiKeys")
        print("     ✓ 4 authorizers authenticated")
        print("     ✓ 1 executive authorizer confirmed")
        print()
        print("  9. DEFCON1 session ACTIVATED")
        print("     → Duration: 1 hour")
        print("     → Access: EMERGENCY ONLY")
        print("     → Continuous auth: Every 5 minutes")
        print()

        print("PHASE 4: CONTINUOUS MONITORING")
        print("  10. Every 5 minutes:")
        print("      → User re-authenticates with BOTH YubiKeys")
        print("      → If successful: Session continues")
        print("      → If failed: Session terminates immediately")
        print()
        print("  11. After 1 hour:")
        print("      → Session expires automatically")
        print("      → Full re-authentication required for new session")
        print()

    def show_quick_commands(self):
        """Show quick command reference"""
        self.print_header("QUICK COMMAND REFERENCE")

        print("Register YubiKeys via Web Interface:")
        print("  1. Start tactical UI:")
        print("     cd /home/user/DSLLVM/lat5150drvmil/03-web-interface")
        print("     firefox tactical_yubikey_ui.html")
        print()
        print("  2. Click 'REGISTER NEW KEY' for each YubiKey")
        print("     - Primary user: Register 2 YubiKeys")
        print("     - Each authorizer: Register 1 YubiKey")
        print()

        print("Register YubiKeys via Command Line:")
        print("  # Primary user - Primary YubiKey")
        print("  python3 yubikey_admin.py register --name 'Primary YubiKey' --user tactical_user")
        print()
        print("  # Primary user - Secondary YubiKey")
        print("  python3 yubikey_admin.py register --name 'Secondary YubiKey' --user tactical_user")
        print()
        print("  # Authorizer YubiKeys")
        print("  python3 yubikey_admin.py register --name 'John Doe - Commander' --user johndoe")
        print("  python3 yubikey_admin.py register --name 'Jane Smith - Executive' --user janesmith")
        print()

        print("List Registered YubiKeys:")
        print("  python3 yubikey_admin.py list")
        print()

        print("Test Dual YubiKey Setup:")
        print("  python3 defcon1_admin.py test-dual-auth tactical_user")
        print()

        print("Initialize DEFCON1 Session:")
        print("  python3 defcon1_admin.py init-session tactical_user")
        print()

        print("View Session Status:")
        print("  python3 defcon1_admin.py list-sessions")
        print("  python3 defcon1_admin.py session-status <session-id>")
        print()

    def run(self):
        """Run the complete registration guide"""
        try:
            # Step 0: Prerequisites
            self.check_prerequisites()

            # Explain architecture
            self.explain_architecture()

            # Step 1: Register primary user's 2 YubiKeys
            if not self.register_primary_user_yubikeys():
                print("\n⚠️  Primary user YubiKey registration incomplete")
                print("   Please complete registration before proceeding")
                return

            # Step 2: Register authorizers' YubiKeys
            if not self.register_authorizers():
                print("\n⚠️  Authorizer registration incomplete")
                print("   DEFCON1 requires 4 authorizers including 1 executive")
                print("   Continue registration before activating DEFCON1 sessions")

            # Step 3: Show workflow
            self.show_authentication_workflow()

            # Show quick commands
            self.show_quick_commands()

            # Final summary
            self.print_header("REGISTRATION COMPLETE")

            print("✅ DEFCON1 registration guide completed!")
            print()
            print("Next Steps:")
            print("  1. Verify all YubiKeys registered:")
            print("     python3 yubikey_admin.py list")
            print()
            print("  2. Test dual authentication:")
            print("     python3 defcon1_admin.py test-dual-auth tactical_user")
            print()
            print("  3. Initialize DEFCON1 session when ready:")
            print("     python3 defcon1_admin.py init-session tactical_user")
            print()
            print("  4. View complete workflow demo:")
            print("     python3 defcon1_admin.py demo")
            print()

            print("=" * 80)
            print()

        except KeyboardInterrupt:
            print("\n\n⚠️  Registration cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    guide = DEFCON1RegistrationGuide()
    guide.run()


if __name__ == "__main__":
    main()
