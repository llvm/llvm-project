#!/usr/bin/env python3
"""
DSMIL Yubikey Administration Tool
==================================

Command-line tool for managing Yubikey devices and credentials.

Commands:
- register    - Register a new Yubikey
- list        - List all registered devices
- revoke      - Revoke a device
- suspend     - Suspend a device
- reactivate  - Reactivate a device
- test        - Test Yubikey authentication
- info        - Show device information

Author: DSMIL Platform
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import argparse
from datetime import datetime
from typing import Optional

try:
    from yubikey_auth import YubikeyAuth, YubikeyDevice, DeviceStatus, AuthMethod
except ImportError:
    print("Error: yubikey_auth module not found")
    sys.exit(1)


def format_timestamp(ts: Optional[str]) -> str:
    """Format ISO timestamp for display"""
    if not ts:
        return "Never"
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except:
        return ts


def cmd_list(auth: YubikeyAuth, args):
    """List all registered devices"""
    status_filter = None
    if args.status:
        try:
            status_filter = DeviceStatus(args.status)
        except ValueError:
            print(f"Error: Invalid status: {args.status}")
            print("Valid statuses: active, suspended, revoked")
            return 1

    devices = auth.list_devices(status=status_filter)

    if not devices:
        print("No devices registered.")
        return 0

    print(f"\nRegistered Yubikey Devices ({len(devices)}):")
    print("=" * 80)

    for device in devices:
        print(f"\nDevice ID:      {device.device_id}")
        print(f"Name:           {device.device_name}")
        print(f"Status:         {device.status.upper()}")
        print(f"Serial:         {device.serial_number or 'N/A'}")
        print(f"Firmware:       {device.firmware_version or 'N/A'}")
        print(f"Auth Methods:   {', '.join(device.auth_methods)}")
        print(f"Credentials:    {len(device.credentials)}")
        print(f"Created:        {format_timestamp(device.created_at)}")
        print(f"Last Used:      {format_timestamp(device.last_used)}")

        if args.verbose and device.credentials:
            print(f"\n  Credentials:")
            for i, cred in enumerate(device.credentials, 1):
                print(f"    {i}. {cred.device_name}")
                print(f"       Status:     {cred.status}")
                print(f"       Sign Count: {cred.sign_count}")
                print(f"       Created:    {format_timestamp(cred.created_at)}")
                print(f"       Last Used:  {format_timestamp(cred.last_used)}")

    print("\n" + "=" * 80)

    # Summary
    counts = auth.get_device_count()
    print(f"\nSummary: {counts['total']} total, {counts['active']} active, " +
          f"{counts['suspended']} suspended, {counts['revoked']} revoked")

    return 0


def cmd_info(auth: YubikeyAuth, args):
    """Show device information"""
    device = auth.get_device(args.device_id)

    if not device:
        print(f"Error: Device not found: {args.device_id}")
        return 1

    print(f"\nYubikey Device Information:")
    print("=" * 80)
    print(f"Device ID:      {device.device_id}")
    print(f"Name:           {device.device_name}")
    print(f"Status:         {device.status.upper()}")
    print(f"Serial Number:  {device.serial_number or 'N/A'}")
    print(f"Firmware:       {device.firmware_version or 'N/A'}")
    print(f"Created:        {format_timestamp(device.created_at)}")
    print(f"Last Used:      {format_timestamp(device.last_used)}")

    print(f"\nAuthentication Methods:")
    for method in device.auth_methods:
        print(f"  - {method.upper()}")

    if device.challenge_response_slot:
        print(f"\nChallenge-Response:")
        print(f"  Slot: {device.challenge_response_slot}")

    if device.otp_public_id:
        print(f"\nYubico OTP:")
        print(f"  Public ID: {device.otp_public_id}")

    print(f"\nFIDO2 Credentials ({len(device.credentials)}):")
    if device.credentials:
        for i, cred in enumerate(device.credentials, 1):
            print(f"\n  Credential {i}:")
            print(f"    Name:       {cred.device_name}")
            print(f"    Status:     {cred.status}")
            print(f"    ID:         {cred.credential_id[:32]}...")
            print(f"    AAGUID:     {cred.aaguid}")
            print(f"    Sign Count: {cred.sign_count}")
            print(f"    Created:    {format_timestamp(cred.created_at)}")
            print(f"    Last Used:  {format_timestamp(cred.last_used)}")
    else:
        print("  No credentials registered")

    print("\n" + "=" * 80)

    return 0


def cmd_register(auth: YubikeyAuth, args):
    """Register a new Yubikey"""
    print("\nYubikey Registration")
    print("=" * 80)

    # Get device name
    device_name = args.name or input("Device name (e.g., 'Yubikey 5 NFC'): ").strip()
    if not device_name:
        print("Error: Device name required")
        return 1

    # Get username
    username = args.user or input("Username: ").strip()
    if not username:
        print("Error: Username required")
        return 1

    print(f"\nRegistering: {device_name}")
    print(f"User:        {username}")
    print(f"\nMethod:      FIDO2/WebAuthn")

    print("\nNote: FIDO2 registration requires web browser interaction.")
    print("      Use the tactical UI to complete registration.")
    print("      Or use challenge-response for offline authentication.")

    if not args.yes:
        response = input("\nContinue with FIDO2 registration? [y/N]: ").strip().lower()
        if response != 'y':
            print("Registration cancelled.")
            return 0

    try:
        # Begin registration
        print("\nStarting FIDO2 registration...")
        result = auth.begin_fido2_registration(username, username)

        print("\n✓ Registration initiated")
        print(f"\nNext steps:")
        print(f"  1. Open tactical UI in browser")
        print(f"  2. Navigate to Settings > Yubikey Registration")
        print(f"  3. Follow the on-screen instructions")
        print(f"  4. Touch your Yubikey when prompted")

        print(f"\nRegistration data saved for user: {username}")

        return 0

    except Exception as e:
        print(f"\n✗ Registration failed: {e}")
        return 1


def cmd_revoke(auth: YubikeyAuth, args):
    """Revoke a device"""
    device = auth.get_device(args.device_id)

    if not device:
        print(f"Error: Device not found: {args.device_id}")
        return 1

    print(f"\nRevoke Yubikey Device")
    print("=" * 80)
    print(f"Device ID:   {device.device_id}")
    print(f"Name:        {device.device_name}")
    print(f"Status:      {device.status.upper()}")

    if device.status == DeviceStatus.REVOKED.value:
        print("\nDevice is already revoked.")
        return 0

    reason = args.reason or "Administrative action"

    if not args.yes:
        print(f"\n⚠️  WARNING: Revocation is permanent!")
        print(f"Reason: {reason}")
        response = input("\nAre you sure? [y/N]: ").strip().lower()
        if response != 'y':
            print("Revocation cancelled.")
            return 0

    try:
        success = auth.revoke_device(args.device_id, reason)

        if success:
            print(f"\n✓ Device revoked: {device.device_name}")
            return 0
        else:
            print(f"\n✗ Failed to revoke device")
            return 1

    except Exception as e:
        print(f"\n✗ Revocation failed: {e}")
        return 1


def cmd_suspend(auth: YubikeyAuth, args):
    """Suspend a device"""
    device = auth.get_device(args.device_id)

    if not device:
        print(f"Error: Device not found: {args.device_id}")
        return 1

    print(f"\nSuspend Yubikey Device")
    print("=" * 80)
    print(f"Device ID:   {device.device_id}")
    print(f"Name:        {device.device_name}")
    print(f"Status:      {device.status.upper()}")

    if device.status == DeviceStatus.SUSPENDED.value:
        print("\nDevice is already suspended.")
        return 0

    if device.status == DeviceStatus.REVOKED.value:
        print("\nError: Cannot suspend revoked device.")
        return 1

    reason = args.reason or "Administrative action"

    if not args.yes:
        print(f"\nReason: {reason}")
        response = input("\nAre you sure? [y/N]: ").strip().lower()
        if response != 'y':
            print("Suspension cancelled.")
            return 0

    try:
        success = auth.suspend_device(args.device_id, reason)

        if success:
            print(f"\n✓ Device suspended: {device.device_name}")
            print(f"\nTo reactivate: yubikey_admin.py reactivate {args.device_id}")
            return 0
        else:
            print(f"\n✗ Failed to suspend device")
            return 1

    except Exception as e:
        print(f"\n✗ Suspension failed: {e}")
        return 1


def cmd_reactivate(auth: YubikeyAuth, args):
    """Reactivate a suspended device"""
    device = auth.get_device(args.device_id)

    if not device:
        print(f"Error: Device not found: {args.device_id}")
        return 1

    print(f"\nReactivate Yubikey Device")
    print("=" * 80)
    print(f"Device ID:   {device.device_id}")
    print(f"Name:        {device.device_name}")
    print(f"Status:      {device.status.upper()}")

    if device.status == DeviceStatus.ACTIVE.value:
        print("\nDevice is already active.")
        return 0

    if device.status == DeviceStatus.REVOKED.value:
        print("\nError: Cannot reactivate revoked device.")
        return 1

    if not args.yes:
        response = input("\nReactivate this device? [y/N]: ").strip().lower()
        if response != 'y':
            print("Reactivation cancelled.")
            return 0

    try:
        success = auth.reactivate_device(args.device_id)

        if success:
            print(f"\n✓ Device reactivated: {device.device_name}")
            return 0
        else:
            print(f"\n✗ Failed to reactivate device")
            return 1

    except Exception as e:
        print(f"\n✗ Reactivation failed: {e}")
        return 1


def cmd_test(auth: YubikeyAuth, args):
    """Test Yubikey authentication"""
    print("\nYubikey Authentication Test")
    print("=" * 80)

    if args.device_id:
        device = auth.get_device(args.device_id)
        if not device:
            print(f"Error: Device not found: {args.device_id}")
            return 1

        devices = [device]
    else:
        devices = auth.list_devices(status=DeviceStatus.ACTIVE)

    if not devices:
        print("No active devices to test.")
        return 1

    print(f"\nTesting {len(devices)} device(s)...\n")

    for device in devices:
        print(f"Device: {device.device_name} ({device.device_id})")

        # Test methods
        tested = False

        if AuthMethod.CHALLENGE_RESPONSE.value in device.auth_methods:
            print("  Testing Challenge-Response...")
            try:
                result = auth.authenticate_challenge_response(device.device_id)
                if result:
                    print("  ✓ Challenge-Response authentication successful")
                    tested = True
                else:
                    print("  ✗ Challenge-Response authentication failed")
            except Exception as e:
                print(f"  ✗ Challenge-Response error: {e}")

        if AuthMethod.FIDO2.value in device.auth_methods and not tested:
            print("  FIDO2/WebAuthn requires browser interaction")
            print("  Use tactical UI to test FIDO2 authentication")

        if not tested:
            print("  No testable methods configured")

        print()

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DSMIL Yubikey Administration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all devices
  yubikey_admin.py list

  # List only active devices
  yubikey_admin.py list --status active

  # Register new Yubikey
  yubikey_admin.py register --name "Yubikey 5 NFC" --user admin

  # Show device information
  yubikey_admin.py info <device-id>

  # Revoke a device
  yubikey_admin.py revoke <device-id> --reason "Device lost"

  # Suspend a device
  yubikey_admin.py suspend <device-id> --reason "Under investigation"

  # Reactivate a suspended device
  yubikey_admin.py reactivate <device-id>

  # Test authentication
  yubikey_admin.py test
  yubikey_admin.py test --device-id <device-id>
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    parser_list = subparsers.add_parser('list', help='List registered devices')
    parser_list.add_argument('--status', choices=['active', 'suspended', 'revoked'],
                           help='Filter by status')
    parser_list.add_argument('-v', '--verbose', action='store_true',
                           help='Show detailed information')

    # Info command
    parser_info = subparsers.add_parser('info', help='Show device information')
    parser_info.add_argument('device_id', help='Device ID')

    # Register command
    parser_register = subparsers.add_parser('register', help='Register new Yubikey')
    parser_register.add_argument('--name', help='Device name')
    parser_register.add_argument('--user', help='Username')
    parser_register.add_argument('-y', '--yes', action='store_true',
                               help='Skip confirmation prompts')

    # Revoke command
    parser_revoke = subparsers.add_parser('revoke', help='Revoke a device')
    parser_revoke.add_argument('device_id', help='Device ID')
    parser_revoke.add_argument('--reason', help='Reason for revocation')
    parser_revoke.add_argument('-y', '--yes', action='store_true',
                             help='Skip confirmation prompts')

    # Suspend command
    parser_suspend = subparsers.add_parser('suspend', help='Suspend a device')
    parser_suspend.add_argument('device_id', help='Device ID')
    parser_suspend.add_argument('--reason', help='Reason for suspension')
    parser_suspend.add_argument('-y', '--yes', action='store_true',
                              help='Skip confirmation prompts')

    # Reactivate command
    parser_reactivate = subparsers.add_parser('reactivate', help='Reactivate a device')
    parser_reactivate.add_argument('device_id', help='Device ID')
    parser_reactivate.add_argument('-y', '--yes', action='store_true',
                                 help='Skip confirmation prompts')

    # Test command
    parser_test = subparsers.add_parser('test', help='Test authentication')
    parser_test.add_argument('--device-id', help='Test specific device (optional)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize auth
    try:
        auth = YubikeyAuth()
    except Exception as e:
        print(f"Error: Failed to initialize YubikeyAuth: {e}")
        return 1

    # Execute command
    commands = {
        'list': cmd_list,
        'info': cmd_info,
        'register': cmd_register,
        'revoke': cmd_revoke,
        'suspend': cmd_suspend,
        'reactivate': cmd_reactivate,
        'test': cmd_test
    }

    try:
        return commands[args.command](auth, args)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
