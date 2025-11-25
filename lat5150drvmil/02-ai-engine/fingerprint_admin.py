#!/usr/bin/env python3
"""
DSMIL Fingerprint Administration Tool
======================================

Command-line tool for managing fingerprint enrollments.

Commands:
- enroll      - Enroll a new fingerprint
- list        - List all enrolled fingerprints
- info        - Show enrollment information
- delete      - Delete an enrollment
- suspend     - Suspend an enrollment
- reactivate  - Reactivate an enrollment
- verify      - Test fingerprint verification
- devices     - List fingerprint devices

Author: DSMIL Platform
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import argparse
from datetime import datetime
from typing import Optional

try:
    from fingerprint_auth import FingerprintAuth, FingerType, EnrollmentStatus
except ImportError:
    print("Error: fingerprint_auth module not found")
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


def cmd_devices(auth: FingerprintAuth, args):
    """List fingerprint devices"""
    devices = auth.list_devices()

    if not devices:
        print("\n⚠️  No fingerprint devices found")
        print("\nTroubleshooting:")
        print("  1. Check if fprintd is installed: dpkg -l | grep fprintd")
        print("  2. Check if service is running: systemctl status fprintd")
        print("  3. Check device detection: lsusb | grep -i fingerprint")
        print("  4. Check kernel modules: lsmod | grep -i fprint")
        return 1

    print(f"\nFingerprint Devices ({len(devices)}):")
    print("=" * 80)

    for device in devices:
        print(f"\nDevice:         {device.device_name}")
        print(f"Path:           {device.device_path}")
        print(f"Driver:         {device.driver_name}")
        print(f"Scan Type:      {device.scan_type.upper()}")
        print(f"Available:      {'YES' if device.is_available else 'NO'}")

        if device.claimed_by:
            print(f"Claimed By:     {device.claimed_by}")

    print("\n" + "=" * 80)

    return 0


def cmd_list(auth: FingerprintAuth, args):
    """List all enrolled fingerprints"""
    status_filter = None
    if args.status:
        try:
            status_filter = EnrollmentStatus(args.status)
        except ValueError:
            print(f"Error: Invalid status: {args.status}")
            print("Valid statuses: active, suspended, revoked")
            return 1

    username_filter = args.user if args.user else None

    enrollments = auth.list_enrollments(username=username_filter, status=status_filter)

    if not enrollments:
        print("No fingerprints enrolled.")
        return 0

    print(f"\nEnrolled Fingerprints ({len(enrollments)}):")
    print("=" * 80)

    # Group by user
    by_user = {}
    for fp in enrollments:
        if fp.username not in by_user:
            by_user[fp.username] = []
        by_user[fp.username].append(fp)

    for username, user_fps in sorted(by_user.items()):
        print(f"\nUser: {username}")
        print(f"  Enrolled fingers: {len(user_fps)}")

        for fp in user_fps:
            print(f"\n  Finger ID:      {fp.finger_id}")
            print(f"  Finger:         {fp.finger_type}")
            print(f"  Status:         {fp.status.upper()}")
            print(f"  Quality:        {fp.quality_score if fp.quality_score else 'N/A'}/100")
            print(f"  Enrolled:       {format_timestamp(fp.enrolled_at)}")
            print(f"  Last Used:      {format_timestamp(fp.last_used)}")
            print(f"  Use Count:      {fp.use_count}")

    print("\n" + "=" * 80)

    # Summary
    counts = auth.get_enrollment_count()
    print(f"\nSummary: {counts['total']} total, {counts['active']} active, " +
          f"{counts['suspended']} suspended, {counts['revoked']} revoked")

    return 0


def cmd_info(auth: FingerprintAuth, args):
    """Show enrollment information"""
    fp = auth.get_enrollment(args.finger_id)

    if not fp:
        print(f"Error: Enrollment not found: {args.finger_id}")
        return 1

    print(f"\nFingerprint Enrollment Information:")
    print("=" * 80)
    print(f"Finger ID:      {fp.finger_id}")
    print(f"Username:       {fp.username}")
    print(f"Finger Type:    {fp.finger_type}")
    print(f"Status:         {fp.status.upper()}")
    print(f"Quality Score:  {fp.quality_score if fp.quality_score else 'N/A'}/100")
    print(f"Enrolled:       {format_timestamp(fp.enrolled_at)}")
    print(f"Last Used:      {format_timestamp(fp.last_used)}")
    print(f"Use Count:      {fp.use_count}")
    print("=" * 80)

    return 0


def cmd_enroll(auth: FingerprintAuth, args):
    """Enroll a new fingerprint"""
    print("\nFingerprint Enrollment")
    print("=" * 80)

    # Get username
    username = args.user or input("Username: ").strip()
    if not username:
        print("Error: Username required")
        return 1

    # Get finger type
    if args.finger:
        try:
            finger_type = FingerType(args.finger)
        except ValueError:
            print(f"Error: Invalid finger type: {args.finger}")
            print("\nAvailable fingers:")
            for ft in FingerType:
                print(f"  - {ft.value}")
            return 1
    else:
        print("\nAvailable fingers:")
        fingers = list(FingerType)
        for i, ft in enumerate(fingers, 1):
            print(f"  {i}. {ft.value}")

        choice = input("\nSelect finger (1-10): ").strip()
        try:
            finger_type = fingers[int(choice) - 1]
        except (ValueError, IndexError):
            print("Error: Invalid selection")
            return 1

    # Check if already enrolled
    existing = auth.list_enrollments(username=username)
    for fp in existing:
        if fp.finger_type == finger_type.value and fp.status == EnrollmentStatus.ACTIVE.value:
            print(f"\n⚠️  Warning: {finger_type.value} already enrolled for {username}")
            if not args.yes:
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Enrollment cancelled.")
                    return 0

    print(f"\nEnrolling: {username} - {finger_type.value}")
    print("\nNote: Fingerprint enrollment requires multiple scans.")
    print("      Follow the on-screen prompts and scan your finger when requested.")

    if not args.yes:
        response = input("\nBegin enrollment? [y/N]: ").strip().lower()
        if response != 'y':
            print("Enrollment cancelled.")
            return 0

    try:
        # Start enrollment
        print("\nStarting enrollment...")
        success, message = auth.enroll_finger(username, finger_type)

        if not success:
            print(f"\n✗ Enrollment failed: {message}")
            return 1

        print(f"\n✓ {message}")
        print("\nIMPORTANT: Enrollment requires interactive sensor access.")
        print("           Use the tactical UI or fprintd-enroll command to complete enrollment.")
        print(f"\nCommand: fprintd-enroll -f {finger_type.value} {username}")

        # For now, we'll create a placeholder enrollment
        # Real enrollment would require async D-Bus callbacks
        if args.test_mode:
            print("\n[TEST MODE] Creating test enrollment...")
            auth.complete_enrollment(username, finger_type, quality=85)
            print("✓ Test enrollment created")

        return 0

    except Exception as e:
        print(f"\n✗ Enrollment error: {e}")
        return 1


def cmd_delete(auth: FingerprintAuth, args):
    """Delete an enrollment"""
    fp = auth.get_enrollment(args.finger_id)

    if not fp:
        print(f"Error: Enrollment not found: {args.finger_id}")
        return 1

    print(f"\nDelete Fingerprint Enrollment")
    print("=" * 80)
    print(f"Finger ID:   {fp.finger_id}")
    print(f"Username:    {fp.username}")
    print(f"Finger:      {fp.finger_type}")
    print(f"Status:      {fp.status.upper()}")

    reason = args.reason or "Administrative action"

    if not args.yes:
        print(f"\n⚠️  WARNING: Deletion is permanent!")
        print(f"Reason: {reason}")
        response = input("\nAre you sure? [y/N]: ").strip().lower()
        if response != 'y':
            print("Deletion cancelled.")
            return 0

    try:
        success = auth.delete_enrollment(args.finger_id, reason)

        if success:
            print(f"\n✓ Enrollment deleted: {fp.finger_type}")
            return 0
        else:
            print(f"\n✗ Failed to delete enrollment")
            return 1

    except Exception as e:
        print(f"\n✗ Deletion failed: {e}")
        return 1


def cmd_suspend(auth: FingerprintAuth, args):
    """Suspend an enrollment"""
    fp = auth.get_enrollment(args.finger_id)

    if not fp:
        print(f"Error: Enrollment not found: {args.finger_id}")
        return 1

    print(f"\nSuspend Fingerprint Enrollment")
    print("=" * 80)
    print(f"Finger ID:   {fp.finger_id}")
    print(f"Username:    {fp.username}")
    print(f"Finger:      {fp.finger_type}")
    print(f"Status:      {fp.status.upper()}")

    if fp.status == EnrollmentStatus.SUSPENDED.value:
        print("\nEnrollment is already suspended.")
        return 0

    reason = args.reason or "Administrative action"

    if not args.yes:
        print(f"\nReason: {reason}")
        response = input("\nAre you sure? [y/N]: ").strip().lower()
        if response != 'y':
            print("Suspension cancelled.")
            return 0

    try:
        success = auth.suspend_enrollment(args.finger_id, reason)

        if success:
            print(f"\n✓ Enrollment suspended: {fp.finger_type}")
            print(f"\nTo reactivate: fingerprint_admin.py reactivate {args.finger_id}")
            return 0
        else:
            print(f"\n✗ Failed to suspend enrollment")
            return 1

    except Exception as e:
        print(f"\n✗ Suspension failed: {e}")
        return 1


def cmd_reactivate(auth: FingerprintAuth, args):
    """Reactivate a suspended enrollment"""
    fp = auth.get_enrollment(args.finger_id)

    if not fp:
        print(f"Error: Enrollment not found: {args.finger_id}")
        return 1

    print(f"\nReactivate Fingerprint Enrollment")
    print("=" * 80)
    print(f"Finger ID:   {fp.finger_id}")
    print(f"Username:    {fp.username}")
    print(f"Finger:      {fp.finger_type}")
    print(f"Status:      {fp.status.upper()}")

    if fp.status == EnrollmentStatus.ACTIVE.value:
        print("\nEnrollment is already active.")
        return 0

    if fp.status == EnrollmentStatus.REVOKED.value:
        print("\nError: Cannot reactivate revoked enrollment.")
        return 1

    if not args.yes:
        response = input("\nReactivate this enrollment? [y/N]: ").strip().lower()
        if response != 'y':
            print("Reactivation cancelled.")
            return 0

    try:
        success = auth.reactivate_enrollment(args.finger_id)

        if success:
            print(f"\n✓ Enrollment reactivated: {fp.finger_type}")
            return 0
        else:
            print(f"\n✗ Failed to reactivate enrollment")
            return 1

    except Exception as e:
        print(f"\n✗ Reactivation failed: {e}")
        return 1


def cmd_verify(auth: FingerprintAuth, args):
    """Test fingerprint verification"""
    print("\nFingerprint Verification Test")
    print("=" * 80)

    username = args.user or input("Username: ").strip()
    if not username:
        print("Error: Username required")
        return 1

    # Check enrollments
    enrollments = auth.list_enrollments(username=username, status=EnrollmentStatus.ACTIVE)

    if not enrollments:
        print(f"No active fingerprints enrolled for {username}")
        return 1

    print(f"\nUser: {username}")
    print(f"Enrolled fingers: {len(enrollments)}")
    for fp in enrollments:
        print(f"  - {fp.finger_type} (quality: {fp.quality_score or 'N/A'}/100)")

    print("\nStarting verification...")
    print("Scan your finger on the reader...")

    try:
        success, message, matched = auth.verify_finger(username)

        if success:
            print(f"\n✓ {message}")
            print("\nNote: Complete verification requires interactive sensor access.")
            print("      Use the tactical UI or fprintd-verify command to complete verification.")
            print(f"\nCommand: fprintd-verify {username}")
        else:
            print(f"\n✗ Verification failed: {message}")
            return 1

        return 0

    except Exception as e:
        print(f"\n✗ Verification error: {e}")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DSMIL Fingerprint Administration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List fingerprint devices
  fingerprint_admin.py devices

  # List all enrollments
  fingerprint_admin.py list

  # List enrollments for specific user
  fingerprint_admin.py list --user admin

  # Enroll new fingerprint
  fingerprint_admin.py enroll --user admin --finger right-index-finger

  # Show enrollment information
  fingerprint_admin.py info <finger-id>

  # Delete an enrollment
  fingerprint_admin.py delete <finger-id> --reason "Device compromised"

  # Suspend an enrollment
  fingerprint_admin.py suspend <finger-id> --reason "Under investigation"

  # Reactivate a suspended enrollment
  fingerprint_admin.py reactivate <finger-id>

  # Test verification
  fingerprint_admin.py verify --user admin
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Devices command
    parser_devices = subparsers.add_parser('devices', help='List fingerprint devices')

    # List command
    parser_list = subparsers.add_parser('list', help='List enrolled fingerprints')
    parser_list.add_argument('--user', help='Filter by username')
    parser_list.add_argument('--status', choices=['active', 'suspended', 'revoked'],
                           help='Filter by status')

    # Info command
    parser_info = subparsers.add_parser('info', help='Show enrollment information')
    parser_info.add_argument('finger_id', help='Finger ID')

    # Enroll command
    parser_enroll = subparsers.add_parser('enroll', help='Enroll new fingerprint')
    parser_enroll.add_argument('--user', help='Username')
    parser_enroll.add_argument('--finger', help='Finger type (e.g., right-index-finger)')
    parser_enroll.add_argument('-y', '--yes', action='store_true',
                              help='Skip confirmation prompts')
    parser_enroll.add_argument('--test-mode', action='store_true',
                              help='Create test enrollment (for development)')

    # Delete command
    parser_delete = subparsers.add_parser('delete', help='Delete an enrollment')
    parser_delete.add_argument('finger_id', help='Finger ID')
    parser_delete.add_argument('--reason', help='Reason for deletion')
    parser_delete.add_argument('-y', '--yes', action='store_true',
                             help='Skip confirmation prompts')

    # Suspend command
    parser_suspend = subparsers.add_parser('suspend', help='Suspend an enrollment')
    parser_suspend.add_argument('finger_id', help='Finger ID')
    parser_suspend.add_argument('--reason', help='Reason for suspension')
    parser_suspend.add_argument('-y', '--yes', action='store_true',
                              help='Skip confirmation prompts')

    # Reactivate command
    parser_reactivate = subparsers.add_parser('reactivate', help='Reactivate an enrollment')
    parser_reactivate.add_argument('finger_id', help='Finger ID')
    parser_reactivate.add_argument('-y', '--yes', action='store_true',
                                 help='Skip confirmation prompts')

    # Verify command
    parser_verify = subparsers.add_parser('verify', help='Test fingerprint verification')
    parser_verify.add_argument('--user', help='Username')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize auth
    try:
        auth = FingerprintAuth()
    except Exception as e:
        print(f"Error: Failed to initialize FingerprintAuth: {e}")
        return 1

    # Execute command
    commands = {
        'devices': cmd_devices,
        'list': cmd_list,
        'info': cmd_info,
        'enroll': cmd_enroll,
        'delete': cmd_delete,
        'suspend': cmd_suspend,
        'reactivate': cmd_reactivate,
        'verify': cmd_verify
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
