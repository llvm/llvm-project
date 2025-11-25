#!/usr/bin/env python3
"""
DSMIL Device Functional Prober

Probes DSMIL device functionality to discover what each device does, what
operations are available, and tests safe operations to verify functionality.
Automatically avoids quarantined devices.

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dsmil_auto_discover import (
    get_device, get_all_devices, list_devices,
    get_integration_summary, initialize_all_devices
)
from lib.device_registry import DeviceRiskLevel, DeviceGroup


class DSMILFunctionalProber:
    """Functional testing and capability probing for DSMIL devices"""

    # Quarantined devices - NEVER probe these
    QUARANTINED_DEVICES = [
        0x8009,  # Self-Destruct
        0x800A,  # Secure Erase
        0x800B,  # Emergency Lockdown
        0x8019,  # Remote Disable
        0x8029,  # System Reset
    ]

    def __init__(self):
        self.results = {}
        self.summary = {
            'total_devices': 0,
            'probed': 0,
            'quarantined_skipped': 0,
            'functional': 0,
            'partial': 0,
            'non_functional': 0,
            'errors': 0,
        }

    def is_quarantined(self, device_id: int) -> bool:
        """Check if device is quarantined"""
        return device_id in self.QUARANTINED_DEVICES

    def get_device_purpose(self, device_id: int, name: str) -> str:
        """Get detailed device purpose description"""
        purposes = {
            0x8000: "TPM 2.0 cryptographic operations, secure key storage, attestation, and random number generation",
            0x8001: "Secure boot verification, boot chain integrity, and boot policy enforcement",
            0x8002: "Secure credential storage and management for passwords, keys, and certificates",
            0x8005: "System performance monitoring, TPM/HSM interface coordination, and thermal management",
            0x8010: "Physical intrusion detection and chassis tamper monitoring",
            0x8014: "PKI certificate store management and X.509 certificate operations",
            0x8016: "VPN tunnel management with multi-protocol support (IPsec, WireGuard, OpenVPN)",
            0x801A: "Physical port security control and USB/peripheral access management",
            0x801B: "Wireless communication security and RF signal management",
            0x801E: "Military-grade display security with TEMPEST compliance and content protection",
            0x802B: "Hardware packet filtering and network traffic control",
            0x8050: "Full disk encryption (FDE), self-encrypting drive (SED) management with OPAL 2.0",
            0x805A: "Multi-sensor array for environmental, security, and radiation monitoring with fusion",
        }
        return purposes.get(device_id, f"Unknown function for {name}")

    def get_safe_operations(self, device_id: int) -> List[str]:
        """Get list of safe operations to test for each device"""
        safe_ops = {
            0x8000: ['get_status', 'get_capabilities', 'get_statistics'],
            0x8001: ['get_status', 'get_capabilities', 'get_statistics'],
            0x8002: ['get_status', 'get_capabilities', 'get_statistics'],
            0x8005: ['get_status', 'get_capabilities', 'get_current_metrics',
                     'get_thermal_status', 'get_hsm_status', 'get_tpm_activity',
                     'get_crypto_performance', 'get_statistics'],
            0x8010: ['get_status', 'get_capabilities', 'get_statistics'],
            0x8014: ['get_status', 'get_capabilities', 'get_statistics'],
            0x8016: ['get_status', 'get_capabilities', 'list_tunnels',
                     'get_encryption_info', 'get_statistics'],
            0x801A: ['get_status', 'get_capabilities', 'get_statistics'],
            0x801B: ['get_status', 'get_capabilities', 'get_statistics'],
            0x801E: ['get_status', 'get_capabilities', 'list_displays',
                     'get_security_config', 'get_display_modes', 'get_tempest_status',
                     'get_protection_features', 'get_overlay_status', 'get_statistics'],
            0x802B: ['get_status', 'get_capabilities', 'get_statistics'],
            0x8050: ['get_status', 'get_capabilities', 'list_volumes',
                     'list_sed_drives', 'get_encryption_config', 'get_encryption_performance',
                     'get_key_management_info', 'get_opal_support', 'get_statistics'],
            0x805A: ['get_status', 'get_capabilities', 'list_sensors',
                     'get_environmental_summary', 'get_security_summary', 'get_radiation_status',
                     'get_fusion_data', 'get_alert_summary', 'get_statistics'],
        }
        return safe_ops.get(device_id, ['get_status', 'get_capabilities', 'get_statistics'])

    def probe_device(self, device_id: int) -> Dict[str, Any]:
        """Comprehensively probe a single device"""
        print(f"\n  [*] Probing device 0x{device_id:04X}...")

        if self.is_quarantined(device_id):
            print(f"      ⛔ QUARANTINED - Skipping for safety")
            self.summary['quarantined_skipped'] += 1
            return {
                'device_id': f"0x{device_id:04X}",
                'quarantined': True,
                'reason': 'Destructive operations - permanently blocked',
            }

        device = get_device(device_id)
        if not device:
            print(f"      ✗ Device not found in framework")
            return {
                'device_id': f"0x{device_id:04X}",
                'error': 'Not registered in framework',
            }

        result = {
            'device_id': f"0x{device_id:04X}",
            'name': device.name,
            'description': device.description,
            'purpose': self.get_device_purpose(device_id, device.name),
            'quarantined': False,
            'initialized': False,
            'capabilities': [],
            'register_count': 0,
            'registers': {},
            'operations_tested': 0,
            'operations_successful': 0,
            'operations_failed': 0,
            'operation_results': {},
            'functional': False,
        }

        # Initialize device
        print(f"      Initializing...")
        init_result = device.initialize()
        result['initialized'] = init_result.success

        if not init_result.success:
            print(f"      ✗ Initialization failed: {init_result.error}")
            result['init_error'] = init_result.error
            self.summary['errors'] += 1
            return result

        print(f"      ✓ Initialized successfully")

        # Get capabilities
        try:
            capabilities = device.get_capabilities()
            result['capabilities'] = [cap.value for cap in capabilities]
            print(f"      Capabilities: {len(capabilities)}")
        except Exception as e:
            result['capabilities_error'] = str(e)

        # Get register map
        try:
            register_map = device.get_register_map()
            result['register_count'] = len(register_map)
            result['registers'] = {
                name: {
                    'offset': f"0x{reg['offset']:04X}",
                    'size': reg['size'],
                    'access': reg['access'],
                    'description': reg['description']
                }
                for name, reg in list(register_map.items())[:5]  # First 5 registers
            }
            print(f"      Registers: {len(register_map)}")
        except Exception as e:
            result['register_error'] = str(e)

        # Test safe operations
        safe_ops = self.get_safe_operations(device_id)
        print(f"      Testing {len(safe_ops)} operations...")

        for op_name in safe_ops:
            result['operations_tested'] += 1

            try:
                if hasattr(device, op_name):
                    op_func = getattr(device, op_name)
                    op_result = op_func()

                    if hasattr(op_result, 'success'):
                        # It's an OperationResult
                        if op_result.success:
                            result['operations_successful'] += 1
                            result['operation_results'][op_name] = {
                                'status': 'success',
                                'has_data': bool(op_result.data),
                                'data_keys': list(op_result.data.keys()) if op_result.data else []
                            }
                        else:
                            result['operations_failed'] += 1
                            result['operation_results'][op_name] = {
                                'status': 'failed',
                                'error': op_result.error
                            }
                    else:
                        # Direct return (like get_status)
                        result['operations_successful'] += 1
                        result['operation_results'][op_name] = {
                            'status': 'success',
                            'has_data': bool(op_result),
                            'data_keys': list(op_result.keys()) if isinstance(op_result, dict) else []
                        }
                else:
                    result['operations_failed'] += 1
                    result['operation_results'][op_name] = {
                        'status': 'not_available',
                        'error': 'Method not found'
                    }

            except Exception as e:
                result['operations_failed'] += 1
                result['operation_results'][op_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Determine functional status
        success_rate = (result['operations_successful'] / result['operations_tested'] * 100
                       if result['operations_tested'] > 0 else 0)

        if success_rate >= 80:
            result['functional'] = True
            result['functional_status'] = 'FUNCTIONAL'
            self.summary['functional'] += 1
            print(f"      ✓ Device is FUNCTIONAL ({success_rate:.0f}% operations working)")
        elif success_rate >= 50:
            result['functional'] = True
            result['functional_status'] = 'PARTIAL'
            self.summary['partial'] += 1
            print(f"      ⚠ Device is PARTIALLY FUNCTIONAL ({success_rate:.0f}% operations working)")
        else:
            result['functional'] = False
            result['functional_status'] = 'NON-FUNCTIONAL'
            self.summary['non_functional'] += 1
            print(f"      ✗ Device is NON-FUNCTIONAL ({success_rate:.0f}% operations working)")

        result['success_rate'] = success_rate
        self.summary['probed'] += 1

        return result

    def probe_all_devices(self):
        """Probe all registered devices"""
        print("\n" + "=" * 80)
        print("DSMIL Device Functional Probing")
        print("=" * 80)
        print(f"\nStarting functional probe at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        devices = list_devices()
        self.summary['total_devices'] = len(devices)

        print(f"Found {len(devices)} registered devices")
        print(f"Quarantined devices: {len(self.QUARANTINED_DEVICES)} (will be skipped)\n")

        for device_info in devices:
            device_id = int(device_info['device_id'].split('x')[1], 16)
            result = self.probe_device(device_id)
            self.results[device_id] = result

        print("\n" + "=" * 80)
        print("Probing complete!")
        print("=" * 80)

    def print_comprehensive_report(self):
        """Print comprehensive functional report"""
        print("\n" + "=" * 80)
        print("DEVICE FUNCTIONALITY REPORT")
        print("=" * 80)

        # Summary
        print("\n━━━ Summary ━━━")
        print(f"  Total Devices:          {self.summary['total_devices']}")
        print(f"  Probed:                 {self.summary['probed']}")
        print(f"  Quarantined (Skipped):  {self.summary['quarantined_skipped']}")
        print(f"  Functional:             {self.summary['functional']}")
        print(f"  Partially Functional:   {self.summary['partial']}")
        print(f"  Non-Functional:         {self.summary['non_functional']}")
        print(f"  Errors:                 {self.summary['errors']}")

        # Quarantined Devices
        if self.QUARANTINED_DEVICES:
            print("\n━━━ Quarantined Devices (NEVER PROBE) ━━━")
            quarantine_names = {
                0x8009: "Self-Destruct",
                0x800A: "Secure Erase",
                0x800B: "Emergency Lockdown",
                0x8019: "Remote Disable",
                0x8029: "System Reset",
            }
            for qdev in self.QUARANTINED_DEVICES:
                print(f"  ⛔ 0x{qdev:04X} - {quarantine_names.get(qdev, 'Unknown')}")
                print(f"     Reason: Destructive operations - permanently blocked")

        # Device Details
        for device_id, result in sorted(self.results.items()):
            if result.get('quarantined'):
                continue  # Already shown above

            print(f"\n{'━' * 80}")
            print(f"Device 0x{device_id:04X}: {result.get('name', 'Unknown')}")
            print(f"{'━' * 80}")

            if 'error' in result:
                print(f"  ✗ Error: {result['error']}")
                continue

            # Purpose
            print(f"\n  Purpose:")
            print(f"    {result.get('purpose', 'Unknown')}")

            # Description
            print(f"\n  Description:")
            print(f"    {result.get('description', 'Unknown')}")

            # Initialization
            print(f"\n  Initialization:")
            if result.get('initialized'):
                print(f"    ✓ Successfully initialized")
            else:
                print(f"    ✗ Failed to initialize")
                if 'init_error' in result:
                    print(f"      Error: {result['init_error']}")

            # Capabilities
            if result.get('capabilities'):
                print(f"\n  Capabilities ({len(result['capabilities'])}):")
                for cap in result['capabilities']:
                    print(f"    • {cap}")

            # Registers
            if result.get('registers'):
                print(f"\n  Register Map (showing first 5 of {result['register_count']}):")
                for reg_name, reg_info in result['registers'].items():
                    print(f"    {reg_name:20} Offset: {reg_info['offset']:8} "
                          f"Size: {reg_info['size']:2} Access: {reg_info['access']:4}")
                    print(f"      {reg_info['description']}")

            # Operations
            if result.get('operation_results'):
                print(f"\n  Operations Tested: {result['operations_tested']}")
                print(f"    Successful:  {result['operations_successful']}")
                print(f"    Failed:      {result['operations_failed']}")
                print(f"    Success Rate: {result.get('success_rate', 0):.1f}%")

                # Show successful operations
                successful_ops = [op for op, details in result['operation_results'].items()
                                 if details.get('status') == 'success']
                if successful_ops:
                    print(f"\n  ✓ Working Operations ({len(successful_ops)}):")
                    for op in successful_ops[:10]:  # First 10
                        details = result['operation_results'][op]
                        data_info = f" ({len(details.get('data_keys', []))} fields)" if details.get('has_data') else ""
                        print(f"    • {op}{data_info}")

                # Show failed operations
                failed_ops = [op for op, details in result['operation_results'].items()
                             if details.get('status') in ['failed', 'error', 'not_available']]
                if failed_ops:
                    print(f"\n  ✗ Failed Operations ({len(failed_ops)}):")
                    for op in failed_ops[:5]:  # First 5
                        details = result['operation_results'][op]
                        print(f"    • {op}: {details.get('error', 'Unknown error')}")

            # Functional Status
            print(f"\n  Overall Status: {result.get('functional_status', 'UNKNOWN')}")
            if result.get('functional'):
                print(f"    ✓ Device is operational and responding to commands")
            else:
                print(f"    ✗ Device is not responding correctly")

        # Final Statistics
        print("\n" + "=" * 80)
        print("FUNCTIONAL STATISTICS")
        print("=" * 80)

        total_ops = sum(r.get('operations_tested', 0) for r in self.results.values())
        total_success = sum(r.get('operations_successful', 0) for r in self.results.values())
        total_failed = sum(r.get('operations_failed', 0) for r in self.results.values())

        print(f"  Total Operations Tested:     {total_ops}")
        print(f"  Total Successful:            {total_success}")
        print(f"  Total Failed:                {total_failed}")
        if total_ops > 0:
            print(f"  Overall Success Rate:        {(total_success/total_ops*100):.1f}%")

        print("\n  Device Categories:")
        print(f"    Fully Functional:          {self.summary['functional']}")
        print(f"    Partially Functional:      {self.summary['partial']}")
        print(f"    Non-Functional:            {self.summary['non_functional']}")
        print(f"    Quarantined (Skipped):     {self.summary['quarantined_skipped']}")

        print("=" * 80 + "\n")

    def print_quick_summary(self):
        """Print quick functional summary table"""
        print("\n" + "=" * 100)
        print("QUICK FUNCTIONAL SUMMARY")
        print("=" * 100)
        print(f"{'Device ID':12} {'Name':30} {'Status':15} {'Success Rate':13} {'Operations':12}")
        print("-" * 100)

        for device_id, result in sorted(self.results.items()):
            device_id_str = result.get('device_id', f"0x{device_id:04X}")
            name = result.get('name', 'Unknown')[:29]

            if result.get('quarantined'):
                status = "QUARANTINED"
                success_rate = "N/A"
                ops = "Skipped"
            elif 'error' in result:
                status = "ERROR"
                success_rate = "N/A"
                ops = "N/A"
            else:
                status = result.get('functional_status', 'UNKNOWN')
                success_rate = f"{result.get('success_rate', 0):.1f}%"
                ops = f"{result.get('operations_successful', 0)}/{result.get('operations_tested', 0)}"

            print(f"{device_id_str:12} {name:30} {status:15} {success_rate:13} {ops:12}")

        print("=" * 100 + "\n")


def main():
    """Main entry point"""
    print("\n╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "DSMIL DEVICE FUNCTIONAL PROBER" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")

    prober = DSMILFunctionalProber()
    prober.probe_all_devices()
    prober.print_quick_summary()
    prober.print_comprehensive_report()


if __name__ == "__main__":
    main()
