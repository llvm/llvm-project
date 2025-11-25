#!/usr/bin/env python3
"""
DSMIL Device Control Menu

Interactive text-based menu system for managing and controlling DSMIL devices.
Provides intuitive device browsing, operation execution, and status monitoring.

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dsmil_auto_discover import (
    get_device, get_all_devices, get_devices_by_group,
    initialize_all_devices, get_integration_summary, list_devices
)
from lib.device_registry import DeviceGroup, DeviceRiskLevel


class DSMILMenu:
    """Interactive menu system for DSMIL devices"""

    def __init__(self):
        self.devices_initialized = False
        self.clear_screen()

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name != 'nt' else 'cls')

    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")

    def print_banner(self):
        """Print main banner"""
        self.clear_screen()
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "DSMIL DEVICE CONTROL MENU" + " " * 33 + "║")
        print("║" + " " * 15 + "Dell Latitude 5450 MIL-SPEC Platform" + " " * 27 + "║")
        print("╚" + "═" * 78 + "╝")

    def pause(self):
        """Pause and wait for user input"""
        input("\nPress Enter to continue...")

    def get_input(self, prompt, valid_choices=None):
        """Get user input with validation"""
        while True:
            choice = input(f"\n{prompt}: ").strip()
            if valid_choices is None or choice in valid_choices:
                return choice
            print(f"Invalid choice. Please select from: {', '.join(valid_choices)}")

    # Main Menu

    def main_menu(self):
        """Display main menu"""
        while True:
            self.print_banner()

            summary = get_integration_summary()
            print(f"\n  Total Devices: {summary['total_registered']} | "
                  f"Initialized: {summary['initialized']} | "
                  f"Status: {'Ready' if self.devices_initialized else 'Not Initialized'}\n")

            print("  [1] Browse All Devices (80 devices)")
            print("  [2] Browse by Device Group")
            print("  [3] Browse All Standard Devices (Complete List)")
            print("  [4] System Status")
            print("  [5] Initialize All Devices")
            print("  [6] Quick Device Access")
            print("  [0] Exit")

            choice = self.get_input("Select option", ["0", "1", "2", "3", "4", "5", "6"])

            if choice == "0":
                print("\nExiting DSMIL Control Menu. Goodbye!\n")
                break
            elif choice == "1":
                self.browse_all_devices()
            elif choice == "2":
                self.browse_by_group()
            elif choice == "3":
                self.browse_all_standard_devices()
            elif choice == "4":
                self.system_status()
            elif choice == "5":
                self.initialize_devices()
            elif choice == "6":
                self.quick_device_access()

    # Device Browsing

    def browse_all_devices(self):
        """Browse all registered devices"""
        self.clear_screen()
        self.print_header("All Registered Devices (Grouped)")

        devices = list_devices()

        # Group devices by their group
        grouped = {}
        for device_info in devices:
            group = device_info['group']
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(device_info)

        # Create flat list for indexing
        flat_list = []
        current_index = 1

        # Display grouped
        for group_name in sorted(grouped.keys()):
            print(f"\n  ━━ {group_name} ━━")
            for device_info in grouped[group_name]:
                device_id = device_info['device_id']
                name = device_info['name'].replace('Device', '').strip()
                risk = device_info['risk_level']
                state = device_info.get('state', 'uninitialized')

                print(f"    [{current_index:2}] {device_id:12} {name:30} ({risk})")
                print(f"        State: {state}")

                flat_list.append(device_info)
                current_index += 1

        print("\n  [0] Back to Main Menu")

        choices = ["0"] + [str(i) for i in range(1, len(flat_list) + 1)]
        choice = self.get_input("Select device number for details", choices)

        if choice != "0":
            selected_device = flat_list[int(choice) - 1]
            device_id = int(selected_device['device_id'].split('x')[1], 16)
            self.device_detail_menu(device_id)

    def browse_by_group(self):
        """Browse devices by group"""
        self.clear_screen()
        self.print_header("Browse by Device Group")

        groups = [
            ("1", DeviceGroup.GROUP_0_CORE_SECURITY, "Core Security"),
            ("2", DeviceGroup.GROUP_1_EXTENDED_SECURITY, "Extended Security"),
            ("3", DeviceGroup.GROUP_2_NETWORK_COMMS, "Network/Communications"),
            ("4", DeviceGroup.GROUP_3_DATA_PROCESSING, "Data Processing"),
            ("5", DeviceGroup.GROUP_4_STORAGE_MANAGEMENT, "Storage/Management"),
            ("6", DeviceGroup.GROUP_5_PERIPHERAL_CONTROL, "Peripheral/Control"),
            ("7", DeviceGroup.GROUP_6_TRAINING_SIMULATION, "Training/Simulation"),
        ]

        for choice, _, name in groups:
            print(f"  [{choice}] {name}")

        print("  [0] Back to Main Menu")

        valid = ["0"] + [g[0] for g in groups]
        choice = self.get_input("Select group", valid)

        if choice != "0":
            selected_group = groups[int(choice) - 1]
            self.show_group_devices(selected_group[1], selected_group[2])

    def show_group_devices(self, group, group_name):
        """Show devices in a specific group"""
        self.clear_screen()
        self.print_header(f"Group: {group_name}")

        devices = get_devices_by_group(group)

        if not devices:
            print(f"  No devices registered in {group_name}")
        else:
            for i, device in enumerate(devices, 1):
                print(f"  [{i}] 0x{device.device_id:04X} - {device.name}")
                print(f"      {device.description}")
                print(f"      State: {device.state.value}\n")

            print("  [0] Back")

            choices = ["0"] + [str(i) for i in range(1, len(devices) + 1)]
            choice = self.get_input("Select device for details", choices)

            if choice != "0":
                selected_device = devices[int(choice) - 1]
                self.device_detail_menu(selected_device.device_id)

        self.pause()

    def browse_all_standard_devices(self):
        """Browse all 80 standard devices (v2.0.0 Auto-Discovery)"""
        self.clear_screen()
        self.print_header("All Standard Devices (v2.0.0 - 80 Devices)")

        print("  ━━━ GROUP 0: Core Security (9 devices) ━━━")
        print("  ━━━ GROUP 1: Extended Security (12 devices) ━━━")
        print("  ━━━ GROUP 2: Network/Communications (11 devices) ━━━")
        print("  ━━━ GROUP 3: Data Processing (11 devices) ━━━")
        print("  ━━━ GROUP 4: Storage Management (12 devices) ━━━")
        print("  ━━━ GROUP 5: Peripheral Control (12 devices) ━━━")
        print("  ━━━ GROUP 6: Training/Simulation (12 devices) ━━━")
        print("  ━━━ EXTENDED: Additional Device (0x805A) ━━━")
        print()

        # Get all devices and organize by group
        devices = list_devices()

        # Group by device group
        grouped = {}
        for device_info in devices:
            group = device_info['group']
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(device_info)

        # Display summary with counts
        device_list = []
        device_index = 1

        for group_name in sorted(grouped.keys()):
            group_devices = grouped[group_name]
            print(f"\n  {group_name} ({len(group_devices)} devices):")

            for device_info in sorted(group_devices, key=lambda x: x['device_id']):
                device_id = device_info['device_id']
                name = device_info['name']
                risk = device_info['risk_level']

                # Shorten long names for display
                if len(name) > 25:
                    name = name[:22] + "..."

                print(f"    [{device_index:2}] {device_id} - {name:26} [{risk}]")
                device_list.append(device_info)
                device_index += 1

        print("\n  [0] Back to Main Menu")
        print(f"\n  Total: {len(device_list)} devices integrated")

        choices = ["0"] + [str(i) for i in range(1, len(device_list) + 1)]
        choice = self.get_input("Select device number for details", choices)

        if choice != "0":
            selected_device = device_list[int(choice) - 1]
            device_id = int(selected_device['device_id'].split('x')[1], 16)
            self.device_detail_menu(device_id)

    def quick_device_access(self):
        """Quick access by device ID"""
        self.clear_screen()
        self.print_header("Quick Device Access")

        print("  Enter device ID in hex (e.g., 8005, 8016, 801E, 8050, 805A)")
        print("  or press Enter to cancel\n")

        device_hex = input("Device ID (hex): ").strip()

        if not device_hex:
            return

        try:
            device_id = int(device_hex, 16)
            device = get_device(device_id)

            if device:
                self.device_detail_menu(device_id)
            else:
                print(f"\n  Device 0x{device_id:04X} not found or not registered")
                self.pause()
        except ValueError:
            print("\n  Invalid hex format")
            self.pause()

    # Device Detail and Operations

    def device_detail_menu(self, device_id):
        """Show device details and operations menu"""
        device = get_device(device_id)

        if not device:
            print(f"\n  Device 0x{device_id:04X} not found")
            self.pause()
            return

        while True:
            self.clear_screen()
            self.print_header(f"Device 0x{device_id:04X}: {device.name}")

            print(f"  Description: {device.description}")
            print(f"  State: {device.state.value}")
            print(f"  Capabilities: {len(device.get_capabilities())}")
            print(f"  Registers: {len(device.get_register_map())}\n")

            print("  [1] Initialize Device")
            print("  [2] View Status")
            print("  [3] View Register Map")
            print("  [4] Device-Specific Operations")
            print("  [5] View Statistics")
            print("  [0] Back")

            choice = self.get_input("Select option", ["0", "1", "2", "3", "4", "5"])

            if choice == "0":
                break
            elif choice == "1":
                self.initialize_device(device)
            elif choice == "2":
                self.view_device_status(device)
            elif choice == "3":
                self.view_register_map(device)
            elif choice == "4":
                self.device_operations_menu(device)
            elif choice == "5":
                self.view_device_statistics(device)

    def initialize_device(self, device):
        """Initialize a device"""
        self.clear_screen()
        self.print_header(f"Initialize: {device.name}")

        print("  Initializing device...")
        result = device.initialize()

        if result.success:
            print("\n  ✓ Device initialized successfully!")
            if result.data:
                print("\n  Initialization Details:")
                for key, value in result.data.items():
                    print(f"    {key}: {value}")
        else:
            print(f"\n  ✗ Initialization failed: {result.error}")

        self.pause()

    def view_device_status(self, device):
        """View device status"""
        self.clear_screen()
        self.print_header(f"Status: {device.name}")

        status = device.get_status()

        for key, value in status.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"  {formatted_key:30} {value}")

        self.pause()

    def view_register_map(self, device):
        """View device register map"""
        self.clear_screen()
        self.print_header(f"Register Map: {device.name}")

        register_map = device.get_register_map()

        print(f"  {'Register':20} {'Offset':10} {'Size':8} {'Access':8} Description")
        print("  " + "-" * 75)

        for reg_name, reg_info in register_map.items():
            offset = f"0x{reg_info['offset']:04X}"
            size = str(reg_info['size'])
            access = reg_info['access']
            desc = reg_info['description'][:35]
            print(f"  {reg_name:20} {offset:10} {size:8} {access:8} {desc}")

        self.pause()

    def view_device_statistics(self, device):
        """View device statistics"""
        self.clear_screen()
        self.print_header(f"Statistics: {device.name}")

        stats = device.get_statistics()

        for key, value in stats.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"  {formatted_key:30} {value}")

        self.pause()

    def device_operations_menu(self, device):
        """Show device-specific operations"""
        device_id = device.device_id

        # Route to device-specific menu
        if device_id == 0x8002:
            self.credential_vault_operations(device)
        elif device_id == 0x8003:
            self.audit_log_operations(device)
        elif device_id == 0x8004:
            self.event_logger_operations(device)
        elif device_id == 0x8005:
            self.performance_monitor_operations(device)
        elif device_id == 0x8006:
            self.thermal_sensor_operations(device)
        elif device_id == 0x8007:
            self.power_state_operations(device)
        elif device_id == 0x8008:
            self.emergency_response_operations(device)
        elif device_id == 0x8010:
            self.intrusion_detection_operations(device)
        elif device_id == 0x8013:
            self.key_management_operations(device)
        elif device_id == 0x8014:
            self.certificate_store_operations(device)
        elif device_id == 0x8016:
            self.vpn_controller_operations(device)
        elif device_id == 0x8017:
            self.remote_access_operations(device)
        elif device_id == 0x8018:
            self.pre_isolation_operations(device)
        elif device_id == 0x801A:
            self.port_security_operations(device)
        elif device_id == 0x801B:
            self.wireless_security_operations(device)
        elif device_id == 0x801E:
            self.tactical_display_operations(device)
        elif device_id == 0x802A:
            self.network_monitor_operations(device)
        elif device_id == 0x802B:
            self.packet_filter_operations(device)
        elif device_id == 0x8050:
            self.storage_encryption_operations(device)
        elif device_id == 0x805A:
            self.sensor_array_operations(device)
        else:
            # Generic operations menu for all other devices
            self.generic_device_operations(device)

    # Generic Operations (for all 80 devices)

    def generic_device_operations(self, device):
        """Generic operations menu for devices without custom menus"""
        while True:
            self.clear_screen()
            self.print_header(f"Device Operations: {device.name}")

            print(f"  Device ID: 0x{device.device_id:04X}")
            print(f"  Description: {device.description}")
            print(f"  State: {device.state.value}\n")

            print("  [1] Get Device Status")
            print("  [2] Get Capabilities")
            print("  [3] Read All Registers")
            print("  [4] Get Statistics")
            print("  [5] Perform Self-Test")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5"])

            if choice == "0":
                break
            elif choice == "1":
                self.clear_screen()
                self.print_header(f"Device Status: {device.name}")
                status = device.get_status()
                for key, value in status.items():
                    formatted_key = key.replace('_', ' ').title()
                    print(f"  {formatted_key:30} {value}")
                self.pause()
            elif choice == "2":
                self.clear_screen()
                self.print_header(f"Capabilities: {device.name}")
                caps = device.get_capabilities()
                if caps:
                    print(f"  Total Capabilities: {len(caps)}\n")
                    for i, cap in enumerate(caps, 1):
                        print(f"  [{i}] {cap}")
                else:
                    print("  No capabilities defined")
                self.pause()
            elif choice == "3":
                self.clear_screen()
                self.print_header(f"Register Map: {device.name}")
                registers = device.get_register_map()
                if registers:
                    print(f"  {'Register':20} {'Offset':10} {'Size':8} {'Access':8}")
                    print("  " + "-" * 60)
                    for reg_name, reg_info in sorted(registers.items(), key=lambda x: x[1]['offset']):
                        offset = reg_info['offset']
                        size = reg_info['size']
                        access = reg_info['access']
                        print(f"  {reg_name:20} 0x{offset:04X}     {size:4}    {access:8}")

                        # Try to read the register value
                        try:
                            result = device.read_register(offset)
                            if result.success:
                                print(f"      Value: 0x{result.data:08X} ({result.data})")
                        except:
                            pass
                else:
                    print("  No registers defined")
                self.pause()
            elif choice == "4":
                self.clear_screen()
                self.print_header(f"Statistics: {device.name}")
                # Try to get statistics if the device has this method
                if hasattr(device, 'get_statistics'):
                    stats = device.get_statistics()
                    if stats:
                        for key, value in stats.items():
                            formatted_key = key.replace('_', ' ').title()
                            print(f"  {formatted_key:30} {value}")
                    else:
                        print("  No statistics available")
                else:
                    print("  Statistics not supported for this device")
                self.pause()
            elif choice == "5":
                self.clear_screen()
                self.print_header(f"Self-Test: {device.name}")
                print("  Running self-test...")
                # Try basic initialization as self-test
                result = device.initialize()
                if result.success:
                    print("\n  ✓ Self-test passed")
                    if result.data:
                        print("\n  Test Results:")
                        for key, value in result.data.items():
                            print(f"    {key}: {value}")
                else:
                    print(f"\n  ✗ Self-test failed: {result.error}")
                self.pause()

    # Device-Specific Operations (Custom menus for 20 devices)

    def performance_monitor_operations(self, device):
        """Performance Monitor specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Performance Monitor Operations: {device.name}")

            print("  [1] Get Current Metrics")
            print("  [2] Get Metrics Summary")
            print("  [3] Get Thermal Status")
            print("  [4] Get HSM Status")
            print("  [5] Get TPM Activity")
            print("  [6] Get Crypto Performance")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5", "6"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_current_metrics()
                self.display_operation_result("Current Metrics", result)
            elif choice == "2":
                result = device.get_metrics_summary()
                self.display_operation_result("Metrics Summary", result)
            elif choice == "3":
                result = device.get_thermal_status()
                self.display_operation_result("Thermal Status", result)
            elif choice == "4":
                result = device.get_hsm_status()
                self.display_operation_result("HSM Status", result)
            elif choice == "5":
                result = device.get_tpm_activity()
                self.display_operation_result("TPM Activity", result)
            elif choice == "6":
                result = device.get_crypto_performance()
                self.display_operation_result("Crypto Performance", result)

    def vpn_controller_operations(self, device):
        """VPN Controller specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"VPN Controller Operations: {device.name}")

            print("  [1] List VPN Tunnels")
            print("  [2] Get Encryption Info")
            print("  [3] Get Statistics")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_tunnels()
                self.display_operation_result("VPN Tunnels", result)
            elif choice == "2":
                result = device.get_encryption_info()
                self.display_operation_result("Encryption Info", result)
            elif choice == "3":
                result = device.get_statistics()
                self.display_dict_result("VPN Statistics", result)

    def tactical_display_operations(self, device):
        """Tactical Display specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Tactical Display Operations: {device.name}")

            print("  [1] List Displays")
            print("  [2] Get Security Configuration")
            print("  [3] Get Display Modes")
            print("  [4] Get TEMPEST Status")
            print("  [5] Get Protection Features")
            print("  [6] Get Overlay Status")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5", "6"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_displays()
                self.display_operation_result("Displays", result)
            elif choice == "2":
                result = device.get_security_config()
                self.display_operation_result("Security Configuration", result)
            elif choice == "3":
                result = device.get_display_modes()
                self.display_operation_result("Display Modes", result)
            elif choice == "4":
                result = device.get_tempest_status()
                self.display_operation_result("TEMPEST Status", result)
            elif choice == "5":
                result = device.get_protection_features()
                self.display_operation_result("Protection Features", result)
            elif choice == "6":
                result = device.get_overlay_status()
                self.display_operation_result("Overlay Status", result)

    def storage_encryption_operations(self, device):
        """Storage Encryption specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Storage Encryption Operations: {device.name}")

            print("  [1] List Volumes")
            print("  [2] List SED Drives")
            print("  [3] Get Encryption Configuration")
            print("  [4] Get Encryption Performance")
            print("  [5] Get Key Management Info")
            print("  [6] Get OPAL Support")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5", "6"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_volumes()
                self.display_operation_result("Volumes", result)
            elif choice == "2":
                result = device.list_sed_drives()
                self.display_operation_result("SED Drives", result)
            elif choice == "3":
                result = device.get_encryption_config()
                self.display_operation_result("Encryption Configuration", result)
            elif choice == "4":
                result = device.get_encryption_performance()
                self.display_operation_result("Encryption Performance", result)
            elif choice == "5":
                result = device.get_key_management_info()
                self.display_operation_result("Key Management Info", result)
            elif choice == "6":
                result = device.get_opal_support()
                self.display_operation_result("OPAL Support", result)

    def sensor_array_operations(self, device):
        """Sensor Array specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Sensor Array Operations: {device.name}")

            print("  [1] List Sensors")
            print("  [2] Get Environmental Summary")
            print("  [3] Get Security Summary")
            print("  [4] Get Radiation Status")
            print("  [5] Get Fusion Data")
            print("  [6] Get Alert Summary")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5", "6"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_sensors()
                self.display_operation_result("Sensors", result)
            elif choice == "2":
                result = device.get_environmental_summary()
                self.display_operation_result("Environmental Summary", result)
            elif choice == "3":
                result = device.get_security_summary()
                self.display_operation_result("Security Summary", result)
            elif choice == "4":
                result = device.get_radiation_status()
                self.display_operation_result("Radiation Status", result)
            elif choice == "5":
                result = device.get_fusion_data()
                self.display_operation_result("Sensor Fusion Data", result)
            elif choice == "6":
                result = device.get_alert_summary()
                self.display_operation_result("Alert Summary", result)

    def credential_vault_operations(self, device):
        """Credential Vault specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Credential Vault Operations: {device.name}")

            print("  [1] List Credentials")
            print("  [2] Get Vault Policy")
            print("  [3] Get Capacity Info")
            print("  [4] Get Access Log")
            print("  [5] Get Statistics")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_credentials()
                self.display_operation_result("Credentials", result)
            elif choice == "2":
                result = device.get_vault_policy()
                self.display_operation_result("Vault Policy", result)
            elif choice == "3":
                result = device.get_capacity_info()
                self.display_operation_result("Capacity Info", result)
            elif choice == "4":
                result = device.get_access_log()
                self.display_operation_result("Access Log", result)
            elif choice == "5":
                result = device.get_statistics()
                self.display_dict_result("Vault Statistics", result)

    def packet_filter_operations(self, device):
        """Packet Filter specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Packet Filter Operations: {device.name}")

            print("  [1] Get Filter Rules")
            print("  [2] Get Statistics")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_filter_rules()
                self.display_operation_result("Filter Rules", result)
            elif choice == "2":
                result = device.get_statistics()
                self.display_dict_result("Filter Statistics", result)

    # System Operations

    def system_status(self):
        """Display system status"""
        self.clear_screen()
        self.print_header("System Status")

        summary = get_integration_summary()

        print(f"  Integration: {summary['integration_name']} v{summary['version']}")
        print(f"\n  Total Devices: {summary['total_registered']}")
        print(f"  Enabled: {summary['enabled']}")
        print(f"  Initialized: {summary['initialized']}")

        print("\n  Devices by Group:")
        for group_name, count in sorted(summary['by_group'].items()):
            print(f"    {group_name:40} {count:3} device(s)")

        print("\n  Devices by Risk Level:")
        for risk_name, count in sorted(summary['by_risk'].items()):
            print(f"    {risk_name:15} {count:3} device(s)")

        self.pause()

    def initialize_devices(self):
        """Initialize all devices"""
        self.clear_screen()
        self.print_header("Initialize All Devices")

        print("  Initializing all registered devices...\n")

        results = initialize_all_devices()

        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)

        for device_id, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"  {status} Device 0x{device_id:04X}: {'SUCCESS' if result.success else result.error}")

        print(f"\n  Results: {success_count}/{total_count} devices initialized successfully")

        if success_count == total_count:
            self.devices_initialized = True

        self.pause()

    # Helper Methods

    def display_operation_result(self, title, result):
        """Display operation result"""
        self.clear_screen()
        self.print_header(title)

        if result.success:
            print("  ✓ Operation successful\n")
            if result.data:
                self.print_dict_recursive(result.data, indent=2)
        else:
            print(f"  ✗ Operation failed: {result.error}")

        self.pause()

    def display_dict_result(self, title, data):
        """Display dictionary result"""
        self.clear_screen()
        self.print_header(title)
        self.print_dict_recursive(data, indent=2)
        self.pause()

    def print_dict_recursive(self, data, indent=0):
        """Print dictionary recursively"""
        indent_str = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                formatted_key = str(key).replace('_', ' ').title()
                if isinstance(value, (dict, list)):
                    print(f"{indent_str}{formatted_key}:")
                    self.print_dict_recursive(value, indent + 1)
                else:
                    print(f"{indent_str}{formatted_key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    print(f"{indent_str}[{i + 1}]")
                    self.print_dict_recursive(item, indent + 1)
                else:
                    print(f"{indent_str}[{i + 1}] {item}")
        else:
            print(f"{indent_str}{data}")

    # Device-Specific Operations (v2.0.0 - All 80 Devices)

    def audit_log_operations(self, device):
        """Audit Log specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Audit Log Operations: {device.name}")

            print("  [1] Get Recent Entries")
            print("  [2] Get Entries by Severity")
            print("  [3] Get Entries by Category")
            print("  [4] Get Summary")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_recent_entries(limit=10)
                self.display_operation_result("Recent Audit Entries", result)
            elif choice == "2":
                print("\n  Severities: 0=Debug, 1=Info, 2=Warning, 3=Security, 4=Critical")
                sev = input("  Enter severity level: ").strip()
                try:
                    result = device.get_entries_by_severity(int(sev))
                    self.display_operation_result(f"Entries (Severity {sev})", result)
                except ValueError:
                    print("  Invalid severity level")
                    self.pause()
            elif choice == "3":
                print("\n  Categories: 0=System, 1=Security, 2=Access, 3=Configuration")
                cat = input("  Enter category: ").strip()
                try:
                    result = device.get_entries_by_category(int(cat))
                    self.display_operation_result(f"Entries (Category {cat})", result)
                except ValueError:
                    print("  Invalid category")
                    self.pause()
            elif choice == "4":
                result = device.get_summary()
                self.display_operation_result("Audit Log Summary", result)

    def event_logger_operations(self, device):
        """Event Logger specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Event Logger Operations: {device.name}")

            print("  [1] Get Recent Events")
            print("  [2] Get Events by Level")
            print("  [3] Get Error Events")
            print("  [4] Get Summary")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_recent_events(limit=10)
                self.display_operation_result("Recent Events", result)
            elif choice == "2":
                print("\n  Levels: 0=Debug, 1=Info, 2=Warning, 3=Error, 4=Critical")
                level = input("  Enter event level: ").strip()
                try:
                    result = device.get_events_by_level(int(level))
                    self.display_operation_result(f"Events (Level {level})", result)
                except ValueError:
                    print("  Invalid level")
                    self.pause()
            elif choice == "3":
                result = device.get_error_events()
                self.display_operation_result("Error Events", result)
            elif choice == "4":
                result = device.get_summary()
                self.display_operation_result("Event Logger Summary", result)

    def thermal_sensor_operations(self, device):
        """Thermal Sensor specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Thermal Sensor Operations: {device.name}")

            print("  [1] Get All Temperatures")
            print("  [2] Get CPU Temperature")
            print("  [3] Get Thermal Summary")
            print("  [4] Get Alert History")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_all_temperatures()
                self.display_operation_result("All Temperatures", result)
            elif choice == "2":
                result = device.get_zone_temperature(0)  # CPU = 0
                self.display_operation_result("CPU Temperature", result)
            elif choice == "3":
                result = device.get_thermal_summary()
                self.display_operation_result("Thermal Summary", result)
            elif choice == "4":
                result = device.get_alert_history(limit=10)
                self.display_operation_result("Alert History", result)

    def power_state_operations(self, device):
        """Power State specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Power State Operations: {device.name}")

            print("  [1] Get Power Summary")
            print("  [2] Get Battery Info")
            print("  [3] Get Performance Info")
            print("  [4] Get State Transitions")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_power_summary()
                self.display_operation_result("Power Summary", result)
            elif choice == "2":
                result = device.get_battery_info()
                self.display_operation_result("Battery Info", result)
            elif choice == "3":
                result = device.get_performance_info()
                self.display_operation_result("Performance Info", result)
            elif choice == "4":
                result = device.get_state_transitions(limit=10)
                self.display_operation_result("State Transitions", result)

    def emergency_response_operations(self, device):
        """Emergency Response specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Emergency Response Operations: {device.name}")

            print("  [1] Validate Boot Chain")
            print("  [2] Get Boot Chain Status")
            print("  [3] Get Integrity Report")
            print("  [4] Get Emergency Status")
            print("  [5] Get Alert History")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.validate_boot_chain()
                self.display_operation_result("Boot Chain Validation", result)
            elif choice == "2":
                result = device.get_boot_chain_status()
                self.display_operation_result("Boot Chain Status", result)
            elif choice == "3":
                result = device.get_integrity_report()
                self.display_operation_result("Integrity Report", result)
            elif choice == "4":
                result = device.get_emergency_status()
                self.display_operation_result("Emergency Status", result)
            elif choice == "5":
                result = device.get_alert_history(limit=10)
                self.display_operation_result("Alert History", result)

    def key_management_operations(self, device):
        """Key Management specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Key Management Operations: {device.name}")

            print("  [1] List All Keys")
            print("  [2] List Active Keys")
            print("  [3] Get Keys by Type")
            print("  [4] Get Storage Summary")
            print("  [5] Get Key Summary")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_keys()
                self.display_operation_result("All Keys", result)
            elif choice == "2":
                result = device.list_keys(status_filter=0)  # ACTIVE = 0
                self.display_operation_result("Active Keys", result)
            elif choice == "3":
                print("\n  Key Types: 0=Symmetric, 1=RSA, 2=ECC, 3=HMAC, 4=Session")
                key_type = input("  Enter key type: ").strip()
                try:
                    result = device.get_keys_by_type(int(key_type))
                    self.display_operation_result(f"Keys (Type {key_type})", result)
                except ValueError:
                    print("  Invalid key type")
                    self.pause()
            elif choice == "4":
                result = device.get_storage_summary()
                self.display_operation_result("Storage Summary", result)
            elif choice == "5":
                result = device.get_key_summary()
                self.display_operation_result("Key Summary", result)

    def remote_access_operations(self, device):
        """Remote Access specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Remote Access Operations: {device.name}")

            print("  [1] List All Sessions")
            print("  [2] List Active Sessions")
            print("  [3] Get Access History")
            print("  [4] Get Failed Attempts")
            print("  [5] Get Security Config")
            print("  [6] Get Session Summary")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5", "6"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_sessions()
                self.display_operation_result("All Sessions", result)
            elif choice == "2":
                result = device.get_active_sessions()
                self.display_operation_result("Active Sessions", result)
            elif choice == "3":
                result = device.get_access_history(limit=10)
                self.display_operation_result("Access History", result)
            elif choice == "4":
                result = device.get_failed_attempts()
                self.display_operation_result("Failed Attempts", result)
            elif choice == "5":
                result = device.get_security_config()
                self.display_operation_result("Security Config", result)
            elif choice == "6":
                result = device.get_session_summary()
                self.display_operation_result("Session Summary", result)

    def pre_isolation_operations(self, device):
        """Pre-Isolation specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Pre-Isolation Operations: {device.name}")

            print("  [1] Get Isolation Status")
            print("  [2] List Isolated Systems")
            print("  [3] Get Threat Assessment")
            print("  [4] Get Network Zones")
            print("  [5] Get Isolation Summary")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_isolation_status()
                self.display_operation_result("Isolation Status", result)
            elif choice == "2":
                result = device.list_isolated_systems()
                self.display_operation_result("Isolated Systems", result)
            elif choice == "3":
                result = device.get_threat_assessment()
                self.display_operation_result("Threat Assessment", result)
            elif choice == "4":
                result = device.get_network_zones()
                self.display_operation_result("Network Zones", result)
            elif choice == "5":
                result = device.get_isolation_summary()
                self.display_operation_result("Isolation Summary", result)

    def network_monitor_operations(self, device):
        """Network Monitor specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Network Monitor Operations: {device.name}")

            print("  [1] List Interfaces")
            print("  [2] Get Traffic Stats")
            print("  [3] Get Protocol Breakdown")
            print("  [4] Get Bandwidth Usage")
            print("  [5] Get Anomalies")
            print("  [6] Get Summary")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3", "4", "5", "6"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_interfaces()
                self.display_operation_result("Network Interfaces", result)
            elif choice == "2":
                result = device.get_traffic_stats()
                self.display_operation_result("Traffic Statistics", result)
            elif choice == "3":
                result = device.get_protocol_breakdown()
                self.display_operation_result("Protocol Breakdown", result)
            elif choice == "4":
                result = device.get_bandwidth_usage()
                self.display_operation_result("Bandwidth Usage", result)
            elif choice == "5":
                result = device.get_anomalies(limit=10)
                self.display_operation_result("Anomalies", result)
            elif choice == "6":
                result = device.get_summary()
                self.display_operation_result("Network Monitor Summary", result)

    def intrusion_detection_operations(self, device):
        """Intrusion Detection specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Intrusion Detection Operations: {device.name}")

            print("  [1] Get Sensor States")
            print("  [2] Get Intrusion Events")
            print("  [3] Reset Device")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_sensor_states()
                self.display_operation_result("Sensor States", result)
            elif choice == "2":
                result = device.get_intrusion_events()
                self.display_operation_result("Intrusion Events", result)
            elif choice == "3":
                result = device.reset()
                self.display_operation_result("Device Reset", result)

    def certificate_store_operations(self, device):
        """Certificate Store specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Certificate Store Operations: {device.name}")

            print("  [1] List Certificates")
            print("  [2] Check PQC Compliance")
            print("  [3] Get Statistics")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2", "3"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.list_certificates()
                self.display_operation_result("Certificates", result)
            elif choice == "2":
                result = device.check_pqc_compliance()
                self.display_operation_result("PQC Compliance Report", result)
            elif choice == "3":
                result = device.get_statistics()
                self.display_dict_result("Certificate Store Statistics", result)

    def port_security_operations(self, device):
        """Port Security specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Port Security Operations: {device.name}")

            print("  [1] List Ports")
            print("  [2] Get Statistics")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_port_list()
                self.display_operation_result("Port List", result)
            elif choice == "2":
                result = device.get_statistics()
                self.display_dict_result("Port Security Statistics", result)

    def wireless_security_operations(self, device):
        """Wireless Security specific operations"""
        while True:
            self.clear_screen()
            self.print_header(f"Wireless Security Operations: {device.name}")

            print("  [1] List Wireless Interfaces")
            print("  [2] Get Statistics")
            print("  [0] Back")

            choice = self.get_input("Select operation", ["0", "1", "2"])

            if choice == "0":
                break
            elif choice == "1":
                result = device.get_interface_list()
                self.display_operation_result("Wireless Interfaces", result)
            elif choice == "2":
                result = device.get_statistics()
                self.display_dict_result("Wireless Security Statistics", result)


def main():
    """Main entry point"""
    try:
        menu = DSMILMenu()
        menu.main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
