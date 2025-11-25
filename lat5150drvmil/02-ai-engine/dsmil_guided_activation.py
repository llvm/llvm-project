#!/usr/bin/env python3
"""
⚠️  DEPRECATED - This component will be removed in v3.0.0 (2026 Q3)
⚠️  Use: python3 dsmil.py control (then select Option 2 - Activate Safe Devices)
⚠️  See DEPRECATION_PLAN.md for migration guide

DSMIL Guided Activation System
===============================
Methodical TUI-based system for enumerating and activating all 84 DSMIL devices.

This provides a safe, guided process to:
- Enumerate all devices with memory addresses
- Show dependencies and safety status
- Activate devices system-by-system
- Prevent activation of quarantined devices

Memory Map: 0x8000-0x806B (84 devices in 7 groups)
SMI Ports: 0x164E (command), 0x164F (data)

Author: LAT5150DRVMIL AI Platform
Classification: DSMIL Subsystem Management
"""

import curses
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    filename='/tmp/dsmil_guided_activation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DEPRECATION WARNING - REMOVE IN v3.0.0 (2026 Q3)
# ============================================================================
import warnings
warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATED: dsmil_guided_activation.py\n\n"
    "This component is deprecated and will be removed in v3.0.0 (2026 Q3).\n\n"
    "Migration:\n"
    "  - Use: python3 dsmil.py control\n"
    "  - Then select: Option 2 - Activate Safe Devices\n\n"
    "See DEPRECATION_PLAN.md for complete migration guide.\n"
    + "=" * 80,
    DeprecationWarning,
    stacklevel=2
)
# ============================================================================


class DeviceStatus(Enum):
    """Device safety status"""
    SAFE = "safe"
    MONITORED = "monitored"
    QUARANTINED = "quarantined"
    UNKNOWN = "unknown"


class ActivationState(Enum):
    """Device activation state"""
    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class DSMILDevice:
    """DSMIL device information"""
    device_id: int  # Memory token (0x8000-0x806B)
    group: int  # Group 0-6
    device_in_group: int  # Device 0-11 within group
    name: str  # e.g., "DSMIL0D0"
    function: str  # Device function description
    status: DeviceStatus
    dependencies: List[int]  # Required device IDs
    activation_state: ActivationState = ActivationState.INACTIVE
    last_test_time: Optional[float] = None
    last_test_result: Optional[str] = None

    def get_acpi_name(self) -> str:
        """Get ACPI device name (DSMIL0D0-6DB)"""
        return f"DSMIL{self.group}D{self.device_in_group:X}"

    def get_hex_id(self) -> str:
        """Get hex device ID"""
        return f"0x{self.device_id:04X}"


# Complete device database from DSMIL_CURRENT_REFERENCE.md
DSMIL_DEVICES = [
    # Group 0: Core Security & Emergency (0x8000-0x800B)
    DSMILDevice(0x8000, 0, 0, "DSMIL0D0", "Master Controller", DeviceStatus.SAFE, []),
    DSMILDevice(0x8001, 0, 1, "DSMIL0D1", "Cryptographic Engine", DeviceStatus.MONITORED, [0x8000]),
    DSMILDevice(0x8002, 0, 2, "DSMIL0D2", "Secure Key Storage", DeviceStatus.MONITORED, [0x8000]),
    DSMILDevice(0x8003, 0, 3, "DSMIL0D3", "Authentication Module", DeviceStatus.MONITORED, [0x8000]),
    DSMILDevice(0x8004, 0, 4, "DSMIL0D4", "Access Control", DeviceStatus.MONITORED, [0x8000]),
    DSMILDevice(0x8005, 0, 5, "DSMIL0D5", "Audit Logger", DeviceStatus.SAFE, []),
    DSMILDevice(0x8006, 0, 6, "DSMIL0D6", "Integrity Monitor", DeviceStatus.MONITORED, [0x8000]),
    DSMILDevice(0x8007, 0, 7, "DSMIL0D7", "Secure Boot Controller", DeviceStatus.MONITORED, [0x8000]),
    DSMILDevice(0x8008, 0, 8, "DSMIL0D8", "TPM Interface", DeviceStatus.SAFE, []),
    DSMILDevice(0x8009, 0, 9, "DSMIL0D9", "Emergency Wipe", DeviceStatus.QUARANTINED, []),
    DSMILDevice(0x800A, 0, 10, "DSMIL0DA", "Recovery Controller", DeviceStatus.QUARANTINED, []),
    DSMILDevice(0x800B, 0, 11, "DSMIL0DB", "Hidden Memory Controller", DeviceStatus.QUARANTINED, []),

    # Group 1: Extended Security (0x8010-0x801B)
    DSMILDevice(0x8010, 1, 0, "DSMIL1D0", "Group 1 Controller", DeviceStatus.SAFE, [0x8000]),
    DSMILDevice(0x8011, 1, 1, "DSMIL1D1", "Threat Detection", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x8012, 1, 2, "DSMIL1D2", "Intrusion Prevention", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x8013, 1, 3, "DSMIL1D3", "Network Security", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x8014, 1, 4, "DSMIL1D4", "Malware Scanner", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x8015, 1, 5, "DSMIL1D5", "Behavioral Analysis", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x8016, 1, 6, "DSMIL1D6", "Security Policy Engine", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x8017, 1, 7, "DSMIL1D7", "Incident Response", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x8018, 1, 8, "DSMIL1D8", "Forensics Module", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x8019, 1, 9, "DSMIL1D9", "Network Kill Switch", DeviceStatus.QUARANTINED, []),
    DSMILDevice(0x801A, 1, 10, "DSMIL1DA", "Vulnerability Scanner", DeviceStatus.MONITORED, [0x8000, 0x8010]),
    DSMILDevice(0x801B, 1, 11, "DSMIL1DB", "Security Analytics", DeviceStatus.MONITORED, [0x8000, 0x8010]),

    # Group 2: Network & Communications (0x8020-0x802B)
    DSMILDevice(0x8020, 2, 0, "DSMIL2D0", "Network Controller", DeviceStatus.SAFE, [0x8000]),
    DSMILDevice(0x8021, 2, 1, "DSMIL2D1", "Ethernet Manager", DeviceStatus.MONITORED, [0x8000, 0x8010, 0x8020]),
    DSMILDevice(0x8022, 2, 2, "DSMIL2D2", "WiFi Controller", DeviceStatus.MONITORED, [0x8000, 0x8010, 0x8020]),
    DSMILDevice(0x8023, 2, 3, "DSMIL2D3", "Bluetooth Manager", DeviceStatus.MONITORED, [0x8000, 0x8010, 0x8020]),
    DSMILDevice(0x8024, 2, 4, "DSMIL2D4", "VPN Engine", DeviceStatus.MONITORED, [0x8000, 0x8010, 0x8020]),
    DSMILDevice(0x8025, 2, 5, "DSMIL2D5", "Firewall", DeviceStatus.MONITORED, [0x8000, 0x8010, 0x8020]),
    DSMILDevice(0x8026, 2, 6, "DSMIL2D6", "QoS Manager", DeviceStatus.MONITORED, [0x8000, 0x8020]),
    DSMILDevice(0x8027, 2, 7, "DSMIL2D7", "Network Monitor", DeviceStatus.MONITORED, [0x8000, 0x8020]),
    DSMILDevice(0x8028, 2, 8, "DSMIL2D8", "DNS Security", DeviceStatus.MONITORED, [0x8000, 0x8020]),
    DSMILDevice(0x8029, 2, 9, "DSMIL2D9", "Communications Blackout", DeviceStatus.QUARANTINED, []),
    DSMILDevice(0x802A, 2, 10, "DSMIL2DA", "Router Functions", DeviceStatus.MONITORED, [0x8000, 0x8020]),
    DSMILDevice(0x802B, 2, 11, "DSMIL2DB", "Network Storage", DeviceStatus.MONITORED, [0x8000, 0x8020]),

    # Group 3: Data Processing (0x8030-0x803B)
    DSMILDevice(0x8030, 3, 0, "DSMIL3D0", "Processing Controller", DeviceStatus.SAFE, [0x8000]),
] + [
    DSMILDevice(0x8030 + i, 3, i, f"DSMIL3D{i:X}", "Data Processing", DeviceStatus.MONITORED, [0x8000, 0x8030])
    for i in range(1, 12)
] + [
    # Group 4: Storage Control (0x8040-0x804B)
    DSMILDevice(0x8040 + i, 4, i, f"DSMIL4D{i:X}", "Storage Control", DeviceStatus.MONITORED, [0x8000, 0x8030])
    for i in range(12)
] + [
    # Group 5: Peripheral Management (0x8050-0x805B)
    DSMILDevice(0x8050 + i, 5, i, f"DSMIL5D{i:X}", "Peripheral Management", DeviceStatus.MONITORED, [0x8000])
    for i in range(12)
] + [
    # Group 6: Training Functions (0x8060-0x806B)
    DSMILDevice(0x8060 + i, 6, i, f"DSMIL6D{i:X}", "Training/Simulation", DeviceStatus.MONITORED, [])
    for i in range(12)
]

# Create device lookup dictionary
DEVICE_MAP = {dev.device_id: dev for dev in DSMIL_DEVICES}

# Group information
GROUP_INFO = [
    ("Group 0", "Core Security & Emergency", "CRITICAL", "None (root group)"),
    ("Group 1", "Extended Security", "HIGH", "Requires Group 0"),
    ("Group 2", "Network & Communications", "MEDIUM", "Requires Groups 0, 1"),
    ("Group 3", "Data Processing", "MEDIUM", "Requires Group 0"),
    ("Group 4", "Storage Control", "MEDIUM", "Requires Groups 0, 3"),
    ("Group 5", "Peripheral Management", "MEDIUM-LOW", "Requires Group 0"),
    ("Group 6", "Training Functions", "LOW", "Optional (training mode)"),
]


class DSMILGuidedActivation:
    """TUI-based guided activation system"""

    def __init__(self):
        """Initialize activation system"""
        self.stdscr = None
        self.current_group = 0
        self.current_device = 0
        self.running = True
        self.status_messages = []
        self.activation_log = []

        # Check for existing activation system
        try:
            # Add script directory to path for imports
            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            from dsmil_device_activation import DSMILDeviceActivator
            from dsmil_subsystem_controller import DSMILSubsystemController
            self.activator = DSMILDeviceActivator()
            self.controller = DSMILSubsystemController()
            self.activator_available = True
        except Exception as e:
            logger.warning(f"Activator not available: {e}")
            self.activator = None
            self.controller = None
            self.activator_available = False
            self.add_status(
                "DSMILDeviceActivator unavailable. Ensure kernel driver is built/loaded "
                "and dependencies are installed before attempting activation.",
                "ERROR"
            )

    def add_status(self, message: str, level: str = "INFO"):
        """Add status message"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_messages.append(f"[{timestamp}] {level}: {message}")
        if len(self.status_messages) > 50:
            self.status_messages = self.status_messages[-50:]
        logger.info(f"{level}: {message}")

    def draw_header(self):
        """Draw header"""
        height, width = self.stdscr.getmaxyx()
        header = "DSMIL GUIDED ACTIVATION SYSTEM - 84 DEVICES"
        self.stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD | curses.A_REVERSE)

        # Summary line
        safe_count = sum(1 for d in DSMIL_DEVICES if d.status == DeviceStatus.SAFE)
        quarantined_count = sum(1 for d in DSMIL_DEVICES if d.status == DeviceStatus.QUARANTINED)
        active_count = sum(1 for d in DSMIL_DEVICES if d.activation_state == ActivationState.ACTIVE)

        summary = f"Total: 84 | Safe: {safe_count} | Quarantined: {quarantined_count} | Active: {active_count}"
        self.stdscr.addstr(1, (width - len(summary)) // 2, summary)

    def draw_group_selector(self, start_y: int):
        """Draw group selector"""
        height, width = self.stdscr.getmaxyx()

        self.stdscr.addstr(start_y, 2, "GROUPS:", curses.A_BOLD)

        for i, (name, purpose, risk, deps) in enumerate(GROUP_INFO):
            y = start_y + 2 + i
            if i == self.current_group:
                attr = curses.A_REVERSE
            else:
                attr = curses.A_NORMAL

            # Count devices in group
            group_devs = [d for d in DSMIL_DEVICES if d.group == i]
            active = sum(1 for d in group_devs if d.activation_state == ActivationState.ACTIVE)

            line = f" {name}: {purpose} [{active}/{len(group_devs)} active]"
            self.stdscr.addstr(y, 4, line[:width-6], attr)

    def draw_device_list(self, start_y: int):
        """Draw device list for current group"""
        height, width = self.stdscr.getmaxyx()

        group_devices = [d for d in DSMIL_DEVICES if d.group == self.current_group]

        self.stdscr.addstr(start_y, 2, f"DEVICES IN GROUP {self.current_group}:", curses.A_BOLD)

        for i, dev in enumerate(group_devices):
            y = start_y + 2 + i
            if y >= height - 8:  # Leave room for controls and status
                break

            if i == self.current_device:
                attr = curses.A_REVERSE
            else:
                attr = curses.A_NORMAL

            # Status indicator
            if dev.status == DeviceStatus.QUARANTINED:
                status_sym = "🛑"
            elif dev.status == DeviceStatus.SAFE:
                status_sym = "✓"
            elif dev.activation_state == ActivationState.ACTIVE:
                status_sym = "●"
            else:
                status_sym = "○"

            line = f" {status_sym} {dev.get_hex_id()} {dev.name:<12} {dev.function[:30]}"

            try:
                self.stdscr.addstr(y, 4, line[:width-6], attr)
            except:
                pass

    def draw_device_details(self, start_y: int):
        """Draw details for selected device"""
        height, width = self.stdscr.getmaxyx()

        group_devices = [d for d in DSMIL_DEVICES if d.group == self.current_group]
        if self.current_device >= len(group_devices):
            return

        dev = group_devices[self.current_device]

        self.stdscr.addstr(start_y, 2, "DEVICE DETAILS:", curses.A_BOLD)

        details = [
            f"ID: {dev.get_hex_id()} ({dev.device_id})",
            f"Name: {dev.name} ({dev.get_acpi_name()})",
            f"Function: {dev.function}",
            f"Status: {dev.status.value.upper()}",
            f"State: {dev.activation_state.value.upper()}",
            f"Dependencies: {', '.join(hex(d) for d in dev.dependencies) if dev.dependencies else 'None'}",
        ]

        for i, detail in enumerate(details):
            y = start_y + 2 + i
            if y < height - 4:
                self.stdscr.addstr(y, 4, detail[:width-6])

    def draw_controls(self):
        """Draw control help"""
        height, width = self.stdscr.getmaxyx()
        y = height - 3

        controls = "↑↓: Navigate | ←→: Groups | ENTER: Activate | i: Info | s: Status | q: Quit"
        self.stdscr.addstr(y, 2, controls[:width-4], curses.A_DIM)

    def draw_status(self):
        """Draw status messages"""
        height, width = self.stdscr.getmaxyx()
        y = height - 2

        if self.status_messages:
            msg = self.status_messages[-1][:width-4]
            self.stdscr.addstr(y, 2, msg, curses.A_DIM)

    def activate_device(self, device: DSMILDevice):
        """Attempt to activate device"""
        if device.status == DeviceStatus.QUARANTINED:
            self.add_status(f"BLOCKED: {device.name} is QUARANTINED", "ERROR")
            device.activation_state = ActivationState.BLOCKED
            return False

        # Check dependencies
        for dep_id in device.dependencies:
            dep = DEVICE_MAP.get(dep_id)
            if dep and dep.activation_state != ActivationState.ACTIVE:
                self.add_status(f"ERROR: {device.name} requires {dep.name} to be active first", "ERROR")
                return False

        # Attempt activation
        self.add_status(f"Activating {device.name}...", "INFO")
        device.activation_state = ActivationState.ACTIVATING

        try:
            if self.activator_available and self.activator:
                # Use actual DSMILDeviceActivator with comprehensive safety checks
                result = self.activator.activate_device(device.device_id)

                if result.success:
                    device.activation_state = ActivationState.ACTIVE
                    device.last_test_time = time.time()
                    device.last_test_result = f"{result.method.value if result.method else 'unknown'}: {result.message}"
                    thermal_msg = f" (+{result.thermal_impact:.1f}°C)" if result.thermal_impact else ""
                    self.add_status(f"✓ {device.name} activated via {result.method.value if result.method else 'unknown'}{thermal_msg}", "SUCCESS")
                    return True
                else:
                    device.activation_state = ActivationState.FAILED if result.status.value == "failed" else ActivationState.BLOCKED
                    device.last_test_result = result.message
                    self.add_status(f"✗ {device.name} activation failed: {result.message}", "ERROR")
                    return False
            else:
                device.activation_state = ActivationState.BLOCKED
                device.last_test_time = time.time()
                device.last_test_result = "Activation backend unavailable"
                self.add_status(
                    "Activation backend not available. Load the DSMIL kernel driver "
                    "and rerun the build/install script before attempting activation.",
                    "ERROR"
                )
                return False

        except Exception as e:
            device.activation_state = ActivationState.FAILED
            device.last_test_result = str(e)
            self.add_status(f"✗ Exception activating {device.name}: {e}", "ERROR")
            return False

    def show_info_dialog(self, device: DSMILDevice):
        """Show detailed device info in dialog"""
        height, width = self.stdscr.getmaxyx()

        # Create dialog window
        dialog_h = min(20, height - 4)
        dialog_w = min(70, width - 4)
        dialog_y = (height - dialog_h) // 2
        dialog_x = (width - dialog_w) // 2

        dialog = curses.newwin(dialog_h, dialog_w, dialog_y, dialog_x)
        dialog.box()

        # Title
        title = f" {device.name} Info "
        dialog.addstr(0, (dialog_w - len(title)) // 2, title, curses.A_BOLD)

        # Content
        info = [
            ("Memory Token:", device.get_hex_id()),
            ("ACPI Name:", device.get_acpi_name()),
            ("Group:", f"{device.group} - {GROUP_INFO[device.group][1]}"),
            ("Function:", device.function),
            ("Safety Status:", device.status.value.upper()),
            ("Activation State:", device.activation_state.value.upper()),
            ("Dependencies:", ", ".join(DEVICE_MAP[d].name for d in device.dependencies) if device.dependencies else "None"),
            ("SMI Command Port:", "0x164E"),
            ("SMI Data Port:", "0x164F"),
        ]

        y = 2
        for label, value in info:
            if y < dialog_h - 2:
                dialog.addstr(y, 2, label, curses.A_BOLD)
                dialog.addstr(y, 25, str(value)[:dialog_w-27])
                y += 1

        if device.last_test_time:
            y += 1
            if y < dialog_h - 2:
                test_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(device.last_test_time))
                dialog.addstr(y, 2, "Last Test:", curses.A_BOLD)
                dialog.addstr(y, 25, test_time)
                y += 1
                if device.last_test_result and y < dialog_h - 2:
                    dialog.addstr(y, 2, "Result:", curses.A_BOLD)
                    dialog.addstr(y, 25, device.last_test_result[:dialog_w-27])

        dialog.addstr(dialog_h - 1, (dialog_w - 20) // 2, "Press any key to close", curses.A_DIM)

        dialog.refresh()
        dialog.getch()

    def run(self, stdscr):
        """Main TUI loop"""
        self.stdscr = stdscr
        curses.curs_set(0)
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)

        self.add_status("DSMIL Guided Activation System started")
        if not self.activator_available:
            self.add_status(
                "DSMILDeviceActivator not available - interface is READ-ONLY until the driver is loaded.",
                "ERROR"
            )
        else:
            available_methods = [m.value for m in self.activator.available_methods]
            self.add_status(f"Activator ready with methods: {', '.join(available_methods)}", "INFO")

        while self.running:
            self.stdscr.clear()
            height, width = self.stdscr.getmaxyx()

            # Draw UI
            self.draw_header()
            self.draw_group_selector(3)
            self.draw_device_list(12)
            self.draw_device_details(height - 15)
            self.draw_controls()
            self.draw_status()

            self.stdscr.refresh()

            # Handle input
            try:
                key = self.stdscr.getch()

                if key == ord('q') or key == ord('Q'):
                    self.running = False

                elif key == curses.KEY_UP:
                    self.current_device = max(0, self.current_device - 1)

                elif key == curses.KEY_DOWN:
                    group_devs = [d for d in DSMIL_DEVICES if d.group == self.current_group]
                    self.current_device = min(len(group_devs) - 1, self.current_device + 1)

                elif key == curses.KEY_LEFT:
                    self.current_group = max(0, self.current_group - 1)
                    self.current_device = 0

                elif key == curses.KEY_RIGHT:
                    self.current_group = min(6, self.current_group + 1)
                    self.current_device = 0

                elif key == ord('\n') or key == ord(' '):
                    # Activate selected device
                    group_devs = [d for d in DSMIL_DEVICES if d.group == self.current_group]
                    if self.current_device < len(group_devs):
                        self.activate_device(group_devs[self.current_device])

                elif key == ord('i') or key == ord('I'):
                    # Show info dialog
                    group_devs = [d for d in DSMIL_DEVICES if d.group == self.current_group]
                    if self.current_device < len(group_devs):
                        self.show_info_dialog(group_devs[self.current_device])

                elif key == ord('s') or key == ord('S'):
                    # Export status
                    self.export_status()
                    self.add_status("Status exported to /tmp/dsmil_activation_status.json")

            except KeyboardInterrupt:
                self.running = False

    def export_status(self):
        """Export current activation status to JSON"""
        # Use activator's comprehensive report if available
        if self.activator_available and self.activator:
            from pathlib import Path
            report_path = Path('/tmp/dsmil_activation_status.json')
            self.activator.generate_activation_report(output_path=report_path)
            logger.info(f"Comprehensive activation report generated: {report_path}")
        else:
            # Fallback to simple status export
            status = {
                "timestamp": time.time(),
                "mode": "READ_ONLY",
                "devices": [
                    {
                        "id": dev.get_hex_id(),
                        "name": dev.name,
                        "function": dev.function,
                        "group": dev.group,
                        "status": dev.status.value,
                        "activation_state": dev.activation_state.value,
                        "last_test_time": dev.last_test_time,
                        "last_test_result": dev.last_test_result,
                    }
                    for dev in DSMIL_DEVICES
                ],
                "summary": {
                    "total": len(DSMIL_DEVICES),
                    "safe": sum(1 for d in DSMIL_DEVICES if d.status == DeviceStatus.SAFE),
                    "quarantined": sum(1 for d in DSMIL_DEVICES if d.status == DeviceStatus.QUARANTINED),
                    "active": sum(1 for d in DSMIL_DEVICES if d.activation_state == ActivationState.ACTIVE),
                    "failed": sum(1 for d in DSMIL_DEVICES if d.activation_state == ActivationState.FAILED),
                    "blocked": sum(1 for d in DSMIL_DEVICES if d.activation_state == ActivationState.BLOCKED),
                }
            }

            with open('/tmp/dsmil_activation_status.json', 'w') as f:
                json.dump(status, f, indent=2)
                logger.info("Simulation mode status exported")


def main():
    """Main entry point"""
    print("Starting DSMIL Guided Activation System...")
    print("This will enumerate all 84 DSMIL devices and guide you through safe activation.")
    print()

    if os.geteuid() != 0:
        print("⚠️  WARNING: Not running as root. Some operations may fail.")
        print("   Run with: sudo python3 dsmil_guided_activation.py")
        print()

    input("Press ENTER to start TUI interface...")

    activation = DSMILGuidedActivation()

    try:
        curses.wrapper(activation.run)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nDSMIL Guided Activation System exited.")
    print(f"Activation log: /tmp/dsmil_guided_activation.log")
    print(f"Status export: /tmp/dsmil_activation_status.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
