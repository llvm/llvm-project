#!/usr/bin/env python3
"""
⚠️  DEPRECATED - This component will be removed in v3.0.0 (2026 Q3)
⚠️  Use: python3 dsmil.py control (then select Option 4 - System Monitoring)
⚠️  See DEPRECATION_PLAN.md for migration guide

DSMIL Operation Monitor & Controller
=====================================
Enhanced TUI for monitoring and executing operations on all DSMIL devices.

Features:
- Browse all 656 operations across 80 devices
- View operation signatures and parameters
- Execute operations with real-time monitoring
- Track operation history and results
- Robust error handling with detailed logs

Memory Map: 0x8000-0x806B (84 devices in 7 groups)
Operation Database: DSMIL_DEVICE_CAPABILITIES.json

Author: LAT5150DRVMIL AI Platform
Classification: DSMIL Operation Management
"""

import curses
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Setup logging with rotation
logging.basicConfig(
    filename='/tmp/dsmil_operation_monitor.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DEPRECATION WARNING - REMOVE IN v3.0.0 (2026 Q3)
# ============================================================================
import warnings
warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATED: dsmil_operation_monitor.py\n\n"
    "This component is deprecated and will be removed in v3.0.0 (2026 Q3).\n\n"
    "Migration:\n"
    "  - Use: python3 dsmil.py control\n"
    "  - Then select: Option 4 - System Monitoring\n\n"
    "See DEPRECATION_PLAN.md for complete migration guide.\n"
    + "=" * 80,
    DeprecationWarning,
    stacklevel=2
)
# ============================================================================


class ViewMode(Enum):
    """TUI view modes"""
    DEVICE_LIST = "device_list"
    OPERATION_LIST = "operation_list"
    OPERATION_DETAIL = "operation_detail"
    EXECUTION_LOG = "execution_log"


class DeviceStatus(Enum):
    """Device safety status"""
    SAFE = "safe"
    MONITORED = "monitored"
    QUARANTINED = "quarantined"
    UNKNOWN = "unknown"


class OperationStatus(Enum):
    """Operation execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DeviceOperation:
    """Device operation metadata"""
    name: str
    args: List[Dict[str, str]]
    return_type: Optional[str]
    docstring: Optional[str]
    last_execution: Optional[datetime] = None
    last_status: OperationStatus = OperationStatus.PENDING
    last_result: Optional[str] = None
    execution_count: int = 0

    def get_signature(self) -> str:
        """Get operation signature string"""
        arg_str = ", ".join([f"{a.get('name', '?')}: {a.get('type', '?')}" for a in self.args])
        ret_str = f" -> {self.return_type}" if self.return_type else ""
        return f"{self.name}({arg_str}){ret_str}"

    def get_short_signature(self) -> str:
        """Get short signature for list view"""
        arg_str = ", ".join([a.get('name', '?') for a in self.args])
        return f"{self.name}({arg_str})"


@dataclass
class DeviceInfo:
    """DSMIL device with operations"""
    device_id: str  # Hex ID like "0x8000"
    name: str
    file: str
    description: str
    group: str
    risk_level: str
    operations: List[DeviceOperation] = field(default_factory=list)
    registers: List[Dict] = field(default_factory=list)
    constants: List[Dict] = field(default_factory=list)
    total_operations: int = 0
    total_registers: int = 0

    def get_decimal_id(self) -> int:
        """Get device ID as decimal"""
        return int(self.device_id, 16)

    def get_risk_icon(self) -> str:
        """Get risk level icon"""
        if "QUARANTINED" in self.risk_level:
            return "🔴"
        elif "MONITORED" in self.risk_level:
            return "🟡"
        else:
            return "🟢"


@dataclass
class ExecutionRecord:
    """Record of operation execution"""
    timestamp: datetime
    device_id: str
    operation: str
    status: OperationStatus
    duration_ms: float
    result: Optional[str]
    error: Optional[str]


class DSMILOperationMonitor:
    """Main TUI application for DSMIL operation monitoring"""

    def __init__(self):
        self.devices: List[DeviceInfo] = []
        self.execution_history: List[ExecutionRecord] = []
        self.view_mode = ViewMode.DEVICE_LIST
        self.selected_device_idx = 0
        self.selected_operation_idx = 0
        self.device_scroll = 0
        self.operation_scroll = 0
        self.history_scroll = 0
        self.status_messages: List[Tuple[str, str]] = []  # (message, level)
        self.max_status_messages = 10

        # Try to load actual device implementations
        self.device_implementations = {}
        self.load_device_implementations()

        # Load capabilities
        self.load_capabilities()

    def load_capabilities(self):
        """Load device capabilities from JSON"""
        try:
            # First try relative to script location
            cap_file = Path(__file__).parent.parent / "DSMIL_DEVICE_CAPABILITIES.json"
            if not cap_file.exists():
                # Try current working directory
                cap_file = Path.cwd() / "DSMIL_DEVICE_CAPABILITIES.json"
            if not cap_file.exists():
                # Try common alternate locations
                for alt_path in ["/home/user/LAT5150DRVMIL", "/home/john/Documents/LAT5150DRVMIL"]:
                    alt_file = Path(alt_path) / "DSMIL_DEVICE_CAPABILITIES.json"
                    if alt_file.exists():
                        cap_file = alt_file
                        break

            if not cap_file.exists():
                self.add_status("WARNING: Capabilities file not found", "WARNING")
                logger.warning(f"Capabilities file not found at {cap_file}")
                return

            with open(cap_file, 'r') as f:
                data = json.load(f)

            # Parse devices
            for dev_id, dev_data in data.get('devices', {}).items():
                operations = []
                for op_data in dev_data.get('public_methods', []):
                    op = DeviceOperation(
                        name=op_data.get('name', 'unknown'),
                        args=op_data.get('args', []),
                        return_type=op_data.get('return_type'),
                        docstring=op_data.get('docstring')
                    )
                    operations.append(op)

                device = DeviceInfo(
                    device_id=dev_id,
                    name=dev_data.get('name', 'Unknown'),
                    file=dev_data.get('file', ''),
                    description=dev_data.get('description', ''),
                    group=dev_data.get('group', 'Unknown'),
                    risk_level=dev_data.get('risk_level', 'UNKNOWN'),
                    operations=operations,
                    registers=dev_data.get('registers', []),
                    constants=dev_data.get('constants', []),
                    total_operations=len(operations),
                    total_registers=len(dev_data.get('registers', []))
                )
                self.devices.append(device)

            # Sort by device ID
            self.devices.sort(key=lambda d: d.get_decimal_id())

            self.add_status(f"Loaded {len(self.devices)} devices with {sum(d.total_operations for d in self.devices)} operations", "SUCCESS")
            logger.info(f"Loaded capabilities for {len(self.devices)} devices")

        except Exception as e:
            self.add_status(f"ERROR loading capabilities: {str(e)}", "ERROR")
            logger.error(f"Failed to load capabilities: {e}", exc_info=True)

    def load_device_implementations(self):
        """Try to load actual device implementation classes"""
        try:
            # Try to find device implementations relative to script or common locations
            potential_paths = [
                Path(__file__).parent.parent / "02-tools/dsmil-devices/devices",
                Path.cwd() / "02-tools/dsmil-devices/devices",
                Path("/home/user/LAT5150DRVMIL/02-tools/dsmil-devices/devices"),
                Path("/home/john/Documents/LAT5150DRVMIL/02-tools/dsmil-devices/devices"),
            ]

            for device_path in potential_paths:
                if device_path.exists():
                    sys.path.insert(0, str(device_path))
                    self.add_status(f"Device implementations path: {device_path}", "INFO")
                    logger.info(f"Added device path: {device_path}")
                    break
        except Exception as e:
            logger.warning(f"Could not load device implementations: {e}")

    def add_status(self, message: str, level: str = "INFO"):
        """Add status message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_messages.append((f"[{timestamp}] {message}", level))
        if len(self.status_messages) > self.max_status_messages:
            self.status_messages.pop(0)
        logger.info(f"{level}: {message}")

    def get_selected_device(self) -> Optional[DeviceInfo]:
        """Get currently selected device"""
        if 0 <= self.selected_device_idx < len(self.devices):
            return self.devices[self.selected_device_idx]
        return None

    def get_selected_operation(self) -> Optional[DeviceOperation]:
        """Get currently selected operation"""
        device = self.get_selected_device()
        if device and 0 <= self.selected_operation_idx < len(device.operations):
            return device.operations[self.selected_operation_idx]
        return None

    def execute_operation(self, device: DeviceInfo, operation: DeviceOperation) -> ExecutionRecord:
        """Execute a device operation using the installed device implementation"""
        start_time = time.time()

        try:
            # Check if device is quarantined
            if "QUARANTINED" in device.risk_level:
                raise Exception(f"Device {device.device_id} is QUARANTINED - operation blocked")

            # Try to import and instantiate device
            if device.file:
                module_name = device.file.replace('.py', '')
                try:
                    # Try to import the module
                    import importlib
                    mod = importlib.import_module(module_name)

                    # Try to find device class
                    device_class = getattr(mod, device.name, None)
                    if device_class:
                        instance = device_class()
                        method = getattr(instance, operation.name, None)

                        if method:
                            # For now, only execute safe read operations
                            if operation.name in ['get_status', 'get_capabilities', 'initialize']:
                                result = method()
                                operation.last_execution = datetime.now()
                                operation.last_status = OperationStatus.SUCCESS
                                operation.execution_count += 1

                                duration = (time.time() - start_time) * 1000

                                record = ExecutionRecord(
                                    timestamp=datetime.now(),
                                    device_id=device.device_id,
                                    operation=operation.name,
                                    status=OperationStatus.SUCCESS,
                                    duration_ms=duration,
                                    result=str(result)[:200],
                                    error=None
                                )

                                self.add_status(f"✓ {device.name}.{operation.name}() executed in {duration:.1f}ms", "SUCCESS")
                                return record
                            else:
                                # Skip non-safe operations for now
                                operation.last_status = OperationStatus.SKIPPED
                                raise Exception(f"Operation {operation.name} requires parameters or is not safe for auto-execution")
                        else:
                            raise Exception(f"Method {operation.name} not found in {device.name}")
                    else:
                        raise Exception(f"Device class {device.name} not found in {module_name}")

                except ImportError as ie:
                    raise Exception(f"Could not import {module_name}: {ie}")
            else:
                raise Exception("No implementation file specified")

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            operation.last_execution = datetime.now()
            operation.last_status = OperationStatus.FAILED
            operation.last_result = str(e)

            record = ExecutionRecord(
                timestamp=datetime.now(),
                device_id=device.device_id,
                operation=operation.name,
                status=OperationStatus.FAILED,
                duration_ms=duration,
                result=None,
                error=str(e)
            )

            self.add_status(f"✗ {device.name}.{operation.name}(): {str(e)[:80]}", "ERROR")
            logger.error(f"Operation failed: {device.device_id}.{operation.name}: {e}")
            return record

    def run(self, stdscr):
        """Main TUI loop"""
        curses.curs_set(0)  # Hide cursor
        curses.use_default_colors()

        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # Success
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Warning
        curses.init_pair(3, curses.COLOR_RED, -1)     # Error
        curses.init_pair(4, curses.COLOR_CYAN, -1)    # Info
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected

        while True:
            try:
                stdscr.clear()
                height, width = stdscr.getmaxyx()

                # Draw based on current view
                if self.view_mode == ViewMode.DEVICE_LIST:
                    self.draw_device_list(stdscr, height, width)
                elif self.view_mode == ViewMode.OPERATION_LIST:
                    self.draw_operation_list(stdscr, height, width)
                elif self.view_mode == ViewMode.OPERATION_DETAIL:
                    self.draw_operation_detail(stdscr, height, width)
                elif self.view_mode == ViewMode.EXECUTION_LOG:
                    self.draw_execution_log(stdscr, height, width)

                # Draw status bar at bottom
                self.draw_status_bar(stdscr, height, width)

                stdscr.refresh()

                # Handle input
                key = stdscr.getch()
                if not self.handle_input(key):
                    break

            except Exception as e:
                logger.error(f"TUI error: {e}", exc_info=True)
                self.add_status(f"TUI ERROR: {str(e)}", "ERROR")
                time.sleep(0.5)  # Prevent error spam

    def draw_device_list(self, stdscr, height, width):
        """Draw device list view"""
        # Header
        stdscr.addstr(0, 0, "═" * width)
        title = f" DSMIL OPERATION MONITOR - {len(self.devices)} Devices "
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

        stdscr.addstr(1, 2, f"Total Operations: {sum(d.total_operations for d in self.devices)}", curses.color_pair(4))
        stdscr.addstr(1, 30, f"Total Registers: {sum(d.total_registers for d in self.devices)}", curses.color_pair(4))
        stdscr.addstr(2, 0, "─" * width)

        # Column headers
        stdscr.addstr(3, 2, "ID", curses.A_BOLD)
        stdscr.addstr(3, 12, "Name", curses.A_BOLD)
        stdscr.addstr(3, 45, "Ops", curses.A_BOLD)
        stdscr.addstr(3, 52, "Regs", curses.A_BOLD)
        stdscr.addstr(3, 60, "Group", curses.A_BOLD)

        # Device list
        list_start = 4
        list_height = height - list_start - 8

        for i in range(self.device_scroll, min(self.device_scroll + list_height, len(self.devices))):
            device = self.devices[i]
            y = list_start + (i - self.device_scroll)

            # Highlight selected
            attr = curses.color_pair(5) if i == self.selected_device_idx else 0

            # Risk icon
            icon = device.get_risk_icon()
            stdscr.addstr(y, 0, icon)

            # Device info
            stdscr.addstr(y, 2, device.device_id[:8], attr)
            stdscr.addstr(y, 12, device.name[:30], attr)
            stdscr.addstr(y, 45, f"{device.total_operations:3}", attr)
            stdscr.addstr(y, 52, f"{device.total_registers:3}", attr)
            stdscr.addstr(y, 60, device.group[:15], attr)

        # Help
        help_y = height - 6
        stdscr.addstr(help_y, 0, "─" * width)
        stdscr.addstr(help_y + 1, 2, "↑/↓: Navigate  ENTER: View Operations  L: Execution Log  Q: Quit", curses.color_pair(4))

    def draw_operation_list(self, stdscr, height, width):
        """Draw operation list for selected device"""
        device = self.get_selected_device()
        if not device:
            return

        # Header
        stdscr.addstr(0, 0, "═" * width)
        title = f" {device.name} ({device.device_id}) - {device.total_operations} Operations "
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

        stdscr.addstr(1, 2, device.group, curses.color_pair(4))
        stdscr.addstr(1, 35, f"Risk: {device.risk_level}", curses.color_pair(4))
        stdscr.addstr(2, 0, "─" * width)

        # Column headers
        stdscr.addstr(3, 2, "#", curses.A_BOLD)
        stdscr.addstr(3, 6, "Operation", curses.A_BOLD)
        stdscr.addstr(3, 45, "Executions", curses.A_BOLD)
        stdscr.addstr(3, 60, "Status", curses.A_BOLD)

        # Operation list
        list_start = 4
        list_height = height - list_start - 8

        for i in range(self.operation_scroll, min(self.operation_scroll + list_height, len(device.operations))):
            op = device.operations[i]
            y = list_start + (i - self.operation_scroll)

            # Highlight selected
            attr = curses.color_pair(5) if i == self.selected_operation_idx else 0

            stdscr.addstr(y, 2, f"{i+1:3}", attr)
            stdscr.addstr(y, 6, op.get_short_signature()[:37], attr)
            stdscr.addstr(y, 45, f"{op.execution_count:5}", attr)

            # Status icon
            if op.last_status == OperationStatus.SUCCESS:
                stdscr.addstr(y, 60, "✓", curses.color_pair(1))
            elif op.last_status == OperationStatus.FAILED:
                stdscr.addstr(y, 60, "✗", curses.color_pair(3))
            elif op.last_status == OperationStatus.SKIPPED:
                stdscr.addstr(y, 60, "⊘", curses.color_pair(2))

        # Help
        help_y = height - 6
        stdscr.addstr(help_y, 0, "─" * width)
        stdscr.addstr(help_y + 1, 2, "↑/↓: Navigate  ENTER: Details  E: Execute  ESC: Back  Q: Quit", curses.color_pair(4))

    def draw_operation_detail(self, stdscr, height, width):
        """Draw detailed operation view"""
        device = self.get_selected_device()
        op = self.get_selected_operation()
        if not device or not op:
            return

        # Header
        stdscr.addstr(0, 0, "═" * width)
        title = f" Operation: {op.name} "
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

        y = 2
        stdscr.addstr(y, 2, f"Device: {device.name} ({device.device_id})", curses.color_pair(4))
        y += 1

        stdscr.addstr(y, 2, f"Signature: {op.get_signature()}", curses.A_BOLD)
        y += 2

        if op.docstring:
            stdscr.addstr(y, 2, f"Description: {op.docstring[:width-4]}")
            y += 2

        stdscr.addstr(y, 2, f"Executions: {op.execution_count}")
        y += 1

        if op.last_execution:
            stdscr.addstr(y, 2, f"Last Run: {op.last_execution.strftime('%Y-%m-%d %H:%M:%S')}")
            y += 1

            status_color = curses.color_pair(1 if op.last_status == OperationStatus.SUCCESS else 3)
            stdscr.addstr(y, 2, f"Status: {op.last_status.value}", status_color)
            y += 1

            if op.last_result:
                stdscr.addstr(y, 2, f"Result: {op.last_result[:width-4]}")
                y += 1

        # Parameters
        if op.args:
            y += 1
            stdscr.addstr(y, 2, "Parameters:", curses.A_BOLD)
            y += 1
            for arg in op.args:
                arg_name = arg.get('name', '?')
                arg_type = arg.get('type', '?')
                stdscr.addstr(y, 4, f"• {arg_name}: {arg_type}")
                y += 1

        # Help
        help_y = height - 6
        stdscr.addstr(help_y, 0, "─" * width)
        stdscr.addstr(help_y + 1, 2, "E: Execute  ESC: Back  Q: Quit", curses.color_pair(4))

    def draw_execution_log(self, stdscr, height, width):
        """Draw execution history"""
        stdscr.addstr(0, 0, "═" * width)
        title = f" Execution Log - {len(self.execution_history)} Records "
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

        stdscr.addstr(2, 0, "─" * width)
        stdscr.addstr(3, 2, "Time", curses.A_BOLD)
        stdscr.addstr(3, 12, "Device", curses.A_BOLD)
        stdscr.addstr(3, 30, "Operation", curses.A_BOLD)
        stdscr.addstr(3, 50, "Duration", curses.A_BOLD)
        stdscr.addstr(3, 62, "Status", curses.A_BOLD)

        list_start = 4
        list_height = height - list_start - 8

        # Show most recent first
        for i in range(len(self.execution_history) - 1, max(-1, len(self.execution_history) - list_height - 1), -1):
            record = self.execution_history[i]
            y = list_start + (len(self.execution_history) - 1 - i)

            if y >= list_start + list_height:
                break

            stdscr.addstr(y, 2, record.timestamp.strftime("%H:%M:%S"))
            stdscr.addstr(y, 12, record.device_id[:12])
            stdscr.addstr(y, 30, record.operation[:18])
            stdscr.addstr(y, 50, f"{record.duration_ms:.1f}ms")

            status_color = curses.color_pair(1 if record.status == OperationStatus.SUCCESS else 3)
            stdscr.addstr(y, 62, record.status.value[:10], status_color)

        # Help
        help_y = height - 6
        stdscr.addstr(help_y, 0, "─" * width)
        stdscr.addstr(help_y + 1, 2, "ESC: Back  Q: Quit", curses.color_pair(4))

    def draw_status_bar(self, stdscr, height, width):
        """Draw status messages at bottom"""
        status_start = height - 4
        stdscr.addstr(status_start, 0, "─" * width)

        # Show recent status messages
        for i, (msg, level) in enumerate(self.status_messages[-3:]):
            y = status_start + 1 + i
            if level == "SUCCESS":
                color = curses.color_pair(1)
            elif level == "WARNING":
                color = curses.color_pair(2)
            elif level == "ERROR":
                color = curses.color_pair(3)
            else:
                color = curses.color_pair(4)

            stdscr.addstr(y, 2, msg[:width-4], color)

    def handle_input(self, key):
        """Handle keyboard input"""
        try:
            # Global keys
            if key in [ord('q'), ord('Q')]:
                return False

            # Mode-specific navigation
            if self.view_mode == ViewMode.DEVICE_LIST:
                if key == curses.KEY_UP:
                    if self.selected_device_idx > 0:
                        self.selected_device_idx -= 1
                        if self.selected_device_idx < self.device_scroll:
                            self.device_scroll -= 1
                elif key == curses.KEY_DOWN:
                    if self.selected_device_idx < len(self.devices) - 1:
                        self.selected_device_idx += 1
                        if self.selected_device_idx >= self.device_scroll + 20:
                            self.device_scroll += 1
                elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
                    self.view_mode = ViewMode.OPERATION_LIST
                    self.selected_operation_idx = 0
                    self.operation_scroll = 0
                elif key in [ord('l'), ord('L')]:
                    self.view_mode = ViewMode.EXECUTION_LOG

            elif self.view_mode == ViewMode.OPERATION_LIST:
                device = self.get_selected_device()
                if device:
                    if key == curses.KEY_UP:
                        if self.selected_operation_idx > 0:
                            self.selected_operation_idx -= 1
                            if self.selected_operation_idx < self.operation_scroll:
                                self.operation_scroll -= 1
                    elif key == curses.KEY_DOWN:
                        if self.selected_operation_idx < len(device.operations) - 1:
                            self.selected_operation_idx += 1
                            if self.selected_operation_idx >= self.operation_scroll + 20:
                                self.operation_scroll += 1
                    elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
                        self.view_mode = ViewMode.OPERATION_DETAIL
                    elif key in [ord('e'), ord('E')]:
                        op = self.get_selected_operation()
                        if op:
                            record = self.execute_operation(device, op)
                            self.execution_history.append(record)
                    elif key == 27:  # ESC
                        self.view_mode = ViewMode.DEVICE_LIST

            elif self.view_mode == ViewMode.OPERATION_DETAIL:
                if key in [ord('e'), ord('E')]:
                    device = self.get_selected_device()
                    op = self.get_selected_operation()
                    if device and op:
                        record = self.execute_operation(device, op)
                        self.execution_history.append(record)
                elif key == 27:  # ESC
                    self.view_mode = ViewMode.OPERATION_LIST

            elif self.view_mode == ViewMode.EXECUTION_LOG:
                if key == 27:  # ESC
                    self.view_mode = ViewMode.DEVICE_LIST

            return True

        except Exception as e:
            logger.error(f"Input handling error: {e}", exc_info=True)
            self.add_status(f"Input error: {str(e)}", "ERROR")
            return True


def main():
    """Main entry point"""
    try:
        monitor = DSMILOperationMonitor()

        if len(monitor.devices) == 0:
            print("ERROR: No devices loaded. Check DSMIL_DEVICE_CAPABILITIES.json")
            logger.error("No devices loaded")
            return 1

        # Run TUI
        curses.wrapper(monitor.run)

        # Export execution log on exit
        if monitor.execution_history:
            log_file = Path("/tmp/dsmil_execution_history.json")
            with open(log_file, 'w') as f:
                json.dump([{
                    'timestamp': r.timestamp.isoformat(),
                    'device_id': r.device_id,
                    'operation': r.operation,
                    'status': r.status.value,
                    'duration_ms': r.duration_ms,
                    'result': r.result,
                    'error': r.error
                } for r in monitor.execution_history], f, indent=2)
            print(f"\nExecution history saved to: {log_file}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
