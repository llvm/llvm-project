#!/usr/bin/env python3
"""
DSMIL Real-Time Monitoring Dashboard
Dell Latitude 5450 MIL-SPEC - Advanced Visualization System

COMPREHENSIVE DASHBOARD FEATURES:
- Real-time monitoring of all 84 DSMIL devices
- Multi-panel layout with device groups, system health, and security alerts
- Pattern recognition for anomaly detection
- Interactive terminal-based interface
- Emergency stop integration
- Historical trend analysis

Author: MONITOR Agent
Date: 2025-09-01
Classification: MIL-SPEC Advanced Monitoring
"""

import os
import sys
import time
import json
import threading
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import argparse
import curses
import signal
from collections import deque

# Import our monitoring components
try:
    from dsmil_readonly_monitor import DSMILReadOnlyMonitor, DSMILDevice, SystemHealth, SecurityEvent, AlertLevel
    from dsmil_emergency_stop import DSMILEmergencyStop
except ImportError:
    print("❌ Error: Required monitoring modules not found")
    print("   Ensure dsmil_readonly_monitor.py and dsmil_emergency_stop.py are in the same directory")
    sys.exit(1)

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

# Display refresh rates
DASHBOARD_REFRESH_RATE = 1.0    # seconds
DEVICE_REFRESH_RATE = 0.5       # seconds  
ALERT_REFRESH_RATE = 0.2        # seconds

# Color definitions for curses
COLOR_NORMAL = 1
COLOR_WARNING = 2
COLOR_CRITICAL = 3
COLOR_EMERGENCY = 4
COLOR_SAFE = 5
COLOR_TITLE = 6

# Dashboard layout dimensions
HEADER_HEIGHT = 3
STATUS_PANEL_WIDTH = 40
DEVICE_PANEL_WIDTH = 80
ALERT_PANEL_HEIGHT = 10

class DashboardMode:
    """Dashboard display modes"""
    OVERVIEW = "overview"           # High-level system status
    DEVICES = "devices"             # Detailed device monitoring
    SECURITY = "security"           # Security-focused view
    THERMAL = "thermal"             # Thermal monitoring
    NETWORK = "network"             # System resource monitoring

@dataclass
class DashboardState:
    """Current dashboard state"""
    mode: str
    start_time: datetime
    update_count: int
    emergency_active: bool
    devices_monitored: int
    alerts_count: int
    last_update: datetime

class DSMILDashboard:
    """
    Advanced real-time dashboard for DSMIL monitoring
    
    Provides comprehensive visualization of:
    - All 84 DSMIL device statuses
    - System health metrics
    - Security alerts and anomalies
    - Historical trends
    - Emergency stop controls
    """
    
    def __init__(self):
        self.dashboard_state = DashboardState(
            mode=DashboardMode.OVERVIEW,
            start_time=datetime.now(),
            update_count=0,
            emergency_active=False,
            devices_monitored=0,
            alerts_count=0,
            last_update=datetime.now()
        )
        
        # Initialize monitoring components
        self.monitor = None
        self.emergency_stop = DSMILEmergencyStop()
        
        # Data storage for trends
        self.device_history = {}  # token_id -> deque of historical values
        self.system_history = deque(maxlen=300)  # 5 minutes at 1s intervals
        self.alert_history = deque(maxlen=100)   # Last 100 alerts
        
        # Threading control
        self.running = False
        self.monitor_thread = None
        self.data_collection_thread = None
        
        # Curses screen management
        self.screen = None
        self.panels = {}
        
        # Signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.running = False
        if self.screen:
            curses.endwin()
        print(f"\n🛑 Dashboard shutdown by signal {signum}")
        sys.exit(0)
    
    def init_curses(self):
        """Initialize curses for terminal display"""
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)  # Hide cursor
        self.screen.keypad(True)
        self.screen.nodelay(True)  # Non-blocking input
        
        # Initialize colors if supported
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(COLOR_NORMAL, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(COLOR_WARNING, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(COLOR_CRITICAL, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(COLOR_EMERGENCY, curses.COLOR_WHITE, curses.COLOR_RED)
            curses.init_pair(COLOR_SAFE, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(COLOR_TITLE, curses.COLOR_CYAN, curses.COLOR_BLACK)
    
    def cleanup_curses(self):
        """Clean up curses"""
        if self.screen:
            curses.endwin()
    
    def start_monitoring(self):
        """Start background monitoring threads"""
        try:
            from dsmil_readonly_monitor import MonitoringMode
            
            # Initialize monitor but don't start its display
            self.monitor = DSMILReadOnlyMonitor(MonitoringMode.DASHBOARD)
            self.monitor.initialize_devices()
            self.monitor.running = True
            
            # Start background monitoring thread
            self.monitor_thread = threading.Thread(
                target=self.monitor.run_monitoring_cycle, 
                daemon=True
            )
            self.monitor_thread.start()
            
            # Start data collection thread
            self.data_collection_thread = threading.Thread(
                target=self.collect_data, 
                daemon=True
            )
            self.data_collection_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            return False
    
    def collect_data(self):
        """Background thread for collecting monitoring data"""
        while self.running:
            try:
                if self.monitor:
                    # Collect system health
                    health = self.monitor.check_system_health()
                    if health:
                        self.system_history.append({
                            'timestamp': datetime.now(),
                            'cpu_temp': health.cpu_temp,
                            'cpu_usage': health.cpu_usage,
                            'memory_usage': health.memory_usage,
                            'processes': health.processes_count
                        })
                    
                    # Collect device states
                    for token_id, device in self.monitor.devices.items():
                        if token_id not in self.device_history:
                            self.device_history[token_id] = deque(maxlen=100)
                        
                        self.device_history[token_id].append({
                            'timestamp': datetime.now(),
                            'status': device.current_status,
                            'changed': device.current_status != device.previous_status,
                            'anomaly_score': device.anomaly_score
                        })
                    
                    # Collect recent alerts
                    if hasattr(self.monitor, 'security_events'):
                        recent_events = self.monitor.security_events[-10:]
                        for event in recent_events:
                            if event not in self.alert_history:
                                self.alert_history.append(event)
                
                time.sleep(1.0)  # Collect data every second
                
            except Exception as e:
                # Log error but continue
                time.sleep(1.0)
    
    def draw_header(self, y: int, x: int, width: int):
        """Draw dashboard header"""
        if not self.screen:
            return
        
        try:
            # Title line
            title = "DSMIL 84-Device Real-Time Monitoring Dashboard"
            self.screen.addstr(y, x + (width - len(title)) // 2, title, 
                             curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
            
            # Status line
            runtime = datetime.now() - self.dashboard_state.start_time
            status_info = f"Mode: {self.dashboard_state.mode.upper()} | Runtime: {str(runtime).split('.')[0]} | Updates: {self.dashboard_state.update_count}"
            self.screen.addstr(y + 1, x, status_info[:width-1])
            
            # Safety status
            emergency_status = "🚨 EMERGENCY ACTIVE" if self.dashboard_state.emergency_active else "✅ SAFE"
            safety_info = f"Safety: {emergency_status} | Devices: {self.dashboard_state.devices_monitored} | Alerts: {self.dashboard_state.alerts_count}"
            self.screen.addstr(y + 2, x, safety_info[:width-1])
            
        except curses.error:
            pass  # Ignore curses errors
    
    def draw_system_status(self, y: int, x: int, width: int, height: int):
        """Draw system status panel"""
        if not self.screen or not self.system_history:
            return
        
        try:
            # Panel title
            self.screen.addstr(y, x, "SYSTEM STATUS", 
                             curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
            
            # Latest system metrics
            latest = self.system_history[-1]
            
            # Temperature status
            temp = latest['cpu_temp']
            temp_color = COLOR_SAFE if temp < 70 else (COLOR_WARNING if temp < 85 else COLOR_CRITICAL)
            self.screen.addstr(y + 2, x, f"CPU Temp: {temp:.1f}°C", curses.color_pair(temp_color))
            
            # CPU usage
            cpu = latest['cpu_usage']
            cpu_color = COLOR_SAFE if cpu < 80 else (COLOR_WARNING if cpu < 90 else COLOR_CRITICAL)
            self.screen.addstr(y + 3, x, f"CPU Usage: {cpu:.1f}%", curses.color_pair(cpu_color))
            
            # Memory usage
            memory = latest['memory_usage']
            mem_color = COLOR_SAFE if memory < 80 else (COLOR_WARNING if memory < 90 else COLOR_CRITICAL)
            self.screen.addstr(y + 4, x, f"Memory: {memory:.1f}%", curses.color_pair(mem_color))
            
            # Process count
            proc_count = latest['processes']
            self.screen.addstr(y + 5, x, f"Processes: {proc_count}")
            
            # Trends (last 30 seconds)
            if len(self.system_history) > 30:
                recent = list(self.system_history)[-30:]
                
                # Temperature trend
                temp_trend = recent[-1]['cpu_temp'] - recent[0]['cpu_temp']
                trend_arrow = "↑" if temp_trend > 2 else ("↓" if temp_trend < -2 else "→")
                self.screen.addstr(y + 7, x, f"Temp Trend: {trend_arrow} {temp_trend:+.1f}°C")
                
                # CPU trend
                cpu_trend = recent[-1]['cpu_usage'] - recent[0]['cpu_usage']
                cpu_arrow = "↑" if cpu_trend > 5 else ("↓" if cpu_trend < -5 else "→")
                self.screen.addstr(y + 8, x, f"CPU Trend: {cpu_arrow} {cpu_trend:+.1f}%")
            
        except curses.error:
            pass
    
    def draw_device_groups(self, y: int, x: int, width: int, height: int):
        """Draw DSMIL device group status"""
        if not self.screen or not self.monitor or not self.monitor.devices:
            return
        
        try:
            # Panel title
            self.screen.addstr(y, x, "DSMIL DEVICE GROUPS (84 devices)", 
                             curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
            
            # Group device counts and statuses
            from dsmil_readonly_monitor import DSMILDeviceGroup
            
            groups = {
                DSMILDeviceGroup.GROUP_0: "Core/Security",
                DSMILDeviceGroup.GROUP_1: "Thermal Mgmt", 
                DSMILDeviceGroup.GROUP_2: "Communication",
                DSMILDeviceGroup.GROUP_3: "Sensors",
                DSMILDeviceGroup.GROUP_4: "Crypto/Keys",
                DSMILDeviceGroup.GROUP_5: "Storage Ctrl",
                DSMILDeviceGroup.GROUP_6: "Extended"
            }
            
            row = y + 2
            for group, name in groups.items():
                if row >= y + height - 1:
                    break
                
                # Count devices in this group
                group_devices = [d for d in self.monitor.devices.values() if d.group == group]
                active_count = sum(1 for d in group_devices if d.current_status != 0)
                changed_count = sum(1 for d in group_devices if d.change_count > 0)
                dangerous_count = sum(1 for d in group_devices if d.is_dangerous)
                
                # Status indicators
                status_color = COLOR_CRITICAL if dangerous_count > 0 and active_count > 0 else (
                    COLOR_WARNING if active_count > 0 else COLOR_SAFE
                )
                
                danger_indicator = "🚨" if dangerous_count > 0 else "  "
                active_indicator = f"{active_count:2d}A" if active_count > 0 else "   "
                changed_indicator = f"{changed_count:2d}C" if changed_count > 0 else "   "
                
                group_info = f"{danger_indicator} {name:<12} {active_indicator} {changed_indicator} ({len(group_devices):2d} total)"
                
                try:
                    self.screen.addstr(row, x, group_info[:width-1], curses.color_pair(status_color))
                except curses.error:
                    pass
                
                row += 1
            
            # Summary statistics
            if row < y + height - 2:
                total_devices = len(self.monitor.devices)
                total_active = sum(1 for d in self.monitor.devices.values() if d.current_status != 0)
                total_changed = sum(1 for d in self.monitor.devices.values() if d.change_count > 0)
                total_dangerous = sum(1 for d in self.monitor.devices.values() if d.is_dangerous)
                
                summary = f"TOTAL: {total_devices} devices | {total_active} active | {total_changed} changed | {total_dangerous} dangerous"
                try:
                    self.screen.addstr(row + 1, x, summary[:width-1], curses.color_pair(COLOR_TITLE))
                except curses.error:
                    pass
            
        except curses.error:
            pass
    
    def draw_recent_alerts(self, y: int, x: int, width: int, height: int):
        """Draw recent security alerts"""
        if not self.screen:
            return
        
        try:
            # Panel title
            self.screen.addstr(y, x, "RECENT SECURITY ALERTS", 
                             curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
            
            if not self.alert_history:
                self.screen.addstr(y + 2, x, "No alerts", curses.color_pair(COLOR_SAFE))
                return
            
            # Show recent alerts
            row = y + 2
            recent_alerts = list(self.alert_history)[-min(height-3, len(self.alert_history)):]
            
            for alert in reversed(recent_alerts):  # Most recent first
                if row >= y + height - 1:
                    break
                
                # Format alert
                time_str = alert.timestamp.strftime("%H:%M:%S")
                severity_colors = {
                    AlertLevel.INFO: COLOR_NORMAL,
                    AlertLevel.WARNING: COLOR_WARNING,
                    AlertLevel.CRITICAL: COLOR_CRITICAL,
                    AlertLevel.EMERGENCY: COLOR_EMERGENCY
                }
                
                color = severity_colors.get(alert.severity, COLOR_NORMAL)
                device_info = f"[{alert.device_id:04X}]" if alert.device_id else "[ SYS ]"
                alert_text = f"{time_str} {device_info} {alert.description}"
                
                try:
                    self.screen.addstr(row, x, alert_text[:width-1], curses.color_pair(color))
                except curses.error:
                    pass
                
                row += 1
            
        except curses.error:
            pass
    
    def draw_dangerous_devices(self, y: int, x: int, width: int, height: int):
        """Draw status of dangerous devices (wipe/destruction)"""
        if not self.screen or not self.monitor:
            return
        
        try:
            # Panel title
            self.screen.addstr(y, x, "DANGEROUS DEVICES WATCH", 
                             curses.color_pair(COLOR_EMERGENCY) | curses.A_BOLD)
            
            # Get dangerous devices
            dangerous_devices = [d for d in self.monitor.devices.values() if d.is_dangerous]
            
            if not dangerous_devices:
                self.screen.addstr(y + 2, x, "No dangerous devices configured", curses.color_pair(COLOR_SAFE))
                return
            
            row = y + 2
            for device in dangerous_devices:
                if row >= y + height - 1:
                    break
                
                # Device status
                status_color = COLOR_EMERGENCY if device.current_status != 0 else COLOR_SAFE
                status_text = f"ACTIVE" if device.current_status != 0 else "SAFE"
                
                # Change indicator
                change_text = f"({device.change_count} changes)" if device.change_count > 0 else "(stable)"
                
                device_info = f"Token {device.token_id:04X}: {status_text:<8} Status: {device.current_status:3d} {change_text}"
                
                try:
                    self.screen.addstr(row, x, device_info[:width-1], curses.color_pair(status_color))
                except curses.error:
                    pass
                
                row += 1
            
            # Warning message
            if row < y + height - 2:
                warning = "⚠️  These devices can perform irreversible data wipe operations"
                try:
                    self.screen.addstr(row + 1, x, warning[:width-1], 
                                     curses.color_pair(COLOR_WARNING) | curses.A_BOLD)
                except curses.error:
                    pass
            
        except curses.error:
            pass
    
    def draw_controls(self, y: int, x: int, width: int):
        """Draw control instructions"""
        if not self.screen:
            return
        
        try:
            controls = [
                "CONTROLS: [q] Quit | [e] Emergency Stop | [r] Reset | [1-5] Change Mode",
                "Modes: [1] Overview | [2] Devices | [3] Security | [4] Thermal | [5] Network"
            ]
            
            for i, control in enumerate(controls):
                if y + i < curses.LINES:
                    self.screen.addstr(y + i, x, control[:width-1], curses.color_pair(COLOR_TITLE))
            
        except curses.error:
            pass
    
    def handle_input(self):
        """Handle keyboard input"""
        if not self.screen:
            return True
        
        try:
            key = self.screen.getch()
            
            if key == ord('q') or key == ord('Q'):
                return False  # Quit
            
            elif key == ord('e') or key == ord('E'):
                # Emergency stop
                self.cleanup_curses()
                print("🚨 Executing emergency stop...")
                self.emergency_stop.execute_emergency_stop("Manual emergency stop from dashboard")
                return False
            
            elif key == ord('r') or key == ord('R'):
                # Reset counters
                if self.monitor:
                    for device in self.monitor.devices.values():
                        device.change_count = 0
                        device.status_history = []
            
            elif key == ord('1'):
                self.dashboard_state.mode = DashboardMode.OVERVIEW
            elif key == ord('2'):
                self.dashboard_state.mode = DashboardMode.DEVICES  
            elif key == ord('3'):
                self.dashboard_state.mode = DashboardMode.SECURITY
            elif key == ord('4'):
                self.dashboard_state.mode = DashboardMode.THERMAL
            elif key == ord('5'):
                self.dashboard_state.mode = DashboardMode.NETWORK
            
        except curses.error:
            pass
        
        return True  # Continue running
    
    def run_dashboard(self):
        """Main dashboard loop"""
        try:
            # Initialize curses
            self.init_curses()
            
            # Start background monitoring
            if not self.start_monitoring():
                self.cleanup_curses()
                print("❌ Failed to start monitoring system")
                return False
            
            self.running = True
            
            # Main display loop
            while self.running:
                try:
                    # Get screen dimensions
                    height, width = self.screen.getmaxyx()
                    
                    # Clear screen
                    self.screen.clear()
                    
                    # Draw header
                    self.draw_header(0, 0, width)
                    
                    # Calculate panel dimensions
                    main_y = HEADER_HEIGHT
                    main_height = height - HEADER_HEIGHT - 3  # Reserve space for controls
                    
                    left_width = STATUS_PANEL_WIDTH
                    right_width = width - STATUS_PANEL_WIDTH - 2
                    
                    # Left panels
                    panel_height = main_height // 2
                    
                    # System status (top left)
                    self.draw_system_status(main_y, 0, left_width, panel_height)
                    
                    # Dangerous devices watch (bottom left)  
                    self.draw_dangerous_devices(main_y + panel_height, 0, left_width, panel_height)
                    
                    # Right panels
                    right_panel_height = (main_height * 2) // 3
                    alert_panel_height = main_height - right_panel_height
                    
                    # Device groups (top right)
                    self.draw_device_groups(main_y, left_width + 2, right_width, right_panel_height)
                    
                    # Recent alerts (bottom right)
                    self.draw_recent_alerts(main_y + right_panel_height, left_width + 2, 
                                          right_width, alert_panel_height)
                    
                    # Controls (bottom)
                    self.draw_controls(height - 3, 0, width)
                    
                    # Update dashboard state
                    self.dashboard_state.update_count += 1
                    self.dashboard_state.last_update = datetime.now()
                    
                    if self.monitor:
                        self.dashboard_state.devices_monitored = len(self.monitor.devices)
                        self.dashboard_state.alerts_count = len(self.alert_history)
                    
                    # Refresh screen
                    self.screen.refresh()
                    
                    # Handle input (non-blocking)
                    if not self.handle_input():
                        break
                    
                    # Sleep briefly
                    time.sleep(DASHBOARD_REFRESH_RATE)
                    
                except curses.error as e:
                    # Handle screen resize or other curses errors
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    # Log error but continue
                    time.sleep(1.0)
                    continue
            
            # Clean shutdown
            self.running = False
            self.cleanup_curses()
            print("✅ Dashboard shutdown complete")
            return True
            
        except KeyboardInterrupt:
            self.running = False
            self.cleanup_curses()
            print("\n🛑 Dashboard interrupted by user")
            return True
        except Exception as e:
            self.running = False
            self.cleanup_curses()
            print(f"❌ Dashboard error: {e}")
            return False

def main():
    """Main entry point for dashboard"""
    parser = argparse.ArgumentParser(
        description="DSMIL Real-Time Monitoring Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DASHBOARD FEATURES:
- Real-time monitoring of all 84 DSMIL devices
- System health and thermal monitoring
- Security alert tracking
- Dangerous device special monitoring
- Interactive controls and mode switching

KEYBOARD CONTROLS:
  q/Q     - Quit dashboard
  e/E     - Emergency stop
  r/R     - Reset counters
  1-5     - Switch display modes

REQUIREMENTS:
- Root privileges for SMI access
- Terminal with curses support
- Minimum 80x24 terminal size recommended

EXAMPLES:
  sudo python3 dsmil_dashboard.py
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["overview", "devices", "security", "thermal", "network"], 
        default="overview",
        help="Initial dashboard mode (default: overview)"
    )
    
    args = parser.parse_args()
    
    # Check root privileges
    if os.geteuid() != 0:
        print("❌ ERROR: Root privileges required for SMI access")
        print("Please run with: sudo python3 dsmil_dashboard.py")
        sys.exit(1)
    
    # Check terminal capabilities
    try:
        import curses
        curses.setupterm()
    except:
        print("❌ ERROR: Terminal does not support curses")
        print("Please use a compatible terminal (xterm, gnome-terminal, etc.)")
        sys.exit(1)
    
    # Create and run dashboard
    try:
        dashboard = DSMILDashboard()
        dashboard.dashboard_state.mode = args.mode
        
        success = dashboard.run_dashboard()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard startup interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()