#!/usr/bin/env python3
"""
DSMIL 72-Device Real-Time Monitoring Dashboard
Provides comprehensive monitoring for all DSMIL groups and devices
"""

import os
import sys
import time
import curses
import signal
import threading
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Configuration
SUDO_PASSWORD = "1786"
UPDATE_INTERVAL = 1.0  # seconds
THERMAL_WARNING = 75   # °C
THERMAL_CRITICAL = 85  # °C

class SystemStatus(Enum):
    """System-wide status levels"""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

@dataclass
class GroupMetrics:
    """Metrics for a DSMIL group"""
    group_id: int
    name: str
    active: bool
    temperature: int
    active_devices: int
    total_devices: int
    cpu_usage: float
    memory_usage: float
    errors: int
    
    @property
    def thermal_status(self) -> str:
        if self.temperature >= THERMAL_CRITICAL:
            return "CRITICAL"
        elif self.temperature >= THERMAL_WARNING:
            return "WARNING"
        else:
            return "NORMAL"

class DSMILMonitor:
    """Real-time monitoring system for DSMIL devices"""
    
    def __init__(self):
        self.running = False
        self.groups: List[GroupMetrics] = []
        self.system_status = SystemStatus.NORMAL
        self.total_devices = 72
        self.active_devices = 0
        self.uptime = 0
        self.start_time = time.time()
        self.error_log: List[str] = []
        self.kernel_messages: List[str] = []
        
        # Initialize groups
        self._initialize_groups()
    
    def _initialize_groups(self):
        """Initialize group structures"""
        group_names = [
            "Core Security",
            "Extended Security",
            "Network Operations",
            "Data Processing",
            "Communications",
            "Advanced Features"
        ]
        
        for i in range(6):
            self.groups.append(GroupMetrics(
                group_id=i,
                name=group_names[i],
                active=False,
                temperature=20,
                active_devices=0,
                total_devices=12,
                cpu_usage=0.0,
                memory_usage=0.0,
                errors=0
            ))
    
    def run_command(self, cmd: str, use_sudo: bool = False) -> Tuple[int, str]:
        """Execute a shell command"""
        if use_sudo:
            cmd = f"echo '{SUDO_PASSWORD}' | sudo -S {cmd} 2>/dev/null"
        
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=2
            )
            return result.returncode, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return -1, ""
    
    def update_metrics(self):
        """Update all metrics"""
        self.uptime = int(time.time() - self.start_time)
        
        # Update group metrics
        for group in self.groups:
            self._update_group_metrics(group)
        
        # Update system status
        self._update_system_status()
        
        # Get kernel messages
        self._update_kernel_messages()
        
        # Count active devices
        self.active_devices = sum(g.active_devices for g in self.groups)
    
    def _update_group_metrics(self, group: GroupMetrics):
        """Update metrics for a specific group"""
        # Simulate temperature (would read from sysfs in production)
        import random
        base_temp = 20
        if group.active:
            base_temp = 40 + random.randint(-5, 15)
            if group.group_id == 0:  # Core security runs hotter
                base_temp += 10
        group.temperature = base_temp
        
        # Simulate CPU usage
        if group.active:
            group.cpu_usage = 10.0 + random.random() * 30.0
        else:
            group.cpu_usage = 0.0
        
        # Simulate memory usage
        if group.active:
            group.memory_usage = 100.0 + random.random() * 200.0
        else:
            group.memory_usage = 0.0
        
        # Check for activation (would check sysfs in production)
        ret, out = self.run_command(f"lsmod | grep dsmil")
        if ret == 0:
            # Module is loaded, simulate some groups being active
            if group.group_id == 0:
                group.active = True
                group.active_devices = 12
    
    def _update_system_status(self):
        """Update overall system status"""
        max_temp = max(g.temperature for g in self.groups)
        
        if max_temp >= THERMAL_CRITICAL:
            self.system_status = SystemStatus.CRITICAL
        elif max_temp >= THERMAL_WARNING:
            self.system_status = SystemStatus.WARNING
        elif any(g.errors > 0 for g in self.groups):
            self.system_status = SystemStatus.WARNING
        else:
            self.system_status = SystemStatus.NORMAL
    
    def _update_kernel_messages(self):
        """Get recent kernel messages"""
        ret, out = self.run_command(
            "dmesg | grep -i dsmil | tail -5", use_sudo=True
        )
        if ret == 0 and out:
            self.kernel_messages = out.split('\n')[-5:]
    
    def get_cpu_info(self) -> Dict[str, float]:
        """Get CPU utilization info"""
        ret, out = self.run_command("top -bn1 | grep 'Cpu(s)'")
        if ret == 0:
            # Parse CPU usage
            parts = out.split(',')
            try:
                user = float(parts[0].split()[1].replace('%', ''))
                system = float(parts[1].split()[0].replace('%', ''))
                idle = float(parts[3].split()[0].replace('%', ''))
                return {'user': user, 'system': system, 'idle': idle}
            except:
                pass
        return {'user': 0.0, 'system': 0.0, 'idle': 100.0}
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get memory utilization info"""
        ret, out = self.run_command("free -m | grep Mem")
        if ret == 0:
            parts = out.split()
            try:
                total = int(parts[1])
                used = int(parts[2])
                free = int(parts[3])
                return {'total': total, 'used': used, 'free': free}
            except:
                pass
        return {'total': 0, 'used': 0, 'free': 0}

class DSMILMonitorUI:
    """Curses-based UI for DSMIL monitoring"""
    
    def __init__(self, monitor: DSMILMonitor):
        self.monitor = monitor
        self.screen = None
        self.running = True
    
    def run(self):
        """Run the monitoring UI"""
        try:
            curses.wrapper(self._main)
        except KeyboardInterrupt:
            pass
    
    def _main(self, stdscr):
        """Main UI loop"""
        self.screen = stdscr
        curses.curs_set(0)  # Hide cursor
        self.screen.nodelay(1)  # Non-blocking input
        
        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Normal
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)     # Critical
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Info
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
        
        while self.running:
            self.monitor.update_metrics()
            self._draw_interface()
            
            # Check for key press
            key = self.screen.getch()
            if key == ord('q'):
                self.running = False
            elif key == ord('e'):
                self._trigger_emergency_stop()
            elif key == ord('c'):
                self.monitor.error_log.clear()
            
            time.sleep(UPDATE_INTERVAL)
    
    def _draw_interface(self):
        """Draw the monitoring interface"""
        self.screen.clear()
        height, width = self.screen.getmaxyx()
        
        # Header
        self._draw_header(width)
        
        # System status
        self._draw_system_status(2, width)
        
        # Group status
        self._draw_group_status(6, width)
        
        # CPU and Memory
        self._draw_resource_usage(14, width)
        
        # Kernel messages
        self._draw_kernel_messages(18, width, height)
        
        # Footer
        self._draw_footer(height - 1, width)
        
        self.screen.refresh()
    
    def _draw_header(self, width):
        """Draw the header"""
        title = " DSMIL 72-Device Monitoring Dashboard "
        self.screen.attron(curses.color_pair(5) | curses.A_BOLD)
        self.screen.addstr(0, 0, " " * width)
        self.screen.addstr(0, (width - len(title)) // 2, title)
        self.screen.attroff(curses.color_pair(5) | curses.A_BOLD)
    
    def _draw_system_status(self, y, width):
        """Draw system status section"""
        self.screen.attron(curses.A_BOLD)
        self.screen.addstr(y, 0, "System Status:")
        self.screen.attroff(curses.A_BOLD)
        
        # Status indicator
        status_map = {
            SystemStatus.NORMAL: ("NORMAL", 1),
            SystemStatus.WARNING: ("WARNING", 2),
            SystemStatus.CRITICAL: ("CRITICAL", 3),
            SystemStatus.EMERGENCY: ("EMERGENCY", 3)
        }
        status_text, color = status_map[self.monitor.system_status]
        
        self.screen.attron(curses.color_pair(color))
        self.screen.addstr(y, 16, f"[{status_text}]")
        self.screen.attroff(curses.color_pair(color))
        
        # Device counts
        self.screen.addstr(y + 1, 0, 
            f"Active Devices: {self.monitor.active_devices}/{self.monitor.total_devices}")
        
        # Uptime
        hours = self.monitor.uptime // 3600
        minutes = (self.monitor.uptime % 3600) // 60
        seconds = self.monitor.uptime % 60
        self.screen.addstr(y + 1, 30, 
            f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Max temperature
        max_temp = max(g.temperature for g in self.monitor.groups)
        temp_color = 1 if max_temp < THERMAL_WARNING else (2 if max_temp < THERMAL_CRITICAL else 3)
        self.screen.addstr(y + 2, 0, "Max Temp: ")
        self.screen.attron(curses.color_pair(temp_color))
        self.screen.addstr(f"{max_temp}°C")
        self.screen.attroff(curses.color_pair(temp_color))
    
    def _draw_group_status(self, y, width):
        """Draw group status section"""
        self.screen.attron(curses.A_BOLD)
        self.screen.addstr(y, 0, "Group Status:")
        self.screen.attroff(curses.A_BOLD)
        
        # Header
        self.screen.addstr(y + 1, 0, 
            "ID  Name                 Status    Temp  Devices  CPU%   Mem(MB)")
        self.screen.addstr(y + 2, 0, "-" * 70)
        
        # Group rows
        for i, group in enumerate(self.monitor.groups):
            row_y = y + 3 + i
            
            # Status color
            if group.active:
                if group.temperature >= THERMAL_CRITICAL:
                    color = 3
                elif group.temperature >= THERMAL_WARNING:
                    color = 2
                else:
                    color = 1
            else:
                color = 0
            
            # Draw row
            status = "ACTIVE" if group.active else "INACTIVE"
            self.screen.addstr(row_y, 0, f"{group.group_id:2}")
            self.screen.addstr(row_y, 4, f"{group.name:20}")
            
            if color > 0:
                self.screen.attron(curses.color_pair(color))
            self.screen.addstr(row_y, 25, f"{status:8}")
            if color > 0:
                self.screen.attroff(curses.color_pair(color))
            
            self.screen.addstr(row_y, 34, f"{group.temperature:3}°C")
            self.screen.addstr(row_y, 41, f"{group.active_devices:2}/{group.total_devices:2}")
            self.screen.addstr(row_y, 50, f"{group.cpu_usage:5.1f}")
            self.screen.addstr(row_y, 57, f"{group.memory_usage:6.1f}")
    
    def _draw_resource_usage(self, y, width):
        """Draw resource usage section"""
        cpu_info = self.monitor.get_cpu_info()
        mem_info = self.monitor.get_memory_info()
        
        self.screen.attron(curses.A_BOLD)
        self.screen.addstr(y, 0, "System Resources:")
        self.screen.attroff(curses.A_BOLD)
        
        # CPU
        self.screen.addstr(y + 1, 0, 
            f"CPU: User: {cpu_info['user']:.1f}% System: {cpu_info['system']:.1f}% Idle: {cpu_info['idle']:.1f}%")
        
        # Memory
        if mem_info['total'] > 0:
            mem_percent = (mem_info['used'] / mem_info['total']) * 100
            self.screen.addstr(y + 2, 0,
                f"Memory: {mem_info['used']}MB / {mem_info['total']}MB ({mem_percent:.1f}%)")
    
    def _draw_kernel_messages(self, y, width, height):
        """Draw kernel messages section"""
        self.screen.attron(curses.A_BOLD)
        self.screen.addstr(y, 0, "Kernel Messages:")
        self.screen.attroff(curses.A_BOLD)
        
        # Display last few kernel messages
        max_messages = min(len(self.monitor.kernel_messages), height - y - 2)
        for i in range(max_messages):
            if i < len(self.monitor.kernel_messages):
                msg = self.monitor.kernel_messages[i][:width-1]
                self.screen.addstr(y + 1 + i, 0, msg)
    
    def _draw_footer(self, y, width):
        """Draw the footer"""
        help_text = " [Q]uit | [E]mergency Stop | [C]lear Errors "
        self.screen.attron(curses.color_pair(4))
        self.screen.addstr(y, 0, " " * width)
        self.screen.addstr(y, (width - len(help_text)) // 2, help_text)
        self.screen.attroff(curses.color_pair(4))
    
    def _trigger_emergency_stop(self):
        """Trigger emergency stop"""
        ret, _ = self.monitor.run_command(
            "echo 1 > /sys/class/dsmil/emergency_stop", use_sudo=True
        )
        if ret == 0:
            self.monitor.system_status = SystemStatus.EMERGENCY
            self.monitor.error_log.append(f"{datetime.now()}: Emergency stop triggered")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DSMIL 72-Device Monitor")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON instead of UI")
    parser.add_argument("--once", action="store_true",
                       help="Run once and exit")
    
    args = parser.parse_args()
    
    monitor = DSMILMonitor()
    
    if args.json:
        # JSON output mode
        monitor.update_metrics()
        data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': monitor.system_status.name,
            'active_devices': monitor.active_devices,
            'total_devices': monitor.total_devices,
            'uptime': monitor.uptime,
            'groups': [
                {
                    'id': g.group_id,
                    'name': g.name,
                    'active': g.active,
                    'temperature': g.temperature,
                    'active_devices': g.active_devices,
                    'cpu_usage': g.cpu_usage,
                    'memory_usage': g.memory_usage
                }
                for g in monitor.groups
            ]
        }
        print(json.dumps(data, indent=2))
    elif args.once:
        # Single run mode
        monitor.update_metrics()
        print(f"DSMIL System Status: {monitor.system_status.name}")
        print(f"Active Devices: {monitor.active_devices}/{monitor.total_devices}")
        for group in monitor.groups:
            if group.active:
                print(f"  Group {group.group_id} ({group.name}): "
                      f"{group.temperature}°C, {group.active_devices} devices")
    else:
        # Interactive UI mode
        ui = DSMILMonitorUI(monitor)
        ui.run()

if __name__ == "__main__":
    main()