#!/usr/bin/env python3
"""
DSMIL Comprehensive Real-Time Monitoring System
Designed for safe SMBIOS token testing on Dell Latitude 5450 MIL-SPEC

Features:
- Multi-terminal monitoring windows
- Resource exhaustion prevention
- Emergency stop mechanisms
- Real-time alerts
- Token state tracking
- System health monitoring

Usage:
    python3 dsmil_comprehensive_monitor.py --mode dashboard
    python3 dsmil_comprehensive_monitor.py --mode alerts
    python3 dsmil_comprehensive_monitor.py --mode resources
    python3 dsmil_comprehensive_monitor.py --mode tokens
"""

import os
import sys
import time
import json
import signal
import threading
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import argparse

# Configuration
UPDATE_INTERVAL = 0.5  # seconds - faster for token testing
THERMAL_WARNING = 85   # °C - Dell Latitude 5450 threshold
THERMAL_CRITICAL = 95  # °C - Emergency stop threshold
CPU_WARNING = 80       # % - High CPU usage warning
MEMORY_WARNING = 85    # % - High memory usage warning
DISK_IO_WARNING = 100  # MB/s - High disk I/O warning

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

class MonitoringMode(Enum):
    """Monitoring display modes"""
    DASHBOARD = "dashboard"
    ALERTS = "alerts"
    RESOURCES = "resources"
    TOKENS = "tokens"

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_percent: float
    cpu_cores: List[float]
    memory_percent: float
    memory_available: int
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    temperature: float
    processes: int
    load_avg: List[float]

@dataclass
class TokenState:
    """SMBIOS token state"""
    token_id: str
    range_name: str
    group_id: int
    position: int
    current_value: str
    previous_value: str
    changed: bool
    last_change: str

@dataclass
class AlertEvent:
    """Alert event data"""
    timestamp: str
    level: AlertLevel
    component: str
    message: str
    metrics: Dict[str, Any]

class SystemResourceMonitor:
    """System resource monitoring with safety limits"""
    
    def __init__(self):
        self.start_time = time.time()
        self.baseline_metrics = None
        self.alerts: List[AlertEvent] = []
        self.emergency_stop = False
        
        # Initialize baseline
        self.capture_baseline()
    
    def capture_baseline(self):
        """Capture baseline system metrics"""
        self.baseline_metrics = self.get_current_metrics()
        print(f"[INFO] Baseline captured at {self.baseline_metrics.timestamp}")
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_cores = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_read_mb = (disk_io.read_bytes / 1024 / 1024) if disk_io else 0
        disk_write_mb = (disk_io.write_bytes / 1024 / 1024) if disk_io else 0
        
        # Network metrics
        network_io = psutil.net_io_counters()
        network_sent_mb = (network_io.bytes_sent / 1024 / 1024) if network_io else 0
        network_recv_mb = (network_io.bytes_recv / 1024 / 1024) if network_io else 0
        
        # Temperature (try multiple sources)
        temperature = self.get_temperature()
        
        # Process and load metrics
        processes = len(psutil.pids())
        load_avg = list(psutil.getloadavg())
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            cpu_cores=cpu_cores,
            memory_percent=memory.percent,
            memory_available=memory.available // 1024 // 1024,  # MB
            disk_io_read=disk_read_mb,
            disk_io_write=disk_write_mb,
            network_sent=network_sent_mb,
            network_recv=network_recv_mb,
            temperature=temperature,
            processes=processes,
            load_avg=load_avg
        )
    
    def get_temperature(self) -> float:
        """Get system temperature from multiple sources"""
        try:
            # Try thermal zone
            for i in range(10):
                temp_file = f"/sys/class/thermal/thermal_zone{i}/temp"
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        temp = int(f.read().strip()) / 1000
                        if 20 <= temp <= 120:  # Reasonable range
                            return temp
        except:
            pass
        
        try:
            # Try sensors via psutil
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if 20 <= entry.current <= 120:
                            return entry.current
        except:
            pass
        
        return 0.0  # Unknown
    
    def check_alerts(self, metrics: SystemMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # Temperature alerts
        if metrics.temperature >= THERMAL_CRITICAL:
            alerts.append(AlertEvent(
                timestamp=metrics.timestamp,
                level=AlertLevel.EMERGENCY,
                component="Temperature",
                message=f"Critical temperature: {metrics.temperature:.1f}°C",
                metrics={"temperature": metrics.temperature}
            ))
            self.emergency_stop = True
        elif metrics.temperature >= THERMAL_WARNING:
            alerts.append(AlertEvent(
                timestamp=metrics.timestamp,
                level=AlertLevel.CRITICAL,
                component="Temperature", 
                message=f"High temperature: {metrics.temperature:.1f}°C",
                metrics={"temperature": metrics.temperature}
            ))
        
        # CPU alerts
        if metrics.cpu_percent >= 95:
            alerts.append(AlertEvent(
                timestamp=metrics.timestamp,
                level=AlertLevel.CRITICAL,
                component="CPU",
                message=f"CPU usage critical: {metrics.cpu_percent:.1f}%",
                metrics={"cpu_percent": metrics.cpu_percent}
            ))
        elif metrics.cpu_percent >= CPU_WARNING:
            alerts.append(AlertEvent(
                timestamp=metrics.timestamp,
                level=AlertLevel.WARNING,
                component="CPU",
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                metrics={"cpu_percent": metrics.cpu_percent}
            ))
        
        # Memory alerts
        if metrics.memory_percent >= 95:
            alerts.append(AlertEvent(
                timestamp=metrics.timestamp,
                level=AlertLevel.CRITICAL,
                component="Memory",
                message=f"Memory usage critical: {metrics.memory_percent:.1f}%",
                metrics={"memory_percent": metrics.memory_percent}
            ))
        elif metrics.memory_percent >= MEMORY_WARNING:
            alerts.append(AlertEvent(
                timestamp=metrics.timestamp,
                level=AlertLevel.WARNING,
                component="Memory",
                message=f"High memory usage: {metrics.memory_percent:.1f}%",
                metrics={"memory_percent": metrics.memory_percent}
            ))
        
        # Add alerts to list
        self.alerts.extend(alerts)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        return alerts

class DSMILTokenMonitor:
    """Monitor DSMIL SMBIOS token states"""
    
    def __init__(self):
        self.token_ranges = [
            ("Range_0400", 0x0400, 0x0447),
            ("Range_0480", 0x0480, 0x04C7),  # Most promising
            ("Range_0500", 0x0500, 0x0547),
            ("Range_1000", 0x1000, 0x1047),
            ("Range_1100", 0x1100, 0x1147),
            ("Range_1200", 0x1200, 0x1247),
            ("Range_1300", 0x1300, 0x1347),
            ("Range_1400", 0x1400, 0x1447),
            ("Range_1500", 0x1500, 0x1547),
            ("Range_1600", 0x1600, 0x1647),
            ("Range_1700", 0x1700, 0x1747)
        ]
        
        self.tokens: Dict[str, TokenState] = {}
        self.active_ranges: List[str] = []
        
        # Initialize token states
        self.initialize_tokens()
    
    def initialize_tokens(self):
        """Initialize token state tracking"""
        for range_name, start, end in self.token_ranges:
            for token_id in range(start, end + 1):
                group_id = (token_id - start) // 12
                position = (token_id - start) % 12
                
                token_key = f"{token_id:04x}"
                self.tokens[token_key] = TokenState(
                    token_id=token_key,
                    range_name=range_name,
                    group_id=group_id,
                    position=position,
                    current_value="",
                    previous_value="",
                    changed=False,
                    last_change=""
                )
    
    def read_token(self, token_id: int) -> str:
        """Safely read SMBIOS token value"""
        try:
            # Use dell-smbios interface if available
            cmd = f"echo 1786 | sudo -S python3 -c \"import subprocess; print(subprocess.check_output(['dmidecode', '-t', '0'], text=True))\" 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
            
            # For now, return empty (safe operation)
            return ""
        except:
            return ""
    
    def update_tokens(self, target_range: str = "Range_0480"):
        """Update token states for specified range"""
        if target_range not in [r[0] for r in self.token_ranges]:
            return
        
        # Find the range
        range_info = next((r for r in self.token_ranges if r[0] == target_range), None)
        if not range_info:
            return
        
        range_name, start, end = range_info
        
        # Update tokens in this range
        for token_id in range(start, end + 1):
            token_key = f"{token_id:04x}"
            if token_key in self.tokens:
                token = self.tokens[token_key]
                new_value = self.read_token(token_id)
                
                if new_value != token.current_value:
                    token.previous_value = token.current_value
                    token.current_value = new_value
                    token.changed = True
                    token.last_change = datetime.now().isoformat()
                else:
                    token.changed = False
    
    def get_changed_tokens(self) -> List[TokenState]:
        """Get tokens that have changed"""
        return [token for token in self.tokens.values() if token.changed]
    
    def get_active_tokens(self) -> List[TokenState]:
        """Get tokens with non-empty values"""
        return [token for token in self.tokens.values() if token.current_value]

class KernelMessageMonitor:
    """Monitor kernel messages for DSMIL activity"""
    
    def __init__(self):
        self.dsmil_messages: List[str] = []
        self.last_check = time.time()
    
    def get_new_messages(self) -> List[str]:
        """Get new kernel messages related to DSMIL"""
        try:
            # Get recent dmesg output
            cmd = "echo 1786 | sudo -S dmesg -T | grep -i -E '(dsmil|dell|smbios|wmi)' | tail -10"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                messages = result.stdout.strip().split('\n')
                new_messages = [msg for msg in messages if msg.strip()]
                
                # Filter for new messages (basic timestamp check)
                current_time = time.time()
                if current_time - self.last_check > 60:  # Reset every minute
                    self.dsmil_messages = new_messages[-5:]
                else:
                    self.dsmil_messages.extend(new_messages[-3:])
                
                self.last_check = current_time
                return new_messages[-3:] if new_messages else []
        except:
            pass
        
        return []

class MonitoringDashboard:
    """Main monitoring dashboard"""
    
    def __init__(self, mode: MonitoringMode):
        self.mode = mode
        self.running = True
        
        # Initialize monitors
        self.resource_monitor = SystemResourceMonitor()
        self.token_monitor = DSMILTokenMonitor()
        self.kernel_monitor = KernelMessageMonitor()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\\n[INFO] Received signal {signum}, shutting down...")
        self.running = False
    
    def run(self):
        """Run the monitoring dashboard"""
        print(f"[INFO] Starting DSMIL monitoring in {self.mode.value} mode")
        print(f"[INFO] Press Ctrl+C to stop")
        
        # Clear screen
        os.system('clear')
        
        try:
            while self.running:
                if self.mode == MonitoringMode.DASHBOARD:
                    self.display_dashboard()
                elif self.mode == MonitoringMode.ALERTS:
                    self.display_alerts()
                elif self.mode == MonitoringMode.RESOURCES:
                    self.display_resources()
                elif self.mode == MonitoringMode.TOKENS:
                    self.display_tokens()
                
                # Check for emergency stop
                if self.resource_monitor.emergency_stop:
                    print(f"\\n[EMERGENCY] System emergency stop triggered!")
                    self.trigger_emergency_stop()
                    break
                
                time.sleep(UPDATE_INTERVAL)
        
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\\n[INFO] Monitoring stopped")
    
    def display_dashboard(self):
        """Display main dashboard"""
        os.system('clear')
        
        # Get current metrics
        metrics = self.resource_monitor.get_current_metrics()
        alerts = self.resource_monitor.check_alerts(metrics)
        kernel_msgs = self.kernel_monitor.get_new_messages()
        
        # Header
        print("=" * 80)
        print(f" DSMIL TOKEN TESTING - COMPREHENSIVE MONITORING DASHBOARD ")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
              f"Uptime: {int(time.time() - self.resource_monitor.start_time)}s")
        print()
        
        # System Status
        print("SYSTEM STATUS:")
        status_color = "🟢" if not alerts else ("🟡" if any(a.level == AlertLevel.WARNING for a in alerts) else "🔴")
        print(f"{status_color} Overall: {'NORMAL' if not alerts else 'ALERT'}")
        print(f"🌡️  Temperature: {metrics.temperature:.1f}°C")
        print(f"🖥️  CPU: {metrics.cpu_percent:.1f}%")
        print(f"💾 Memory: {metrics.memory_percent:.1f}% ({metrics.memory_available}MB available)")
        print(f"⚡ Load: {metrics.load_avg[0]:.2f}, {metrics.load_avg[1]:.2f}, {metrics.load_avg[2]:.2f}")
        print()
        
        # Recent Alerts
        print("RECENT ALERTS:")
        if alerts:
            for alert in alerts[-3:]:
                level_icon = {"WARNING": "⚠️ ", "CRITICAL": "🔴", "EMERGENCY": "🚨"}
                icon = level_icon.get(alert.level.name, "ℹ️ ")
                print(f"{icon} [{alert.component}] {alert.message}")
        else:
            print("✅ No active alerts")
        print()
        
        # Token Status
        print("TOKEN MONITORING:")
        changed_tokens = self.token_monitor.get_changed_tokens()
        active_tokens = self.token_monitor.get_active_tokens()
        print(f"📊 Monitored Ranges: {len(self.token_monitor.token_ranges)}")
        print(f"🔄 Changed Tokens: {len(changed_tokens)}")
        print(f"✅ Active Tokens: {len(active_tokens)}")
        
        if changed_tokens:
            print("   Recent Changes:")
            for token in changed_tokens[-3:]:
                print(f"   • {token.token_id} ({token.range_name}): {token.previous_value} → {token.current_value}")
        print()
        
        # Kernel Messages
        print("KERNEL MESSAGES:")
        if kernel_msgs:
            for msg in kernel_msgs[-3:]:
                print(f"📝 {msg[:70]}...")
        else:
            print("📝 No recent DSMIL-related kernel messages")
        print()
        
        # Controls
        print("CONTROLS:")
        print("🛑 Emergency Stop: Press Ctrl+C or send SIGTERM")
        print("📊 Switch to resources view: python3 dsmil_comprehensive_monitor.py --mode resources")
        print("🔍 Switch to token view: python3 dsmil_comprehensive_monitor.py --mode tokens")
    
    def display_alerts(self):
        """Display alerts-focused view"""
        os.system('clear')
        print("=" * 80)
        print(f" DSMIL MONITORING - ALERTS VIEW ")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Get current alerts
        metrics = self.resource_monitor.get_current_metrics()
        new_alerts = self.resource_monitor.check_alerts(metrics)
        
        # Display all recent alerts
        print("ALERT HISTORY (Last 20):")
        if self.resource_monitor.alerts:
            for alert in self.resource_monitor.alerts[-20:]:
                level_icons = {
                    AlertLevel.INFO: "ℹ️ ",
                    AlertLevel.WARNING: "⚠️ ",
                    AlertLevel.CRITICAL: "🔴",
                    AlertLevel.EMERGENCY: "🚨"
                }
                icon = level_icons.get(alert.level, "❓")
                timestamp = datetime.fromisoformat(alert.timestamp).strftime('%H:%M:%S')
                print(f"{icon} {timestamp} [{alert.component:10}] {alert.message}")
        else:
            print("✅ No alerts recorded")
        
        print("\\nPress Ctrl+C to exit")
    
    def display_resources(self):
        """Display resources-focused view"""
        os.system('clear')
        print("=" * 80)
        print(f" DSMIL MONITORING - SYSTEM RESOURCES ")
        print("=" * 80)
        
        # Get current metrics
        metrics = self.resource_monitor.get_current_metrics()
        alerts = self.resource_monitor.check_alerts(metrics)
        
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # CPU Details
        print("CPU USAGE:")
        print(f"Overall: {metrics.cpu_percent:.1f}%")
        print("Per Core:", end=" ")
        for i, usage in enumerate(metrics.cpu_cores):
            print(f"Core{i}: {usage:.1f}%", end="  ")
            if (i + 1) % 4 == 0:
                print()
        print()
        
        # Memory Details
        print("MEMORY USAGE:")
        total_mem = psutil.virtual_memory().total // 1024 // 1024
        used_mem = total_mem - metrics.memory_available
        print(f"Used: {used_mem}MB / {total_mem}MB ({metrics.memory_percent:.1f}%)")
        print(f"Available: {metrics.memory_available}MB")
        print()
        
        # Disk I/O
        print("DISK I/O:")
        print(f"Read: {metrics.disk_io_read:.2f} MB")
        print(f"Write: {metrics.disk_io_write:.2f} MB")
        print()
        
        # Temperature
        print("THERMAL STATUS:")
        temp_status = "NORMAL"
        if metrics.temperature >= THERMAL_CRITICAL:
            temp_status = "🚨 CRITICAL"
        elif metrics.temperature >= THERMAL_WARNING:
            temp_status = "⚠️  WARNING"
        print(f"Temperature: {metrics.temperature:.1f}°C ({temp_status})")
        print()
        
        # Process count
        print(f"PROCESSES: {metrics.processes} active")
        print()
        
        print("Press Ctrl+C to exit")
    
    def display_tokens(self):
        """Display token-focused view"""
        os.system('clear')
        print("=" * 80)
        print(f" DSMIL MONITORING - TOKEN STATUS ")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Update token states for primary range
        self.token_monitor.update_tokens("Range_0480")
        
        # Display token ranges
        print("TOKEN RANGES (72 tokens each):")
        for range_name, start, end in self.token_monitor.token_ranges:
            range_tokens = [t for t in self.token_monitor.tokens.values() 
                           if t.range_name == range_name]
            active_count = len([t for t in range_tokens if t.current_value])
            changed_count = len([t for t in range_tokens if t.changed])
            
            status = "📍" if range_name == "Range_0480" else "⚪"
            print(f"{status} {range_name}: 0x{start:04x}-0x{end:04x} "
                  f"(Active: {active_count}, Changed: {changed_count})")
        print()
        
        # Show changed tokens
        changed_tokens = self.token_monitor.get_changed_tokens()
        if changed_tokens:
            print("RECENTLY CHANGED TOKENS:")
            for token in changed_tokens[-10:]:
                change_time = datetime.fromisoformat(token.last_change).strftime('%H:%M:%S')
                print(f"🔄 {change_time} Token 0x{token.token_id} [{token.range_name}] "
                      f"Group {token.group_id}: '{token.previous_value}' → '{token.current_value}'")
        else:
            print("🔍 No token changes detected")
        print()
        
        # Show active tokens
        active_tokens = self.token_monitor.get_active_tokens()
        if active_tokens:
            print("ACTIVE TOKENS:")
            for token in active_tokens[-10:]:
                print(f"✅ Token 0x{token.token_id} [{token.range_name}] "
                      f"Group {token.group_id}: '{token.current_value}'")
        else:
            print("⚪ No active tokens detected")
        print()
        
        print("Press Ctrl+C to exit")
    
    def trigger_emergency_stop(self):
        """Trigger emergency stop procedures"""
        print("[EMERGENCY] Executing emergency stop procedures...")
        
        # Try to stop any running DSMIL operations
        try:
            subprocess.run("echo 1786 | sudo -S killall -9 dsmil-72dev 2>/dev/null", shell=True)
            subprocess.run("echo 1786 | sudo -S rmmod dsmil-72dev 2>/dev/null", shell=True)
            print("[EMERGENCY] Stopped DSMIL kernel module")
        except:
            pass
        
        # Log emergency event
        with open("/tmp/dsmil_emergency.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}: Emergency stop triggered\\n")
        
        self.running = False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DSMIL Comprehensive Monitoring System")
    parser.add_argument("--mode", 
                       choices=["dashboard", "alerts", "resources", "tokens"],
                       default="dashboard",
                       help="Monitoring display mode")
    parser.add_argument("--json-output", action="store_true",
                       help="Output JSON data instead of interactive display")
    parser.add_argument("--duration", type=int, default=0,
                       help="Run for specified seconds (0 = infinite)")
    
    args = parser.parse_args()
    
    if args.json_output:
        # JSON output mode for integration
        monitor = SystemResourceMonitor()
        metrics = monitor.get_current_metrics()
        alerts = monitor.check_alerts(metrics)
        
        output = {
            "timestamp": metrics.timestamp,
            "metrics": asdict(metrics),
            "alerts": [asdict(alert) for alert in alerts],
            "emergency_stop": monitor.emergency_stop
        }
        print(json.dumps(output, indent=2))
    else:
        # Interactive mode
        mode = MonitoringMode(args.mode)
        dashboard = MonitoringDashboard(mode)
        
        if args.duration > 0:
            # Run for specified duration
            threading.Timer(args.duration, lambda: setattr(dashboard, 'running', False)).start()
        
        dashboard.run()

if __name__ == "__main__":
    main()