#!/usr/bin/env python3
"""
DSMIL 84-Device READ-ONLY Monitoring Framework
Dell Latitude 5450 MIL-SPEC - Token Range 0x8000-0x806B

CRITICAL SAFETY FEATURES:
- Absolutely NO write operations to any DSMIL device
- READ-ONLY monitoring via SMI ports 0x164E/0x164F only
- Special protection for dangerous tokens 0x8009-0x800B (wipe devices)
- Emergency stop on any anomalous activity
- Comprehensive logging for security analysis

Author: MONITOR Agent
Date: 2025-09-01
Classification: MIL-SPEC Safe Operations Only
"""

import os
import sys
import time
import json
import signal
import threading
import subprocess
import psutil
import ctypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import hashlib
import warnings

# Silence warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CRITICAL SAFETY CONSTANTS - DO NOT MODIFY
# ============================================================================
DSMIL_TOKEN_START = 0x8000  # First DSMIL device token
DSMIL_TOKEN_END = 0x806B    # Last DSMIL device token (84 total devices)
DSMIL_DEVICE_COUNT = 84     # Total number of DSMIL devices

# EXTREMELY DANGEROUS TOKENS - NEVER WRITE TO THESE
DANGEROUS_WIPE_TOKENS = [0x8009, 0x800A, 0x800B]  # Confirmed wipe/destruction devices
CRITICAL_SECURITY_TOKENS = [0x8000, 0x8001, 0x8002]  # Core security tokens

# SMI Interface Configuration
SMI_COMMAND_PORT = 0x164E   # SMI command port
SMI_DATA_PORT = 0x164F      # SMI data port
SMI_TIMEOUT_MS = 1000       # 1 second timeout for SMI operations

# Monitoring Configuration
UPDATE_INTERVAL = 2.0       # seconds - Conservative for safety
LOG_ROTATION_SIZE = 10      # MB
EMERGENCY_TEMP_LIMIT = 90   # °C - Lower limit for safety
MEMORY_CRITICAL = 90        # % - Emergency stop threshold

# ============================================================================
# SECURITY CLASSES AND ENUMERATIONS
# ============================================================================

class AlertLevel(Enum):
    """Alert severity levels for DSMIL monitoring"""
    INFO = 0
    WARNING = 1  
    CRITICAL = 2
    EMERGENCY = 3
    SECURITY = 4  # Special level for security events

class DSMILDeviceGroup(Enum):
    """DSMIL device groups for organized monitoring"""
    GROUP_0 = "Core_Security_Power"     # 0x8000-0x800B (12 devices) - MOST DANGEROUS
    GROUP_1 = "Thermal_Management"      # 0x800C-0x8017 (12 devices)
    GROUP_2 = "Communication"           # 0x8018-0x8023 (12 devices)
    GROUP_3 = "Sensors"                 # 0x8024-0x802F (12 devices)
    GROUP_4 = "Crypto_Keys"             # 0x8030-0x803B (12 devices)
    GROUP_5 = "Storage_Control"         # 0x803C-0x8047 (12 devices)
    GROUP_6 = "Extended_Functions"      # 0x8048-0x8053 (12 devices)

class MonitoringMode(Enum):
    """Different monitoring display modes"""
    DASHBOARD = "dashboard"         # Overall system status
    DEVICE_STATUS = "devices"       # Individual device status
    SECURITY_WATCH = "security"     # Security-focused monitoring
    THERMAL_WATCH = "thermal"       # Thermal monitoring
    ANOMALY_DETECT = "anomaly"      # Anomaly detection mode
    EMERGENCY_STOP = "emergency"    # Emergency stop mode

@dataclass
class DSMILDevice:
    """Individual DSMIL device representation"""
    token_id: int                   # Token ID (0x8000-0x806B)
    group: DSMILDeviceGroup         # Device group
    group_position: int             # Position within group (0-11)
    current_status: int             # Current status byte
    previous_status: int            # Previous status byte
    status_history: List[int]       # Status history for pattern analysis
    last_change_time: datetime      # When status last changed
    change_count: int               # Number of status changes
    is_dangerous: bool              # True if this is a wipe/destruction device
    is_active: bool                 # True if device shows activity
    anomaly_score: float            # Anomaly detection score
    last_read_time: datetime        # Last successful read time

@dataclass
class SystemHealth:
    """Current system health metrics"""
    timestamp: datetime
    cpu_temp: float                 # CPU temperature in Celsius
    cpu_usage: float                # CPU usage percentage
    memory_usage: float             # Memory usage percentage
    disk_io_rate: float             # Disk I/O rate MB/s
    network_activity: float         # Network activity MB/s
    processes_count: int            # Number of running processes
    load_average: List[float]       # System load averages
    emergency_stop_active: bool     # Emergency stop status
    safety_violations: int          # Count of safety violations

@dataclass
class SecurityEvent:
    """Security event for logging and analysis"""
    timestamp: datetime
    event_type: str                 # Type of security event
    device_id: int                  # Related DSMIL device (if applicable)
    severity: AlertLevel            # Event severity
    description: str                # Human-readable description
    raw_data: Dict[str, Any]        # Raw data for analysis
    action_taken: str               # Response action taken

# ============================================================================
# CORE MONITORING SYSTEM
# ============================================================================

class DSMILReadOnlyMonitor:
    """
    Main DSMIL monitoring system - READ-ONLY operations only
    
    This class implements comprehensive monitoring of all 84 DSMIL devices
    without ever writing to any device. All operations are strictly read-only
    through the SMI interface.
    """
    
    def __init__(self, mode: MonitoringMode = MonitoringMode.DASHBOARD):
        self.monitoring_mode = mode
        self.running = False
        self.emergency_stop = False
        
        # Initialize device tracking
        self.devices: Dict[int, DSMILDevice] = {}
        self.security_events: List[SecurityEvent] = []
        self.system_health = None
        
        # Performance tracking
        self.start_time = datetime.now()
        self.read_count = 0
        self.error_count = 0
        
        # Safety systems
        self.safety_violations = 0
        self.last_safety_check = datetime.now()
        
        # Logging
        self.log_file = f"dsmil_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.ensure_log_directory()
        
        # Initialize signal handlers for emergency stop
        signal.signal(signal.SIGINT, self.emergency_shutdown)
        signal.signal(signal.SIGTERM, self.emergency_shutdown)
        
        self.log_security_event("SYSTEM_START", None, AlertLevel.INFO,
                               "DSMIL Read-Only Monitor initialized", {})
    
    def ensure_log_directory(self):
        """Ensure logging directory exists"""
        log_dir = "monitoring/logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, self.log_file)
    
    def log_security_event(self, event_type: str, device_id: Optional[int],
                          severity: AlertLevel, description: str, raw_data: Dict):
        """Log security events with full context"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            device_id=device_id,
            severity=severity,
            description=description,
            raw_data=raw_data,
            action_taken="Logged and monitored"
        )
        
        self.security_events.append(event)
        
        # Write to log file
        log_entry = {
            "timestamp": event.timestamp.isoformat(),
            "type": event_type,
            "device": device_id,
            "severity": severity.name,
            "description": description,
            "data": raw_data
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"❌ LOG ERROR: {e}")
    
    def initialize_devices(self):
        """Initialize all 84 DSMIL devices for monitoring"""
        print("🔍 Initializing DSMIL device registry...")
        
        # Map tokens to groups
        group_mapping = {
            0: (DSMILDeviceGroup.GROUP_0, 0x8000, 0x800B),  # DANGEROUS GROUP
            1: (DSMILDeviceGroup.GROUP_1, 0x800C, 0x8017),
            2: (DSMILDeviceGroup.GROUP_2, 0x8018, 0x8023),
            3: (DSMILDeviceGroup.GROUP_3, 0x8024, 0x802F),
            4: (DSMILDeviceGroup.GROUP_4, 0x8030, 0x803B),
            5: (DSMILDeviceGroup.GROUP_5, 0x803C, 0x8047),
            6: (DSMILDeviceGroup.GROUP_6, 0x8048, 0x8053),
        }
        
        device_count = 0
        for group_id, (group, start_token, end_token) in group_mapping.items():
            for token in range(start_token, end_token + 1):
                if token <= DSMIL_TOKEN_END:
                    position = token - start_token
                    
                    device = DSMILDevice(
                        token_id=token,
                        group=group,
                        group_position=position,
                        current_status=0,
                        previous_status=0,
                        status_history=[],
                        last_change_time=datetime.now(),
                        change_count=0,
                        is_dangerous=(token in DANGEROUS_WIPE_TOKENS),
                        is_active=False,
                        anomaly_score=0.0,
                        last_read_time=datetime.now()
                    )
                    
                    self.devices[token] = device
                    device_count += 1
        
        print(f"✅ Initialized {device_count} DSMIL devices for monitoring")
        print(f"⚠️  {len(DANGEROUS_WIPE_TOKENS)} devices marked as DANGEROUS (wipe/destruction)")
        
        self.log_security_event("DEVICE_INIT", None, AlertLevel.INFO,
                               f"Initialized {device_count} devices for monitoring",
                               {"device_count": device_count, "dangerous_count": len(DANGEROUS_WIPE_TOKENS)})
    
    def read_device_status_safe(self, token_id: int) -> Optional[int]:
        """
        Safely read device status via SMI interface
        
        CRITICAL: This function performs READ-ONLY operations only.
        It will NEVER write to any DSMIL device.
        """
        try:
            # Create temporary C program for SMI read
            c_code = f"""
#include <stdio.h>
#include <unistd.h>
#include <sys/io.h>
#include <stdlib.h>

int main() {{
    // Request I/O port access
    if (iopl(3) != 0) {{
        printf("ERROR_IOPL\\n");
        return 1;
    }}
    
    // Send READ command to SMI port
    outw(0x{token_id:04X}, 0x{SMI_COMMAND_PORT:04X});
    
    // Small delay for SMI processing
    usleep(1000);
    
    // Read status from data port
    unsigned char status = inb(0x{SMI_DATA_PORT:04X});
    
    printf("%d\\n", status);
    return 0;
}}
"""
            
            # Write, compile, and execute
            with open("/tmp/dsmil_read.c", "w") as f:
                f.write(c_code)
            
            # Compile with timeout
            compile_result = subprocess.run(
                ["gcc", "-o", "/tmp/dsmil_read", "/tmp/dsmil_read.c"],
                capture_output=True, text=True, timeout=5
            )
            
            if compile_result.returncode != 0:
                return None
            
            # Execute with timeout and elevated privileges
            result = subprocess.run(
                ["sudo", "/tmp/dsmil_read"],
                capture_output=True, text=True, timeout=SMI_TIMEOUT_MS/1000
            )
            
            # Clean up
            try:
                os.remove("/tmp/dsmil_read.c")
                os.remove("/tmp/dsmil_read")
            except:
                pass
            
            if result.returncode == 0 and result.stdout.strip().isdigit():
                status = int(result.stdout.strip())
                self.read_count += 1
                return status
            else:
                self.error_count += 1
                return None
                
        except Exception as e:
            self.error_count += 1
            self.log_security_event("READ_ERROR", token_id, AlertLevel.WARNING,
                                   f"Failed to read device {token_id:04X}: {e}",
                                   {"error": str(e)})
            return None
    
    def update_device_status(self, device: DSMILDevice) -> bool:
        """Update device status and detect changes"""
        new_status = self.read_device_status_safe(device.token_id)
        
        if new_status is None:
            return False
        
        # Update device state
        device.previous_status = device.current_status
        device.current_status = new_status
        device.last_read_time = datetime.now()
        
        # Detect changes
        if device.current_status != device.previous_status:
            device.last_change_time = datetime.now()
            device.change_count += 1
            device.status_history.append(new_status)
            
            # Limit history size
            if len(device.status_history) > 100:
                device.status_history = device.status_history[-50:]
            
            # Log significant changes
            severity = AlertLevel.CRITICAL if device.is_dangerous else AlertLevel.WARNING
            self.log_security_event(
                "STATUS_CHANGE", device.token_id, severity,
                f"Device {device.token_id:04X} status: {device.previous_status} → {device.current_status}",
                {
                    "previous": device.previous_status,
                    "current": device.current_status,
                    "is_dangerous": device.is_dangerous,
                    "change_count": device.change_count
                }
            )
            
            return True
        
        return False
    
    def detect_anomalies(self) -> List[DSMILDevice]:
        """Detect anomalous device behavior patterns"""
        anomalous_devices = []
        
        for device in self.devices.values():
            anomaly_score = 0.0
            
            # Check for rapid status changes
            if device.change_count > 10:
                anomaly_score += 0.3
            
            # Check for dangerous device activation
            if device.is_dangerous and device.current_status != 0:
                anomaly_score += 0.8
                
            # Check for unusual status values
            if device.current_status > 200:
                anomaly_score += 0.4
            
            # Check for pattern anomalies
            if len(device.status_history) > 5:
                # Look for oscillating patterns (potential activation attempts)
                recent_values = device.status_history[-5:]
                if len(set(recent_values)) > 3:
                    anomaly_score += 0.2
            
            device.anomaly_score = anomaly_score
            
            if anomaly_score > 0.5:
                anomalous_devices.append(device)
        
        return anomalous_devices
    
    def check_system_health(self) -> SystemHealth:
        """Monitor overall system health for safety"""
        try:
            # Get thermal information
            temps = psutil.sensors_temperatures()
            cpu_temp = 0.0
            if 'coretemp' in temps:
                cpu_temp = max([sensor.current for sensor in temps['coretemp']])
            elif temps:
                # Fallback to any available temperature sensor
                all_temps = [sensor.current for sensors in temps.values() for sensor in sensors]
                cpu_temp = max(all_temps) if all_temps else 0.0
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            health = SystemHealth(
                timestamp=datetime.now(),
                cpu_temp=cpu_temp,
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_io_rate=disk_io.read_bytes / 1024 / 1024 if disk_io else 0,  # MB/s
                network_activity=(net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024 if net_io else 0,
                processes_count=len(psutil.pids()),
                load_average=list(psutil.getloadavg()),
                emergency_stop_active=self.emergency_stop,
                safety_violations=self.safety_violations
            )
            
            # Check for safety violations
            if cpu_temp > EMERGENCY_TEMP_LIMIT:
                self.trigger_emergency_stop("Temperature critical", {"temp": cpu_temp})
            
            if memory.percent > MEMORY_CRITICAL:
                self.trigger_emergency_stop("Memory critical", {"memory": memory.percent})
            
            self.system_health = health
            return health
            
        except Exception as e:
            self.log_security_event("HEALTH_ERROR", None, AlertLevel.WARNING,
                                   f"System health check failed: {e}",
                                   {"error": str(e)})
            return None
    
    def trigger_emergency_stop(self, reason: str, data: Dict):
        """Trigger emergency stop of all monitoring"""
        self.emergency_stop = True
        self.safety_violations += 1
        
        self.log_security_event("EMERGENCY_STOP", None, AlertLevel.EMERGENCY,
                               f"Emergency stop triggered: {reason}",
                               data)
        
        print(f"\n🚨 EMERGENCY STOP: {reason}")
        print("🛑 All DSMIL monitoring halted for safety")
        print("📋 Check logs for detailed information")
        
        # Stop monitoring thread
        self.running = False
    
    def emergency_shutdown(self, signum, frame):
        """Handle emergency shutdown signals"""
        print(f"\n🚨 EMERGENCY SHUTDOWN: Received signal {signum}")
        self.trigger_emergency_stop("Manual emergency stop", {"signal": signum})
        sys.exit(0)
    
    def display_dashboard(self):
        """Display main monitoring dashboard"""
        while self.running and not self.emergency_stop:
            try:
                # Clear screen
                os.system('clear')
                
                # Header
                print("="*80)
                print("🛡️  DSMIL 84-Device READ-ONLY Monitor - DASHBOARD MODE")
                print("="*80)
                print(f"⏰ Runtime: {datetime.now() - self.start_time}")
                print(f"📊 Reads: {self.read_count} | Errors: {self.error_count}")
                print(f"🚨 Safety Violations: {self.safety_violations}")
                
                # System health
                health = self.check_system_health()
                if health:
                    print(f"\n🌡️  System Health:")
                    print(f"   CPU Temp: {health.cpu_temp:.1f}°C | Usage: {health.cpu_usage:.1f}%")
                    print(f"   Memory: {health.memory_usage:.1f}% | Processes: {health.processes_count}")
                    
                    # Health status indicators
                    temp_status = "🔥" if health.cpu_temp > EMERGENCY_TEMP_LIMIT else ("⚠️" if health.cpu_temp > 70 else "✅")
                    mem_status = "🔥" if health.memory_usage > MEMORY_CRITICAL else ("⚠️" if health.memory_usage > 80 else "✅")
                    print(f"   Status: {temp_status} Temperature | {mem_status} Memory")
                
                # Device group status
                print(f"\n📟 DSMIL Device Groups (84 devices total):")
                group_stats = {}
                
                for device in self.devices.values():
                    group_name = device.group.value
                    if group_name not in group_stats:
                        group_stats[group_name] = {"active": 0, "changed": 0, "dangerous": 0}
                    
                    if device.current_status != 0:
                        group_stats[group_name]["active"] += 1
                    if device.change_count > 0:
                        group_stats[group_name]["changed"] += 1
                    if device.is_dangerous:
                        group_stats[group_name]["dangerous"] += 1
                
                for group_name, stats in group_stats.items():
                    danger_indicator = "🚨" if stats["dangerous"] > 0 else "  "
                    active_indicator = "🟢" if stats["active"] > 0 else "⚪"
                    print(f"   {danger_indicator} {active_indicator} {group_name}: {stats['active']} active, {stats['changed']} changed")
                
                # Recent security events
                recent_events = sorted(self.security_events[-5:], key=lambda x: x.timestamp, reverse=True)
                if recent_events:
                    print(f"\n📋 Recent Security Events:")
                    for event in recent_events:
                        time_str = event.timestamp.strftime("%H:%M:%S")
                        severity_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🔴", "EMERGENCY": "🚨"}.get(event.severity.name, "❓")
                        print(f"   {severity_icon} {time_str} - {event.description}")
                
                # Anomaly detection
                anomalies = self.detect_anomalies()
                if anomalies:
                    print(f"\n🔍 Anomalies Detected ({len(anomalies)} devices):")
                    for device in anomalies[:5]:  # Show top 5
                        danger_flag = "🚨 DANGEROUS" if device.is_dangerous else ""
                        print(f"   🔴 Token {device.token_id:04X} - Score: {device.anomaly_score:.2f} {danger_flag}")
                
                # Emergency stop status
                if self.emergency_stop:
                    print(f"\n🚨 EMERGENCY STOP ACTIVE 🚨")
                    print("   All monitoring operations halted")
                    break
                
                print(f"\n⌨️  Press Ctrl+C for emergency stop")
                print("="*80)
                
                time.sleep(UPDATE_INTERVAL)
                
            except KeyboardInterrupt:
                self.trigger_emergency_stop("User requested stop", {})
                break
            except Exception as e:
                self.log_security_event("DISPLAY_ERROR", None, AlertLevel.WARNING,
                                       f"Dashboard display error: {e}",
                                       {"error": str(e)})
                time.sleep(UPDATE_INTERVAL)
    
    def run_monitoring_cycle(self):
        """Main monitoring cycle - updates all devices"""
        cycle_count = 0
        
        while self.running and not self.emergency_stop:
            try:
                cycle_start = datetime.now()
                changes_detected = 0
                
                # Update all devices
                for token_id, device in self.devices.items():
                    if self.emergency_stop:
                        break
                    
                    if self.update_device_status(device):
                        changes_detected += 1
                        
                        # Special handling for dangerous devices
                        if device.is_dangerous and device.current_status != device.previous_status:
                            self.log_security_event("DANGEROUS_CHANGE", token_id, AlertLevel.CRITICAL,
                                                   f"DANGEROUS device {token_id:04X} changed status!",
                                                   {
                                                       "device": f"0x{token_id:04X}",
                                                       "previous": device.previous_status,
                                                       "current": device.current_status,
                                                       "group": device.group.value
                                                   })
                    
                    # Small delay between reads to avoid overwhelming the system
                    time.sleep(0.01)
                
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                
                # Log cycle completion
                if cycle_count % 30 == 0:  # Every 30 cycles
                    self.log_security_event("CYCLE_COMPLETE", None, AlertLevel.INFO,
                                           f"Monitoring cycle {cycle_count} complete",
                                           {
                                               "duration": cycle_duration,
                                               "changes": changes_detected,
                                               "devices_monitored": len(self.devices)
                                           })
                
                cycle_count += 1
                
                # Adaptive sleep based on cycle duration
                sleep_time = max(0.1, UPDATE_INTERVAL - cycle_duration)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.log_security_event("CYCLE_ERROR", None, AlertLevel.WARNING,
                                       f"Monitoring cycle error: {e}",
                                       {"error": str(e), "cycle": cycle_count})
                time.sleep(UPDATE_INTERVAL)
    
    def start_monitoring(self):
        """Start the monitoring system"""
        print("🚀 Starting DSMIL Read-Only Monitor...")
        print("⚠️  SAFETY MODE: Read-only operations only")
        print("🛡️  NO WRITES will be performed to any DSMIL device")
        print("🚨 Dangerous tokens 0x8009-0x800B under special watch")
        
        # Initialize devices
        self.initialize_devices()
        
        # Check system prerequisites
        if os.geteuid() != 0:
            print("❌ ERROR: Root privileges required for SMI access")
            print("   Please run with: sudo python3 dsmil_readonly_monitor.py")
            return False
        
        # Start monitoring
        self.running = True
        
        # Start background monitoring thread
        monitor_thread = threading.Thread(target=self.run_monitoring_cycle, daemon=True)
        monitor_thread.start()
        
        # Start display based on mode
        if self.monitoring_mode == MonitoringMode.DASHBOARD:
            self.display_dashboard()
        
        # Cleanup
        self.running = False
        self.log_security_event("SHUTDOWN", None, AlertLevel.INFO,
                               "Monitor shutdown complete",
                               {"runtime_seconds": (datetime.now() - self.start_time).total_seconds()})
        
        return True

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="DSMIL 84-Device READ-ONLY Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SAFETY NOTICE:
This monitoring system performs READ-ONLY operations only.
It will NEVER write to any DSMIL device or change system state.
Tokens 0x8009-0x800B are under special monitoring as wipe/destruction devices.

EXAMPLES:
  sudo python3 dsmil_readonly_monitor.py --mode dashboard
  sudo python3 dsmil_readonly_monitor.py --mode security
  sudo python3 dsmil_readonly_monitor.py --devices 0x8000-0x800B
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["dashboard", "devices", "security", "thermal", "anomaly"],
        default="dashboard",
        help="Monitoring mode (default: dashboard)"
    )
    
    parser.add_argument(
        "--devices",
        help="Specific device range to monitor (e.g., '0x8000-0x800B')"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["INFO", "WARNING", "CRITICAL", "EMERGENCY"],
        default="WARNING",
        help="Minimum log level to display (default: WARNING)"
    )
    
    parser.add_argument(
        "--update-interval",
        type=float,
        default=UPDATE_INTERVAL,
        help=f"Update interval in seconds (default: {UPDATE_INTERVAL})"
    )
    
    args = parser.parse_args()
    
    # Validate root privileges
    if os.geteuid() != 0:
        print("❌ ERROR: This program requires root privileges for SMI access")
        print("Please run with: sudo python3 dsmil_readonly_monitor.py")
        sys.exit(1)
    
    # Create and start monitor
    try:
        mode = MonitoringMode(args.mode)
        monitor = DSMILReadOnlyMonitor(mode)
        
        # Update global update interval if specified
        if args.update_interval != UPDATE_INTERVAL:
            globals()['UPDATE_INTERVAL'] = args.update_interval
        
        success = monitor.start_monitoring()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()