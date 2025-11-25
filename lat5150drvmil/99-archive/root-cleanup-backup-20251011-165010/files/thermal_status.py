#!/usr/bin/env python3
"""
Thermal Status Monitor - CLI interface for Thermal Guardian
Agent 3 Implementation

Provides real-time monitoring and status information for the thermal
management system on Dell LAT5150DRVMIL.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Add the thermal guardian directory to path
sys.path.insert(0, '/opt/thermal-guardian')

try:
    from thermal_guardian import ThermalManager, ThermalConfig, load_config
except ImportError:
    # Fallback for development/testing
    try:
        from thermal_guardian import ThermalManager, ThermalConfig, load_config
    except ImportError:
        print("Error: Cannot import thermal_guardian. Please ensure it's installed correctly.")
        sys.exit(1)

class ThermalStatusMonitor:
    """Interactive thermal status monitoring"""
    
    def __init__(self, config_path: str = "/etc/thermal_guardian.conf"):
        self.config = load_config(config_path)
        self.thermal_manager = None
        
    def initialize_manager(self):
        """Initialize thermal manager for status reading"""
        try:
            self.thermal_manager = ThermalManager(self.config)
            return True
        except Exception as e:
            print(f"Error initializing thermal manager: {e}")
            return False
            
    def get_service_status(self):
        """Get systemd service status"""
        try:
            result = subprocess.run(['systemctl', 'is-active', 'thermal-guardian'], 
                                  capture_output=True, text=True)
            service_active = result.stdout.strip() == 'active'
            
            result = subprocess.run(['systemctl', 'is-enabled', 'thermal-guardian'], 
                                  capture_output=True, text=True)
            service_enabled = result.stdout.strip() == 'enabled'
            
            return {
                'active': service_active,
                'enabled': service_enabled,
                'status': 'running' if service_active else 'stopped'
            }
        except Exception:
            return {
                'active': False,
                'enabled': False,
                'status': 'unknown'
            }
            
    def format_temperature(self, temp: float) -> str:
        """Format temperature with color coding"""
        if temp >= 100:
            return f"\033[91m{temp:.1f}°C\033[0m"  # Red
        elif temp >= 90:
            return f"\033[93m{temp:.1f}°C\033[0m"  # Yellow
        elif temp >= 80:
            return f"\033[92m{temp:.1f}°C\033[0m"  # Green
        else:
            return f"{temp:.1f}°C"
            
    def format_state(self, state: str) -> str:
        """Format thermal state with color coding"""
        colors = {
            'normal': '\033[92m',     # Green
            'warm': '\033[93m',       # Yellow
            'hot': '\033[91m',        # Red
            'critical': '\033[95m',   # Magenta
            'emergency': '\033[41m',  # Red background
            'shutdown': '\033[41m'    # Red background
        }
        color = colors.get(state.lower(), '')
        return f"{color}{state.upper()}\033[0m"
        
    def show_brief_status(self):
        """Show brief status information"""
        if not self.thermal_manager:
            if not self.initialize_manager():
                return
                
        status = self.thermal_manager.get_status()
        service = self.get_service_status()
        
        print(f"Thermal Guardian Status: {service['status']}")
        print(f"Current State: {self.format_state(status['current_state'])}")
        print(f"Max Temperature: {self.format_temperature(status['max_temperature'])} ({status['max_sensor']})")
        
        if status['predicted_temperature']:
            print(f"Predicted (5s): {self.format_temperature(status['predicted_temperature'])}")
            
        print(f"Sensors: {status['sensors_healthy']}/{status['total_sensors']} healthy")
        
        if status['emergency_triggered']:
            print("\033[91m⚠ EMERGENCY MODE ACTIVE ⚠\033[0m")
            
    def show_detailed_status(self):
        """Show detailed status information"""
        if not self.thermal_manager:
            if not self.initialize_manager():
                return
                
        status = self.thermal_manager.get_status()
        service = self.get_service_status()
        
        print("=" * 60)
        print("THERMAL GUARDIAN - DETAILED STATUS")
        print("=" * 60)
        print()
        
        # Service status
        print("SERVICE STATUS:")
        print(f"  Active: {'Yes' if service['active'] else 'No'}")
        print(f"  Enabled: {'Yes' if service['enabled'] else 'No'}")
        print(f"  Status: {service['status']}")
        print()
        
        # Current thermal state
        print("THERMAL STATE:")
        print(f"  Current: {self.format_state(status['current_state'])}")
        print(f"  Emergency Triggered: {'Yes' if status['emergency_triggered'] else 'No'}")
        print()
        
        # Temperature readings
        print("TEMPERATURE SENSORS:")
        print(f"  Maximum: {self.format_temperature(status['max_temperature'])} ({status['max_sensor']})")
        
        if status['predicted_temperature']:
            print(f"  Predicted (5s): {self.format_temperature(status['predicted_temperature'])}")
            
        print(f"  All sensors ({len(status['all_temperatures'])}):")
        for sensor, temp in sorted(status['all_temperatures'].items()):
            print(f"    {sensor:30s}: {self.format_temperature(temp)}")
        print()
        
        # System health
        print("SYSTEM HEALTH:")
        print(f"  Healthy Sensors: {status['sensors_healthy']}/{status['total_sensors']}")
        print(f"  Fan Controllers: {status['fan_controllers']}")
        print(f"  CPU Control: {'Available' if status['cpu_control_available'] else 'Not Available'}")
        
        if status['total_throttle_time'] > 0:
            print(f"  Total Throttle Time: {status['total_throttle_time']:.1f} seconds")
        print()
        
        # Configuration summary
        print("CONFIGURATION:")
        print(f"  Normal Threshold: {self.config.temp_normal}°C")
        print(f"  Critical Threshold: {self.config.temp_critical}°C")
        print(f"  Emergency Threshold: {self.config.temp_emergency}°C")
        print(f"  Shutdown Threshold: {self.config.temp_shutdown}°C")
        print(f"  Poll Interval: {self.config.poll_interval}s")
        print()
        
    def watch_temperatures(self, interval: float = 2.0):
        """Continuously monitor temperatures"""
        if not self.thermal_manager:
            if not self.initialize_manager():
                return
                
        print("Thermal Guardian - Live Temperature Monitor")
        print("Press Ctrl+C to exit")
        print()
        
        try:
            while True:
                # Clear screen (ANSI escape sequence)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                status = self.thermal_manager.get_status()
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                print(f"Thermal Guardian Live Monitor - {timestamp}")
                print("=" * 50)
                print(f"State: {self.format_state(status['current_state'])}")
                print(f"Max:   {self.format_temperature(status['max_temperature'])} ({status['max_sensor']})")
                
                if status['predicted_temperature']:
                    print(f"Pred:  {self.format_temperature(status['predicted_temperature'])} (5s ahead)")
                    
                print()
                print("All Sensors:")
                
                for sensor, temp in sorted(status['all_temperatures'].items()):
                    # Truncate long sensor names
                    sensor_short = sensor[:25] + "..." if len(sensor) > 28 else sensor
                    print(f"  {sensor_short:28s}: {self.format_temperature(temp)}")
                    
                if status['emergency_triggered']:
                    print()
                    print("\033[91m⚠ EMERGENCY MODE ACTIVE ⚠\033[0m")
                    
                print()
                print("Press Ctrl+C to exit")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            
    def show_log_tail(self, lines: int = 20):
        """Show recent log entries"""
        log_file = self.config.log_file
        
        if not Path(log_file).exists():
            print(f"Log file not found: {log_file}")
            return
            
        try:
            result = subprocess.run(['tail', f'-{lines}', log_file], 
                                  capture_output=True, text=True)
            if result.stdout:
                print(f"Last {lines} lines from {log_file}:")
                print("-" * 60)
                print(result.stdout)
            else:
                print("No log entries found.")
                
        except Exception as e:
            print(f"Error reading log file: {e}")
            
    def show_hardware_info(self):
        """Show hardware information"""
        print("HARDWARE INFORMATION:")
        print("=" * 40)
        
        # System information
        try:
            with open('/sys/class/dmi/id/product_name', 'r') as f:
                product = f.read().strip()
            print(f"System: {product}")
        except:
            print("System: Unknown")
            
        # Thermal zones
        thermal_zones = list(Path("/sys/class/thermal").glob("thermal_zone*"))
        print(f"Thermal Zones: {len(thermal_zones)}")
        
        for zone in sorted(thermal_zones):
            try:
                with open(zone / "type", 'r') as f:
                    zone_type = f.read().strip()
                with open(zone / "temp", 'r') as f:
                    temp = int(f.read().strip()) / 1000
                print(f"  {zone.name}: {zone_type} ({temp:.1f}°C)")
            except:
                print(f"  {zone.name}: Error reading")
                
        # Hardware monitors
        hwmon_dirs = list(Path("/sys/class/hwmon").glob("hwmon*"))
        print(f"Hardware Monitors: {len(hwmon_dirs)}")
        
        for hwmon in sorted(hwmon_dirs):
            try:
                with open(hwmon / "name", 'r') as f:
                    name = f.read().strip()
                print(f"  {hwmon.name}: {name}")
            except:
                print(f"  {hwmon.name}: Error reading")
                
        # CPU information
        if Path("/sys/devices/system/cpu/intel_pstate").exists():
            print("CPU Frequency Control: Intel P-State available")
        else:
            print("CPU Frequency Control: Not available")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Thermal Status Monitor for Thermal Guardian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  thermal_status.py              Show brief status
  thermal_status.py --detailed   Show detailed status
  thermal_status.py --watch      Live temperature monitoring
  thermal_status.py --logs       Show recent log entries
  thermal_status.py --hardware   Show hardware information
        """
    )
    
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Show detailed status information")
    parser.add_argument("--watch", "-w", action="store_true",
                       help="Continuously monitor temperatures")
    parser.add_argument("--interval", "-i", type=float, default=2.0,
                       help="Update interval for watch mode (default: 2.0s)")
    parser.add_argument("--logs", "-l", action="store_true",
                       help="Show recent log entries")
    parser.add_argument("--log-lines", type=int, default=20,
                       help="Number of log lines to show (default: 20)")
    parser.add_argument("--hardware", action="store_true",
                       help="Show hardware information")
    parser.add_argument("--config", "-c", default="/etc/thermal_guardian.conf",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    monitor = ThermalStatusMonitor(args.config)
    
    try:
        if args.hardware:
            monitor.show_hardware_info()
        elif args.logs:
            monitor.show_log_tail(args.log_lines)
        elif args.watch:
            monitor.watch_temperatures(args.interval)
        elif args.detailed:
            monitor.show_detailed_status()
        else:
            monitor.show_brief_status()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())