#!/usr/bin/env python3
"""
DSMIL Emergency Stop and Safety System
Dell Latitude 5450 MIL-SPEC - Emergency Response Framework

CRITICAL SAFETY FUNCTIONS:
- Immediate emergency stop of all DSMIL monitoring
- Detection of dangerous device activation patterns
- System resource protection and recovery
- Security incident response procedures
- Safe system shutdown protocols

Author: MONITOR Agent
Date: 2025-09-01
Classification: MIL-SPEC Safety Critical
"""

import os
import sys
import time
import json
import signal
import psutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import argparse

# ============================================================================
# EMERGENCY STOP CONSTANTS
# ============================================================================

# DSMIL Token Ranges
DSMIL_TOKEN_START = 0x8000
DSMIL_TOKEN_END = 0x806B
DANGEROUS_TOKENS = [0x8009, 0x800A, 0x800B]  # Wipe/destruction devices

# Emergency Thresholds
EMERGENCY_TEMP_LIMIT = 90   # °C
EMERGENCY_MEMORY_LIMIT = 95 # %
EMERGENCY_CPU_LIMIT = 95    # %
EMERGENCY_DISK_IO_LIMIT = 500  # MB/s

# Process Names to Monitor/Stop
DSMIL_PROCESSES = [
    "dsmil_readonly_monitor.py",
    "dsmil_comprehensive_monitor.py", 
    "safe_token_tester.py",
    "dsmil-72dev",
    "milspec-control"
]

# Kernel Modules to Check/Remove
DSMIL_MODULES = [
    "dsmil-72dev",
    "dell-milspec",
    "dell-milspec-enhanced"
]

class EmergencyLevel:
    """Emergency response levels"""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

class DSMILEmergencyStop:
    """
    DSMIL Emergency Stop System
    
    Provides immediate response to dangerous situations:
    - System resource exhaustion
    - Dangerous DSMIL token activation
    - Thermal emergencies
    - Manual emergency stop requests
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.emergency_log = f"dsmil_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.ensure_log_directory()
        
        # Emergency state tracking
        self.emergency_active = False
        self.emergency_reason = ""
        self.emergency_data = {}
        
        # System state before emergency
        self.baseline_system_state = self.capture_system_state()
        
        self.log_event(EmergencyLevel.INFO, "EMERGENCY_SYSTEM_INIT", 
                      "Emergency stop system initialized", {})
    
    def ensure_log_directory(self):
        """Ensure emergency log directory exists"""
        log_dir = "/tmp"  # Use /tmp for emergency logs (always writable)
        self.emergency_log = os.path.join(log_dir, os.path.basename(self.emergency_log))
    
    def log_event(self, level: int, event_type: str, description: str, data: Dict):
        """Log emergency events with timestamp"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "type": event_type,
            "description": description,
            "data": data
        }
        
        try:
            with open(self.emergency_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"❌ EMERGENCY LOG ERROR: {e}")
            # Fallback to stdout
            print(f"EMERGENCY: {log_entry}")
    
    def capture_system_state(self) -> Dict:
        """Capture current system state for comparison"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "processes": len(psutil.pids()),
                "temperature": self.get_cpu_temperature(),
                "disk_io": self.get_disk_io_rate(),
                "running_dsmil_processes": self.find_dsmil_processes(),
                "loaded_dsmil_modules": self.check_dsmil_modules()
            }
            return state
        except Exception as e:
            self.log_event(EmergencyLevel.WARNING, "STATE_CAPTURE_ERROR",
                          f"Failed to capture system state: {e}", {"error": str(e)})
            return {}
    
    def get_cpu_temperature(self) -> float:
        """Get CPU temperature safely"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max([sensor.current for sensor in temps['coretemp']])
            elif temps:
                all_temps = [sensor.current for sensors in temps.values() for sensor in sensors]
                return max(all_temps) if all_temps else 0.0
            return 0.0
        except:
            return 0.0
    
    def get_disk_io_rate(self) -> float:
        """Get disk I/O rate in MB/s"""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024
            return 0.0
        except:
            return 0.0
    
    def find_dsmil_processes(self) -> List[Dict]:
        """Find all running DSMIL-related processes"""
        dsmil_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
                try:
                    proc_info = proc.info
                    cmdline = ' '.join(proc_info['cmdline']) if proc_info['cmdline'] else ''
                    
                    # Check if this is a DSMIL-related process
                    is_dsmil = False
                    for process_name in DSMIL_PROCESSES:
                        if (process_name in proc_info['name'] or 
                            process_name in cmdline or
                            'dsmil' in cmdline.lower()):
                            is_dsmil = True
                            break
                    
                    if is_dsmil:
                        dsmil_processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'cmdline': cmdline,
                            'status': proc_info['status']
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.log_event(EmergencyLevel.WARNING, "PROCESS_SCAN_ERROR",
                          f"Error scanning processes: {e}", {"error": str(e)})
        
        return dsmil_processes
    
    def check_dsmil_modules(self) -> List[str]:
        """Check for loaded DSMIL kernel modules"""
        loaded_modules = []
        
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    for module in DSMIL_MODULES:
                        if module in line:
                            loaded_modules.append(module)
        except Exception as e:
            self.log_event(EmergencyLevel.WARNING, "MODULE_CHECK_ERROR",
                          f"Error checking modules: {e}", {"error": str(e)})
        
        return loaded_modules
    
    def terminate_dsmil_processes(self) -> bool:
        """Terminate all DSMIL-related processes"""
        print("🛑 Terminating DSMIL processes...")
        
        processes = self.find_dsmil_processes()
        terminated_count = 0
        
        for proc_info in processes:
            try:
                pid = proc_info['pid']
                proc = psutil.Process(pid)
                
                print(f"   🔹 Terminating PID {pid}: {proc_info['name']}")
                
                # Try graceful termination first
                proc.terminate()
                
                # Wait up to 3 seconds for graceful termination
                try:
                    proc.wait(timeout=3)
                    terminated_count += 1
                    print(f"   ✅ Process {pid} terminated gracefully")
                except psutil.TimeoutExpired:
                    # Force kill if necessary
                    print(f"   ⚠️  Force killing PID {pid}")
                    proc.kill()
                    terminated_count += 1
                    print(f"   ✅ Process {pid} force killed")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"   ❌ Could not terminate PID {proc_info['pid']}: {e}")
        
        self.log_event(EmergencyLevel.CRITICAL, "PROCESS_TERMINATION",
                      f"Terminated {terminated_count} DSMIL processes",
                      {"terminated": terminated_count, "found": len(processes)})
        
        return terminated_count > 0
    
    def unload_dsmil_modules(self) -> bool:
        """Unload DSMIL kernel modules"""
        print("🔧 Unloading DSMIL kernel modules...")
        
        modules = self.check_dsmil_modules()
        if not modules:
            print("   ✅ No DSMIL modules currently loaded")
            return True
        
        unloaded_count = 0
        for module in modules:
            try:
                print(f"   🔹 Unloading module: {module}")
                result = subprocess.run(['sudo', 'rmmod', module], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    unloaded_count += 1
                    print(f"   ✅ Module {module} unloaded successfully")
                else:
                    print(f"   ⚠️  Failed to unload {module}: {result.stderr}")
                    
            except Exception as e:
                print(f"   ❌ Error unloading {module}: {e}")
        
        self.log_event(EmergencyLevel.CRITICAL, "MODULE_UNLOAD",
                      f"Unloaded {unloaded_count} DSMIL modules",
                      {"unloaded": unloaded_count, "found": len(modules)})
        
        return unloaded_count > 0
    
    def check_system_safety(self) -> tuple[bool, List[str]]:
        """Check if system is in safe state"""
        safety_issues = []
        
        try:
            # Check temperature
            temp = self.get_cpu_temperature()
            if temp > EMERGENCY_TEMP_LIMIT:
                safety_issues.append(f"CPU temperature critical: {temp:.1f}°C")
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > EMERGENCY_MEMORY_LIMIT:
                safety_issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > EMERGENCY_CPU_LIMIT:
                safety_issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            
            # Check disk I/O
            disk_io_rate = self.get_disk_io_rate()
            if disk_io_rate > EMERGENCY_DISK_IO_LIMIT:
                safety_issues.append(f"Disk I/O critical: {disk_io_rate:.1f} MB/s")
            
            # Check for DSMIL processes
            dsmil_processes = self.find_dsmil_processes()
            if dsmil_processes:
                safety_issues.append(f"Active DSMIL processes: {len(dsmil_processes)}")
            
            # Check for loaded modules
            dsmil_modules = self.check_dsmil_modules()
            if dsmil_modules:
                safety_issues.append(f"Loaded DSMIL modules: {', '.join(dsmil_modules)}")
            
        except Exception as e:
            safety_issues.append(f"Safety check error: {e}")
        
        is_safe = len(safety_issues) == 0
        return is_safe, safety_issues
    
    def execute_emergency_stop(self, reason: str, data: Dict = None) -> bool:
        """Execute full emergency stop procedure"""
        if data is None:
            data = {}
        
        print("🚨" * 20)
        print("🚨 DSMIL EMERGENCY STOP ACTIVATED 🚨")
        print("🚨" * 20)
        print(f"Reason: {reason}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.emergency_active = True
        self.emergency_reason = reason
        self.emergency_data = data
        
        self.log_event(EmergencyLevel.EMERGENCY, "EMERGENCY_STOP_START",
                      f"Emergency stop activated: {reason}", data)
        
        # Step 1: Terminate DSMIL processes
        processes_terminated = self.terminate_dsmil_processes()
        
        # Step 2: Unload kernel modules  
        modules_unloaded = self.unload_dsmil_modules()
        
        # Step 3: Capture final system state
        final_state = self.capture_system_state()
        
        # Step 4: Safety verification
        is_safe, safety_issues = self.check_system_safety()
        
        # Log emergency completion
        completion_data = {
            "processes_terminated": processes_terminated,
            "modules_unloaded": modules_unloaded,
            "final_state": final_state,
            "safety_issues": safety_issues,
            "is_safe": is_safe
        }
        
        self.log_event(EmergencyLevel.EMERGENCY, "EMERGENCY_STOP_COMPLETE",
                      "Emergency stop procedure completed", completion_data)
        
        # Display results
        print("\n📋 Emergency Stop Results:")
        print(f"   🛑 Processes terminated: {'✅' if processes_terminated else '❌'}")
        print(f"   🔧 Modules unloaded: {'✅' if modules_unloaded else '❌'}")
        print(f"   🛡️  System safe: {'✅' if is_safe else '❌'}")
        
        if safety_issues:
            print(f"\n⚠️  Remaining safety issues:")
            for issue in safety_issues:
                print(f"   ⚠️  {issue}")
        
        print(f"\n📄 Emergency log: {self.emergency_log}")
        
        if not is_safe:
            print("\n🚨 WARNING: System may still be in unsafe state!")
            print("   Consider system reboot if issues persist")
            return False
        
        print("\n✅ Emergency stop completed successfully")
        print("   System is now in safe state")
        return True
    
    def monitor_for_emergencies(self, duration_seconds: int = 300) -> bool:
        """Monitor system for emergency conditions"""
        print(f"👁️  Monitoring for emergency conditions ({duration_seconds}s)...")
        
        start_time = time.time()
        check_interval = 2.0  # Check every 2 seconds
        last_check = start_time
        
        try:
            while (time.time() - start_time) < duration_seconds:
                current_time = time.time()
                
                if (current_time - last_check) >= check_interval:
                    # Check for emergency conditions
                    is_safe, safety_issues = self.check_system_safety()
                    
                    if not is_safe:
                        # Determine if this is a true emergency
                        critical_issues = [issue for issue in safety_issues 
                                         if any(word in issue.lower() 
                                               for word in ['critical', 'emergency'])]
                        
                        if critical_issues:
                            print(f"\n🚨 EMERGENCY CONDITIONS DETECTED:")
                            for issue in critical_issues:
                                print(f"   🔴 {issue}")
                            
                            # Auto-trigger emergency stop
                            emergency_data = {
                                "auto_trigger": True,
                                "conditions": critical_issues,
                                "monitoring_duration": current_time - start_time
                            }
                            
                            return self.execute_emergency_stop(
                                "Automatic emergency - critical conditions detected",
                                emergency_data
                            )
                    
                    last_check = current_time
                
                # Brief sleep
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Emergency monitoring stopped by user")
            return self.execute_emergency_stop("User requested emergency stop", 
                                               {"monitoring_duration": time.time() - start_time})
        
        print(f"✅ Emergency monitoring completed - no emergencies detected")
        return True

def main():
    """Main entry point for emergency stop system"""
    parser = argparse.ArgumentParser(
        description="DSMIL Emergency Stop and Safety System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EMERGENCY STOP MODES:
  --stop              Execute immediate emergency stop
  --monitor           Monitor for emergency conditions
  --check             Check system safety status
  --status            Show current system status

EXAMPLES:
  sudo python3 dsmil_emergency_stop.py --stop
  sudo python3 dsmil_emergency_stop.py --monitor --duration 300
  sudo python3 dsmil_emergency_stop.py --check
        """
    )
    
    parser.add_argument("--stop", action="store_true",
                       help="Execute immediate emergency stop")
    
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor for emergency conditions")
    
    parser.add_argument("--check", action="store_true",
                       help="Check current system safety status")
    
    parser.add_argument("--status", action="store_true",
                       help="Show detailed system status")
    
    parser.add_argument("--duration", type=int, default=300,
                       help="Monitoring duration in seconds (default: 300)")
    
    parser.add_argument("--reason", type=str, default="Manual emergency stop",
                       help="Reason for emergency stop")
    
    args = parser.parse_args()
    
    # Create emergency stop system
    emergency = DSMILEmergencyStop()
    
    try:
        if args.stop:
            # Execute immediate emergency stop
            success = emergency.execute_emergency_stop(args.reason)
            sys.exit(0 if success else 1)
            
        elif args.monitor:
            # Monitor for emergency conditions
            success = emergency.monitor_for_emergencies(args.duration)
            sys.exit(0 if success else 1)
            
        elif args.check:
            # Check system safety
            is_safe, issues = emergency.check_system_safety()
            
            print("🛡️  System Safety Check:")
            print(f"   Status: {'✅ SAFE' if is_safe else '⚠️  UNSAFE'}")
            
            if issues:
                print("   Issues found:")
                for issue in issues:
                    print(f"     ⚠️  {issue}")
            else:
                print("   No safety issues detected")
            
            sys.exit(0 if is_safe else 1)
            
        elif args.status:
            # Show detailed system status
            state = emergency.capture_system_state()
            
            print("📊 DSMIL System Status:")
            print(f"   CPU: {state.get('cpu_percent', 'N/A'):.1f}%")
            print(f"   Memory: {state.get('memory_percent', 'N/A'):.1f}%")
            print(f"   Temperature: {state.get('temperature', 'N/A'):.1f}°C")
            print(f"   Processes: {state.get('processes', 'N/A')}")
            print(f"   DSMIL processes: {len(state.get('running_dsmil_processes', []))}")
            print(f"   DSMIL modules: {len(state.get('loaded_dsmil_modules', []))}")
            
            sys.exit(0)
            
        else:
            # No specific action - show help
            parser.print_help()
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n🛑 Emergency stop interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ EMERGENCY SYSTEM ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()