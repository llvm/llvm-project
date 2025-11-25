#!/usr/bin/env python3
"""
DSMIL Phase 1 Monitoring Dashboard
Real-time monitoring of 29 safe devices
"""

import asyncio
import subprocess
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys
import os

# Add backend to path
sys.path.append('/home/john/LAT5150DRVMIL/web-interface/backend')
from expanded_safe_devices import (
    SAFE_MONITORING_DEVICES, QUARANTINED_DEVICES,
    get_device_risk_assessment, get_monitoring_expansion_plan
)

class Phase1MonitoringDashboard:
    """Real-time monitoring dashboard for Phase 1 devices"""
    
    def __init__(self):
        self.password = "1786"
        self.monitoring_active = True
        self.device_stats = {}
        self.start_time = datetime.now()
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        
        # Initialize device stats
        for device_id in SAFE_MONITORING_DEVICES:
            self.device_stats[device_id] = {
                "name": SAFE_MONITORING_DEVICES[device_id]["name"],
                "status": SAFE_MONITORING_DEVICES[device_id]["status"],
                "confidence": SAFE_MONITORING_DEVICES[device_id]["confidence"],
                "last_read": None,
                "last_status": None,
                "read_count": 0,
                "error_count": 0,
                "avg_response_ms": 0
            }
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_thermal_status(self) -> Tuple[float, str]:
        """Get current thermal status"""
        try:
            cmd = f'echo "{self.password}" | sudo -S sensors | grep "Core 0" | head -1'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
            
            if result.stdout:
                # Extract temperature
                temp_str = result.stdout.split('+')[1].split('°C')[0].strip()
                temp = float(temp_str)
                
                if temp < 75:
                    status = "🟢 OPTIMAL"
                elif temp < 85:
                    status = "🟡 NORMAL"
                elif temp < 95:
                    status = "🟠 WARNING"
                else:
                    status = "🔴 CRITICAL"
                
                return temp, status
        except:
            return 0.0, "❓ UNKNOWN"
    
    def read_device_status(self, device_id: int) -> Tuple[bool, int, int]:
        """Read device status via SMI"""
        start_time = time.time()
        
        smi_code = f"""
#include <stdio.h>
#include <sys/io.h>
#include <unistd.h>

int main() {{
    if (iopl(3) != 0) return 1;
    outw(0x{device_id:04X}, 0x164E);
    unsigned char status = inb(0x164F);
    printf("%d\\n", status);
    return 0;
}}
"""
        
        try:
            # Write and compile
            with open("/tmp/monitor_smi.c", "w") as f:
                f.write(smi_code)
            
            subprocess.run("gcc -O0 -o /tmp/monitor_smi /tmp/monitor_smi.c", 
                         shell=True, check=True, capture_output=True)
            
            # Execute
            cmd = f'echo "{self.password}" | sudo -S /tmp/monitor_smi'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            
            response_time = int((time.time() - start_time) * 1000)
            
            if result.stdout:
                status = int(result.stdout.strip())
                return True, status, response_time
            
            return False, 0, response_time
            
        except:
            return False, 0, int((time.time() - start_time) * 1000)
        finally:
            subprocess.run("rm -f /tmp/monitor_smi /tmp/monitor_smi.c", shell=True)
    
    def display_dashboard(self):
        """Display monitoring dashboard"""
        self.clear_screen()
        
        # Header
        print("=" * 80)
        print("DSMIL PHASE 1 MONITORING DASHBOARD".center(80))
        print("=" * 80)
        
        # System info
        uptime = datetime.now() - self.start_time
        thermal_temp, thermal_status = self.get_thermal_status()
        
        print(f"Uptime: {uptime}  |  Temp: {thermal_temp:.1f}°C {thermal_status}")
        print(f"Operations: {self.total_operations}  |  Success: {self.successful_operations}  |  Failed: {self.failed_operations}")
        
        if self.total_operations > 0:
            success_rate = (self.successful_operations / self.total_operations) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print("-" * 80)
        
        # Device groups
        print("\n📊 DEVICE STATUS BY GROUP\n")
        
        # Group devices by category
        groups = {
            "Core Monitoring (100% confidence)": [],
            "Security (65-90% confidence)": [],
            "Network (65-90% confidence)": [],
            "Training (50-60% confidence)": []
        }
        
        for device_id, stats in self.device_stats.items():
            if device_id <= 0x8006:
                groups["Core Monitoring (100% confidence)"].append((device_id, stats))
            elif device_id <= 0x801B:
                groups["Security (65-90% confidence)"].append((device_id, stats))
            elif device_id <= 0x802B:
                groups["Network (65-90% confidence)"].append((device_id, stats))
            else:
                groups["Training (50-60% confidence)"].append((device_id, stats))
        
        # Display each group
        for group_name, devices in groups.items():
            if not devices:
                continue
                
            print(f"\n{group_name}:")
            print("-" * 60)
            
            for device_id, stats in devices:
                status_icon = "✅" if stats["last_status"] else "⚪"
                error_icon = "❌" if stats["error_count"] > 0 else ""
                
                print(f"  0x{device_id:04X} {status_icon} {stats['name'][:30]:30} "
                      f"Reads: {stats['read_count']:3} "
                      f"Avg: {stats['avg_response_ms']:3}ms {error_icon}")
        
        # Quarantine reminder
        print("\n" + "=" * 80)
        print("⚠️  QUARANTINED DEVICES (NEVER ACCESS):")
        for device_id, name in QUARANTINED_DEVICES.items():
            print(f"  🔴 0x{device_id:04X}: {name}")
        
        # Phase progress
        plan = get_monitoring_expansion_plan()
        print("\n" + "=" * 80)
        print("📈 PHASE 1 PROGRESS:")
        print(f"  Duration: {plan['phase_1']['duration']}")
        print(f"  Devices: {plan['phase_1']['device_count']} devices")
        print(f"  Status: {plan['phase_1']['status']}")
        
        # Commands
        print("\n" + "=" * 80)
        print("Commands: [Q]uit | [P]ause | [R]esume | [S]can All | [T]hermal Check")
        print("=" * 80)
    
    async def monitor_device(self, device_id: int):
        """Monitor a single device"""
        success, status, response_time = self.read_device_status(device_id)
        
        self.total_operations += 1
        stats = self.device_stats[device_id]
        stats["read_count"] += 1
        
        if success:
            self.successful_operations += 1
            stats["last_status"] = status
            stats["last_read"] = datetime.now()
            
            # Update average response time
            if stats["avg_response_ms"] == 0:
                stats["avg_response_ms"] = response_time
            else:
                stats["avg_response_ms"] = (stats["avg_response_ms"] + response_time) // 2
        else:
            self.failed_operations += 1
            stats["error_count"] += 1
    
    async def scan_all_devices(self):
        """Scan all monitored devices"""
        print("\nScanning all devices...")
        
        for device_id in SAFE_MONITORING_DEVICES:
            if device_id in QUARANTINED_DEVICES:
                continue
            
            await self.monitor_device(device_id)
            await asyncio.sleep(0.1)  # Small delay between devices
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        cycle_count = 0
        
        while self.monitoring_active:
            cycle_count += 1
            
            # Display dashboard
            self.display_dashboard()
            
            # Periodic full scan (every 10 cycles)
            if cycle_count % 10 == 0:
                await self.scan_all_devices()
            else:
                # Monitor subset of devices each cycle
                device_subset = list(SAFE_MONITORING_DEVICES.keys())[:6]
                for device_id in device_subset:
                    if device_id not in QUARANTINED_DEVICES:
                        await self.monitor_device(device_id)
            
            # Check for user input (non-blocking)
            await asyncio.sleep(5)  # Update every 5 seconds
    
    async def run(self):
        """Run the monitoring dashboard"""
        print("Starting DSMIL Phase 1 Monitoring Dashboard...")
        print("Initializing...")
        
        # Initial scan
        await self.scan_all_devices()
        
        # Start monitoring loop
        try:
            await self.monitoring_loop()
        except KeyboardInterrupt:
            print("\n\nShutting down monitoring dashboard...")
            self.monitoring_active = False
        
        # Final summary
        print("\n" + "=" * 80)
        print("MONITORING SESSION SUMMARY")
        print("=" * 80)
        print(f"Total Operations: {self.total_operations}")
        print(f"Successful: {self.successful_operations}")
        print(f"Failed: {self.failed_operations}")
        
        if self.total_operations > 0:
            success_rate = (self.successful_operations / self.total_operations) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nPhase 1 monitoring complete.")

def main():
    """Main entry point"""
    dashboard = Phase1MonitoringDashboard()
    asyncio.run(dashboard.run())

if __name__ == "__main__":
    main()