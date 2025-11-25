#!/usr/bin/env python3
"""
DSMIL Phase 2A Health Check Script
Continuous monitoring and alerting system
"""

import os
import sys
import json
import time
import psutil
import subprocess
from datetime import datetime
from pathlib import Path

def check_system_health():
    """Comprehensive system health check"""
    health_data = {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "kernel_module_loaded": False,
        "device_node_present": False,
        "process_count": len(psutil.pids())
    }
    
    # Check kernel module
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        health_data["kernel_module_loaded"] = 'dsmil_72dev' in result.stdout
    except:
        pass
        
    # Check device node
    health_data["device_node_present"] = Path("/dev/dsmil-72dev").exists()
    
    # Check temperature (if available)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            health_data["temperature"] = max([temp.current for sensors in temps.values() for temp in sensors])
    except:
        pass
        
    return health_data

def main():
    health = check_system_health()
    
    # Log health data
    log_dir = Path("/var/log/dsmil/health")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_dir / f"health_{datetime.now().strftime('%Y%m%d')}.jsonl", 'a') as f:
        f.write(json.dumps(health) + '\n')
        
    # Check for alerts
    alerts = []
    if health["cpu_usage"] > 80:
        alerts.append(f"High CPU usage: {health['cpu_usage']:.1f}%")
    if health["memory_usage"] > 85:
        alerts.append(f"High memory usage: {health['memory_usage']:.1f}%")
    if not health["kernel_module_loaded"]:
        alerts.append("DSMIL kernel module not loaded")
    if not health["device_node_present"]:
        alerts.append("DSMIL device node missing")
        
    if alerts:
        alert_msg = "DSMIL Health Alert: " + "; ".join(alerts)
        print(alert_msg)
        
        # Log alert
        with open(log_dir / "alerts.log", 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {alert_msg}\n")

if __name__ == "__main__":
    main()
