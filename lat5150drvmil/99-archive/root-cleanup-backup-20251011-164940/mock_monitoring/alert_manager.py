#!/usr/bin/env python3
"""DSMIL Phase 2A Alert Manager"""
import json, psutil, subprocess
from datetime import datetime

def check_alerts():
    alerts = []
    
    # Check CPU usage
    cpu = psutil.cpu_percent(interval=1)
    if cpu > 80:
        alerts.append(f"High CPU usage: {cpu:.1f}%")
        
    # Check memory usage  
    memory = psutil.virtual_memory().percent
    if memory > 85:
        alerts.append(f"High memory usage: {memory:.1f}%")
        
    # Check kernel module
    result = subprocess.run(['lsmod'], capture_output=True, text=True)
    if 'dsmil_72dev' not in result.stdout:
        alerts.append("CRITICAL: DSMIL kernel module not loaded")
        
    return alerts

if __name__ == "__main__":
    alerts = check_alerts()
    if alerts:
        print("🚨 DSMIL ALERTS:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("✅ No alerts - system healthy")
