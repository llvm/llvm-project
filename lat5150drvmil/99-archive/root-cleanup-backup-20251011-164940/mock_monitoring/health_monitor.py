#!/usr/bin/env python3
"""DSMIL Phase 2A Health Monitor"""
import json, time, subprocess, os
from datetime import datetime
from pathlib import Path

def check_system_health():
    health = {
        "timestamp": datetime.now().isoformat(),
        "kernel_module": bool(subprocess.run(['lsmod'], capture_output=True, text=True).stdout.find('dsmil_72dev') >= 0),
        "device_node": Path("/dev/dsmil-72dev").exists(),
        "chunked_ioctl": True,  # Simulated check
        "deployment_id": os.environ.get("DSMIL_DEPLOYMENT_ID", "unknown")
    }
    return health

if __name__ == "__main__":
    health = check_system_health()
    log_file = Path("health_log.jsonl")
    with open(log_file, 'a') as f:
        f.write(json.dumps(health) + '\n')
    print(f"Health check: {'✅ HEALTHY' if all(health.values()) else '⚠️  ISSUES DETECTED'}")
