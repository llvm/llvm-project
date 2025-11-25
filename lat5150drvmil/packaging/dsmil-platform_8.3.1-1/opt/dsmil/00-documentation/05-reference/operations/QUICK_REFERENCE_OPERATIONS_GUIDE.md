# DSMIL Control System - Quick Reference Operations Guide

**⚡ EMERGENCY STOP:** `Ctrl+C` in any script or dashboard  
**🔴 NEVER ACCESS:** Devices 0x8009, 0x800A, 0x800B, 0x8019, 0x8029  

---

## 🚀 Quick Start Commands

### Daily Operations
```bash
# Start monitoring dashboard
python3 /home/john/LAT5150DRVMIL/phase1_monitoring_dashboard.py

# Test all safe devices
python3 /home/john/LAT5150DRVMIL/test_phase1_safe_devices.py

# Check system status
python3 -c "from web_interface.backend.expanded_safe_devices import *; print(f'Safe: {len(SAFE_MONITORING_DEVICES)}, Quarantined: {len(QUARANTINED_DEVICES)}')"
```

### Device Operations
```bash
# Check if device is safe
python3 -c "from web_interface.backend.expanded_safe_devices import *; print(get_device_risk_assessment(0x8010))"

# View monitoring plan
python3 -c "from web_interface.backend.expanded_safe_devices import *; import json; print(json.dumps(get_monitoring_expansion_plan()['phase_1'], indent=2))"

# List all safe devices
python3 -c "from web_interface.backend.expanded_safe_devices import *; [print(f'0x{d:04X}: {SAFE_MONITORING_DEVICES[d][\"name\"]}') for d in sorted(SAFE_MONITORING_DEVICES.keys())]"
```

---

## 🔒 Critical Safety Information

### ⛔ QUARANTINED DEVICES - NEVER ACCESS
| Device | Function | Action if Accessed |
|--------|----------|-------------------|
| **0x8009** | Emergency Wipe Controller | IMMEDIATE DATA DESTRUCTION |
| **0x800A** | Secondary Wipe Trigger | CASCADE WIPE ACTIVATION |
| **0x800B** | Final Sanitization | COMPLETE SYSTEM WIPE |
| **0x8019** | Network Isolation/Wipe | NETWORK DESTRUCTION |
| **0x8029** | Communications Blackout | TOTAL COMMS LOSS |

### ✅ Safe Monitoring Devices (29 total)

#### Core Monitoring (6 devices - 100% safe)
```
0x8000: Group 0 Controller     0x8003: Fan Control
0x8001: Thermal Monitoring     0x8004: CPU Status
0x8002: Power Status           0x8006: System Supervisor
```

#### Security (5 devices - 65-90% safe)
```
0x8007: Security Audit Logger         0x8015: Certificate Authority
0x8010: Multi-Factor Authentication   0x8016: Security Baseline Monitor
0x8012: Security Event Correlator
```

#### Network (6 devices - 65-90% safe)
```
0x8020: Network Interface Controller  0x8024: VPN Hardware Accelerator
0x8021: Wireless Communication        0x8025: Network QoS
0x8023: Network Performance Monitor
```

#### Training (12 devices - 50-60% safe)
```
0x8060-0x8063: Scenario Controllers
0x8064-0x8067: Data Collection
0x8068-0x806B: Environment Control
```

---

## 📊 System Status Checks

### Thermal Monitoring
```bash
# Check current temperature
echo "1786" | sudo -S sensors | grep "Core 0"

# Safe: <75°C  |  Normal: 75-85°C  |  Warning: 85-95°C  |  Critical: >95°C
```

### Performance Metrics
```bash
# View current metrics from dashboard
# Or check last report:
cat phase1_activation_report_*.json | jq '.coverage'
```

### Error Checking
```bash
# Check for any errors in logs
grep -i "error\|fail\|critical" phase1_activation_*.log | tail -20
```

---

## 🛠️ Troubleshooting

### Dashboard Not Starting
```bash
# Check Python path
python3 --version  # Should be 3.8+

# Verify files exist
ls -la /home/john/LAT5150DRVMIL/*.py

# Try manual start
cd /home/john/LAT5150DRVMIL
python3 phase1_monitoring_dashboard.py
```

### Device Not Responding
```bash
# Test specific device
python3 << EOF
from test_phase1_safe_devices import read_device_smi
success, status = read_device_smi(0x8020)
print(f"Success: {success}, Status: 0x{status:02X}")
EOF
```

### High Temperature Warning
```bash
# 1. Stop operations immediately
Ctrl+C

# 2. Check thermal status
sensors

# 3. Wait for cooling
# 4. Resume when <85°C
```

---

## 📋 Phase 1 Daily Checklist

### Morning (Start of Day)
- [ ] Check thermal status (<85°C)
- [ ] Start monitoring dashboard
- [ ] Verify all 29 devices responding
- [ ] Check overnight logs for errors
- [ ] Note any anomalies

### Afternoon (Mid-Day)
- [ ] Review performance metrics
- [ ] Check thermal trends
- [ ] Run device test suite
- [ ] Document any discoveries

### Evening (End of Day)
- [ ] Generate daily report
- [ ] Archive logs
- [ ] Plan next day activities
- [ ] Update phase progress

---

## 📈 Expansion Schedule

### Current: Phase 1 (Days 1-30)
**Focus:** Monitor 29 safe devices
```bash
# Devices active now
python3 -c "from web_interface.backend.expanded_safe_devices import *; print(f'Active: {len([d for d in SAFE_MONITORING_DEVICES.values() if d[\"status\"] == \"ACTIVE\"])}')"
```

### Next: Phase 2 (Days 31-60)
**Target:** Add 7 devices (TPM, Secure Boot, Encryption)
```
0x8005: TPM/HSM Interface
0x8008: Secure Boot Validator
0x8011: Encryption Key Management
0x8013: Intrusion Detection System
0x8014: Security Policy Enforcement
0x8022: Network Security Filter
0x8027: Network Authentication Gateway
```

---

## 🔧 API Quick Reference

### Backend Server
```bash
# Start backend (if needed)
cd /home/john/LAT5150DRVMIL/web-interface/backend
python3 main.py

# API Base URL: http://localhost:8000
```

### Key Endpoints
```
GET  /api/v1/devices              # List all devices
GET  /api/v1/devices/safe         # List safe devices only
GET  /api/v1/devices/{id}/status  # Get device status
POST /api/v1/devices/{id}/read    # Read device (safe only)
GET  /api/v1/monitoring/metrics   # System metrics
```

### WebSocket Monitoring
```
ws://localhost:8000/ws/monitoring  # Real-time updates
```

---

## 📞 Emergency Procedures

### 1. Suspected Quarantine Access
```bash
# IMMEDIATE ACTIONS:
1. Press Ctrl+C to stop all operations
2. DO NOT attempt any device operations
3. Check logs for device access:
   grep "0x8009\|0x800A\|0x800B\|0x8019\|0x8029" *.log
4. If confirmed, DO NOT PROCEED
5. Consult safety documentation
```

### 2. System Freeze
```bash
# Recovery steps:
1. Wait 30 seconds for timeout
2. If still frozen, open new terminal
3. Kill monitoring process:
   pkill -f phase1_monitoring
4. Check system status
5. Restart with safety checks
```

### 3. Thermal Critical (>95°C)
```bash
# Cooling protocol:
1. Stop all operations immediately
2. Check fan status
3. Reduce system load
4. Monitor temperature decline
5. Resume only when <85°C
```

---

## 📝 Reporting Templates

### Daily Status Report
```
Date: [DATE]
Phase: 1 (Day X of 30)
Devices Monitored: 29
Uptime: XX hours
Success Rate: XX%
Thermal Max: XX°C
Errors: X
Notes: [Any observations]
```

### Issue Report
```
Time: [TIMESTAMP]
Device: 0x[XXXX]
Operation: [READ/STATUS]
Error: [Description]
Impact: [None/Minor/Major]
Action: [Resolution taken]
```

---

## 🔗 Important Files

### Core Scripts
- `test_phase1_safe_devices.py` - Device testing
- `phase1_monitoring_dashboard.py` - Live monitoring
- `activate_phase1_production.sh` - Production activation

### Configuration
- `expanded_safe_devices.py` - Device registry
- `config.py` - System configuration

### Documentation
- `MASTER_DOCUMENTATION_INDEX.md` - Complete docs index
- `PROJECT_COMPLETE_SUMMARY.md` - Project overview
- `PHASE_1_DEVICE_EXPANSION_DOCUMENTATION.md` - Current phase

---

## ⚡ Quick Decision Tree

```
Need to check device safety?
├─ YES → Check expanded_safe_devices.py
│   ├─ Quarantined? → NEVER ACCESS
│   ├─ Safe list? → READ ONLY
│   └─ Unknown? → DO NOT ACCESS
└─ NO → Continue

Temperature > 85°C?
├─ YES → Monitor closely
│   └─ > 95°C? → STOP IMMEDIATELY
└─ NO → Continue normal ops

Error occurred?
├─ YES → Check if quarantine device
│   ├─ YES → EMERGENCY STOP
│   └─ NO → Log and continue
└─ NO → Continue

End of day?
├─ YES → Generate report
└─ NO → Continue monitoring
```

---

## 💡 Pro Tips

1. **Always check thermal first** - High temps affect reliability
2. **Never skip safety checks** - Even for "quick tests"
3. **Document anomalies immediately** - Patterns emerge over time
4. **Use dashboard for monitoring** - Manual checks are error-prone
5. **Backup before changes** - Even config changes
6. **Read logs daily** - Early warning signs appear there
7. **Follow the phases** - Rushing increases risk
8. **When in doubt, don't** - Safety over speed

---

**Remember:** 5 devices can destroy everything. 29 devices are safe. 50 devices are unknown.  
**Goal:** Safe, systematic expansion to 94% coverage over 150 days.  
**Current:** Day 1 of 150 - Phase 1 Active

---

*Keep this guide handy during all operations. Updated: September 2, 2025*