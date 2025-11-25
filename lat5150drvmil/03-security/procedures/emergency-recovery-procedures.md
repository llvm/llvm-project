# DSMIL Emergency Recovery Procedures
## Dell Latitude 5450 MIL-SPEC Emergency Response Guide

### IMMEDIATE RESPONSE ACTIONS

---

## 🚨 LEVEL 1: IMMEDIATE STOP (0-30 seconds)

### If System Becomes Unresponsive
```bash
# Press Ctrl+C in monitoring terminal
# If no response, try Ctrl+Z then kill %1

# Open new terminal (Ctrl+Alt+T) and execute:
sudo killall -9 python3
sudo killall -9 dsmil-72dev
sudo rmmod dsmil-72dev
```

### Emergency Module Removal
```bash
# Force remove DSMIL kernel module
echo 1786 | sudo -S rmmod dsmil-72dev 2>/dev/null

# Verify removal
lsmod | grep dsmil
# Should show no results

# Check for hanging processes
ps aux | grep -E "(dsmil|monitor)"
sudo kill -9 [PID_IF_FOUND]
```

---

## 🔥 LEVEL 2: THERMAL EMERGENCY (30-60 seconds)

### If Temperature Exceeds 95°C
```bash
# Immediate system thermal throttling
echo 1786 | sudo -S cpufreq-set -g powersave
echo 1786 | sudo -S echo 1 > /sys/class/thermal/cooling_device*/cur_state

# Stop all DSMIL operations
sudo rmmod dsmil-72dev
sudo killall -9 python3

# Monitor temperature drop
watch -n1 "cat /sys/class/thermal/thermal_zone*/temp | awk '{print \$1/1000 \"°C\"}'"

# If temperature doesn't drop in 30 seconds: REBOOT IMMEDIATELY
sudo reboot
```

### Thermal Recovery Actions
1. **Immediate**: Stop all CPU-intensive processes
2. **Within 1 minute**: Verify temperature below 85°C
3. **Wait 5 minutes**: Allow complete thermal recovery
4. **Verify**: Check system logs before resuming

---

## 💻 LEVEL 3: SYSTEM FREEZE/HANG (60-120 seconds)

### If System Stops Responding
```bash
# Try Alt+SysRq+R (Raw keyboard mode)
# Try Alt+SysRq+E (Terminate processes)
# Try Alt+SysRq+I (Kill processes)
# Try Alt+SysRq+S (Sync disks)
# Try Alt+SysRq+U (Unmount filesystems)
# Try Alt+SysRq+B (Reboot)
```

### If Magic SysRq Doesn't Work
1. **Wait 2 minutes** for potential recovery
2. **Power button press** (1 second - soft shutdown)
3. **Power button hold** (10 seconds - hard shutdown)
4. **Remove power** if laptop doesn't respond

---

## 📊 LEVEL 4: POST-RECOVERY VALIDATION

### System Health Check
```bash
# Boot system and immediately check:
dmesg | tail -50                    # Check for error messages
cat /proc/meminfo                   # Verify memory state
lscpu                              # Check CPU status
sensors                            # Verify thermal sensors
lsmod | grep -E "(dell|dsmil)"     # Ensure modules not loaded
```

### Compare Against Baseline
```bash
cd /home/john/LAT5150DRVMIL

# Extract latest baseline
tar -xzf baseline_20250901_024305.tar.gz
cd baseline_20250901_024305

# Compare critical parameters
diff system_info.txt <(dmidecode -t system)
diff thermal_baseline.txt <(sensors)
diff smbios_baseline.txt <(dmidecode | head -50)
```

### Recovery Validation Checklist
- [ ] System boots normally
- [ ] Temperature sensors reading < 60°C
- [ ] No DSMIL modules loaded (`lsmod | grep dsmil`)
- [ ] Memory usage normal (< 50%)
- [ ] No error messages in dmesg
- [ ] SMBIOS data matches baseline
- [ ] All hardware detected properly

---

## 📋 RECOVERY LOG TEMPLATE

### Document Every Recovery Event
```bash
# Create recovery log entry
cat >> /home/john/LAT5150DRVMIL/recovery.log << EOF
=== RECOVERY EVENT ===
Date: $(date -Iseconds)
Trigger: [DESCRIBE WHAT CAUSED EMERGENCY]
Recovery Level: [1/2/3/4]
Actions Taken:
- [ACTION 1]
- [ACTION 2]
- [ACTION 3]
System Status After Recovery:
- Temperature: $(sensors | grep -i cpu | head -1)
- Memory: $(free -h | grep Mem)
- Modules: $(lsmod | grep -E "(dell|dsmil)" | wc -l) DSMIL modules
Time to Recovery: [X minutes]
Notes: [ANY ADDITIONAL OBSERVATIONS]
========================
EOF
```

---

## 🔧 SPECIFIC RECOVERY SCENARIOS

### Scenario A: Memory Mapping Freeze
**Symptoms**: System hangs during memory mapping
**Recovery**:
```bash
# Immediate module removal
sudo rmmod dsmil-72dev
# Check memory mapping
cat /proc/iomem | grep -i dsmil
# Should show no mappings
```

### Scenario B: Token Read Hang
**Symptoms**: System freezes during token enumeration
**Recovery**:
```bash
# Kill monitoring processes
sudo killall -9 python3
sudo rmmod dsmil-72dev
# Check for stuck I/O operations
ps aux | grep -E "D.*dsmil"
```

### Scenario C: Thermal Runaway
**Symptoms**: Temperature climbing rapidly
**Recovery**:
```bash
# Emergency thermal management
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo rmmod dsmil-72dev
# If continues climbing: immediate reboot
sudo reboot
```

### Scenario D: ACPI Method Hang
**Symptoms**: System freeze after ACPI method call
**Recovery**:
```bash
# Force ACPI reset (if system responsive)
echo 1786 | sudo -S modprobe -r acpi_wmi
echo 1786 | sudo -S modprobe -r dell_wmi
sudo rmmod dsmil-72dev
# Wait 30 seconds, then reload
echo 1786 | sudo -S modprobe dell_wmi
echo 1786 | sudo -S modprobe acpi_wmi
```

---

## ⚠️ ESCALATION PROCEDURES

### When to Escalate
- Recovery taking > 15 minutes
- Repeated thermal emergencies
- Hardware detection failures
- Data corruption suspected
- Unknown system behavior

### Escalation Actions
1. **Document everything** in recovery log
2. **Capture system state** before more invasive recovery
3. **Consider BIOS reset** if hardware issues persist
4. **Preserve logs** for analysis

### BIOS Recovery (Last Resort)
```bash
# If system corruption suspected:
# 1. Boot to BIOS (F2 during startup)
# 2. Load Optimized Defaults
# 3. Save and Exit
# 4. Compare post-recovery state to baseline
```

---

## 🎯 SUCCESS CRITERIA

### Recovery is Complete When:
- [x] System boots normally without intervention
- [x] All hardware properly detected and functional
- [x] Temperature stable under 60°C idle
- [x] No DSMIL kernel modules loaded
- [x] System performance matches baseline
- [x] No error messages in system logs
- [x] SMBIOS data integrity verified

### Ready to Resume Testing When:
- [x] All recovery validation checks pass
- [x] System stable for 5+ minutes
- [x] Recovery event documented
- [x] Root cause identified and addressed
- [x] Additional safety measures implemented if needed

**Emergency Contact**: System Administrator (local recovery only)
**Recovery Documentation**: All procedures tested and verified
**Last Updated**: 2025-09-01 (System preparation complete)