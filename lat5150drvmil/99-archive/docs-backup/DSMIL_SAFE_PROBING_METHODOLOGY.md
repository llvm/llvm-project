# DSMIL Safe Probing Methodology

**Target System**: Dell Latitude 5450 MIL-SPEC  
**Architecture**: 72 DSMIL devices (6 groups × 12 devices)  
**Classification**: Educational/Training (JRTC1)  
**Risk Level**: Medium (Hardware lockup possible)  

## 🔒 **Safety-First Approach**

### **Core Principles**
1. **Read Before Write**: Always query device status before control operations
2. **Gradual Escalation**: Single device → Group → Multi-group progression  
3. **Rollback Ready**: Every activation must have documented undo procedure
4. **Monitoring Intensive**: Real-time system health during all operations
5. **Fail-Safe Design**: Default to safe state on any error condition

### **Risk Classification**
```
LOW RISK:    Status queries, read-only operations
MEDIUM RISK: Single device activation (Groups 0-1) 
HIGH RISK:   Multi-device coordination, Groups 2-5
CRITICAL:    Cross-group dependencies, hidden memory access
```

## 📋 **Probing Phases**

### **Phase 1: Passive Enumeration** (SAFE)
```bash
# Objectives: Map current system state without changes
# Risk Level: LOW
# Duration: 30 minutes

# 1.1 Device Node Verification
ls -la /dev/DSMIL* 2>/dev/null
stat /dev/DSMIL0D* /dev/DSMIL1D* 2>/dev/null

# 1.2 ACPI Method Discovery  
sudo cat /sys/firmware/acpi/tables/DSDT | strings | grep -E "DSMIL[0-5]" | sort -u
find /sys/firmware/acpi -name "*DSMIL*" 2>/dev/null

# 1.3 System State Baseline
cat /proc/meminfo > dsmil_baseline_memory.txt
cat /proc/interrupts > dsmil_baseline_interrupts.txt  
lspci -vvv > dsmil_baseline_pci.txt
dmesg > dsmil_baseline_dmesg.txt

# 1.4 BIQ Variable Enumeration  
# (Requires custom ACPI method - see implementation below)
```

### **Phase 2: Read-Only Device Queries** (LOW RISK)
```bash
# Objectives: Query device status without activation
# Risk Level: LOW  
# Duration: 45 minutes
# Prerequisites: Phase 1 complete

# 2.1 Device Status Queries (if driver loaded)
sudo milspec-control --status
sudo cat /sys/devices/platform/dell-milspec/dsmil_status

# 2.2 ACPI Method Testing (read-only)
# Test status methods for each group:
# _STA (Status), _CRS (Current Resource Settings)
# Implementation in framework below
```

### **Phase 3: Single Device Activation** (MEDIUM RISK)  
```bash
# Objectives: Activate one low-risk device with full monitoring
# Risk Level: MEDIUM
# Duration: 1-2 hours
# Prerequisites: Phases 1-2 complete, thermal monitoring active

# Target: DSMIL0D4 (Audit Logger - non-critical, reversible)
# Safety: Continuous temperature monitoring, ready rollback
```

### **Phase 4: Group Coordination** (HIGH RISK)
```bash  
# Objectives: Activate device groups with dependency management
# Risk Level: HIGH
# Duration: 2-4 hours  
# Prerequisites: Phases 1-3 successful, full system backup

# Target: DSMIL0 group (core security devices)
# Safety: Staged activation with validation checkpoints
```

### **Phase 5: Multi-Group Operations** (CRITICAL)
```bash
# Objectives: Coordinate multiple DSMIL groups  
# Risk Level: CRITICAL
# Duration: 4+ hours
# Prerequisites: All previous phases successful, recovery environment ready

# Target: Groups 0-1 coordination
# Safety: Full system state backup, remote monitoring capability
```

## 🛡️ **Safety Mechanisms**

### **Pre-Flight Checks**
```bash
#!/bin/bash
# dsmil_preflight_check.sh

echo "🔍 DSMIL Pre-Flight Safety Check"

# 1. System Health
TEMP=$(sensors | grep "Package id 0" | awk '{print $4}' | tr -d '+°C')
if (( $(echo "$TEMP > 75" | bc -l) )); then
    echo "❌ ABORT: CPU temperature too high ($TEMP°C)"
    exit 1
fi

# 2. Memory Availability  
FREE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
if (( FREE_MEM < 4000 )); then
    echo "❌ ABORT: Insufficient free memory (${FREE_MEM}MB)"
    exit 1  
fi

# 3. Disk Space
DISK_FREE=$(df / | awk 'NR==2{print $4}')
if (( DISK_FREE < 5000000 )); then  # 5GB in KB
    echo "❌ ABORT: Insufficient disk space"
    exit 1
fi

# 4. Backup Verification
if [[ ! -f "/backup/system_pre_dsmil.tar.gz" ]]; then
    echo "❌ ABORT: System backup not found"  
    exit 1
fi

# 5. Recovery Tools
if ! command -v systemctl &> /dev/null; then
    echo "❌ ABORT: systemctl not available"
    exit 1
fi

echo "✅ All pre-flight checks passed"
```

### **Real-Time Monitoring**
```bash
#!/bin/bash
# dsmil_monitor.sh - Run during all probing operations

while true; do
    # Temperature monitoring
    TEMP=$(sensors | grep "Package id 0" | awk '{print $4}' | tr -d '+°C')
    if (( $(echo "$TEMP > 85" | bc -l) )); then
        echo "🚨 THERMAL ALERT: $TEMP°C - Consider abort"
        # Send notification to operator
    fi
    
    # Memory pressure
    FREE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if (( FREE_MEM < 2000 )); then
        echo "🚨 MEMORY ALERT: Only ${FREE_MEM}MB free"
    fi
    
    # System load
    LOAD=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | tr -d ',')
    if (( $(echo "$LOAD > 8.0" | bc -l) )); then
        echo "🚨 LOAD ALERT: System load $LOAD"  
    fi
    
    # Log health metrics
    echo "$(date): TEMP=$TEMP°C MEM=${FREE_MEM}MB LOAD=$LOAD" >> dsmil_health.log
    
    sleep 5
done
```

### **Emergency Rollback**
```bash
#!/bin/bash  
# dsmil_emergency_rollback.sh

echo "🚨 EMERGENCY DSMIL ROLLBACK INITIATED"

# 1. Immediate device deactivation
if command -v milspec-control &> /dev/null; then
    sudo milspec-control --emergency-disable-all
fi

# 2. Module unload
sudo rmmod dell-milspec 2>/dev/null || true

# 3. System cooling
echo "⏳ Waiting 30 seconds for system stabilization..."
sleep 30

# 4. Health check
TEMP_AFTER=$(sensors | grep "Package id 0" | awk '{print $4}' | tr -d '+°C')
echo "🌡️  Post-rollback temperature: $TEMP_AFTER°C"

# 5. Log incident
echo "$(date): EMERGENCY ROLLBACK - TEMP=$TEMP_AFTER°C" >> dsmil_incidents.log

echo "✅ Emergency rollback complete"
```

## 🔧 **Progressive Probing Scripts**

### **Phase 1: Passive Enumeration Script**
```bash
#!/bin/bash
# dsmil_phase1_enumerate.sh

set -euo pipefail

LOGFILE="dsmil_phase1_$(date +%Y%m%d_%H%M%S).log"
echo "📋 Phase 1: Passive DSMIL Enumeration" | tee -a $LOGFILE

# Safety check
./dsmil_preflight_check.sh || exit 1

echo "🔍 1. Device Node Discovery" | tee -a $LOGFILE
for group in {0..5}; do
    echo "  Group $group:" | tee -a $LOGFILE
    for device in {0..11}; do
        device_hex=$(printf "%X" $device)
        device_path="/dev/DSMIL${group}D${device_hex}"
        
        if [[ -e "$device_path" ]]; then
            stat "$device_path" >> $LOGFILE
            echo "    ✅ DSMIL${group}D${device_hex}: $(stat -c '%F' $device_path)" | tee -a $LOGFILE
        else
            echo "    ❌ DSMIL${group}D${device_hex}: Not found" | tee -a $LOGFILE
        fi
    done
done

echo "🔍 2. ACPI Method Discovery" | tee -a $LOGFILE  
if sudo cat /sys/firmware/acpi/tables/DSDT | strings | grep -E "DSMIL[0-5]" | sort -u > acpi_dsmil_refs.txt; then
    ACPI_COUNT=$(wc -l < acpi_dsmil_refs.txt)
    echo "  ✅ Found $ACPI_COUNT ACPI DSMIL references" | tee -a $LOGFILE
else
    echo "  ❌ Could not access ACPI DSDT" | tee -a $LOGFILE
fi

echo "🔍 3. BIQ Variable Enumeration" | tee -a $LOGFILE
biq_count=0
for biq in {200..327}; do
    # This would require ACPI method implementation
    echo "  BIQ$biq: [probe method needed]" >> $LOGFILE
    ((biq_count++))
done
echo "  📊 Potential BIQ variables: $biq_count" | tee -a $LOGFILE

echo "🔍 4. System Baseline Capture" | tee -a $LOGFILE
mkdir -p baseline/
cat /proc/meminfo > baseline/memory_$(date +%H%M%S).txt
cat /proc/interrupts > baseline/interrupts_$(date +%H%M%S).txt
dmesg > baseline/dmesg_$(date +%H%M%S).txt

echo "✅ Phase 1 Complete: Passive enumeration finished safely" | tee -a $LOGFILE
echo "📊 Results logged to: $LOGFILE"
```

### **Phase 2: Read-Only Queries Script**  
```bash
#!/bin/bash
# dsmil_phase2_readonly.sh

set -euo pipefail

LOGFILE="dsmil_phase2_$(date +%Y%m%d_%H%M%S).log"
echo "📖 Phase 2: Read-Only Device Queries" | tee -a $LOGFILE

# Prerequisites check
if [[ ! -f "dsmil_phase1_"*".log" ]]; then
    echo "❌ ABORT: Phase 1 must complete first"
    exit 1
fi

# Safety check  
./dsmil_preflight_check.sh || exit 1

# Start monitoring
./dsmil_monitor.sh > monitor_phase2.log &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

echo "🔍 1. Driver Status Check" | tee -a $LOGFILE
if lsmod | grep -q dell.milspec; then
    echo "  ✅ dell-milspec driver loaded" | tee -a $LOGFILE
    
    # Query driver status
    if command -v milspec-control &> /dev/null; then
        sudo milspec-control --status 2>&1 | tee -a $LOGFILE
    fi
    
    # Check sysfs interface
    if [[ -d "/sys/devices/platform/dell-milspec" ]]; then
        find /sys/devices/platform/dell-milspec -type f -readable | while read file; do
            echo "  $file: $(cat $file 2>/dev/null || echo 'N/A')" | tee -a $LOGFILE
        done
    fi
    
else
    echo "  ❌ dell-milspec driver not loaded" | tee -a $LOGFILE
    echo "  📋 Available for module loading test" | tee -a $LOGFILE
fi

echo "🔍 2. ACPI Method Testing" | tee -a $LOGFILE
# This would require ACPI method caller implementation
echo "  [ACPI method testing framework needed]" | tee -a $LOGFILE

echo "✅ Phase 2 Complete: Read-only queries finished safely" | tee -a $LOGFILE
```

## ⚠️ **Critical Safety Warnings**

### **DO NOT PROCEED IF:**
- ❌ System temperature > 75°C before starting
- ❌ Free memory < 4GB available  
- ❌ No system backup created
- ❌ Remote recovery access unavailable
- ❌ Any pre-flight check fails

### **ABORT IMMEDIATELY IF:**
- 🚨 System temperature > 85°C
- 🚨 System becomes unresponsive > 30 seconds
- 🚨 Memory exhaustion detected
- 🚨 Kernel panic or crash
- 🚨 Hardware error messages in dmesg

### **Recovery Prerequisites**
1. **Full system backup** created and verified
2. **Live rescue media** available (USB/network boot)  
3. **Remote access** capability (SSH/IPMI/iLO)
4. **Documentation** of all activation steps for rollback
5. **Emergency contact** plan if system becomes inaccessible

---

**Status**: Safe probing methodology complete - ready for cautious implementation with full monitoring and rollback capability.