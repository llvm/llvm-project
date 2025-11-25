#!/bin/bash
#
# DSMIL SMI Access Testing Script
# Tests both accessible and locked tokens
#

set -e

MODULE_PATH="/home/john/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko"
LOG_DIR="/home/john/LAT5150DRVMIL/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/dsmil_test_$TIMESTAMP.log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Ensure log directory exists
mkdir -p "$LOG_DIR"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

check_thermal() {
    local max_temp=0
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        if [ -r "$zone" ]; then
            temp=$(cat "$zone")
            temp_c=$((temp / 1000))
            if [ $temp_c -gt $max_temp ]; then
                max_temp=$temp_c
            fi
        fi
    done
    echo $max_temp
}

log "${BLUE}============================================${NC}"
log "${BLUE}DSMIL SMI Access Testing${NC}"
log "${BLUE}============================================${NC}"
log ""

# Check current temperature
current_temp=$(check_thermal)
log "📊 Current system temperature: ${current_temp}°C"

if [ $current_temp -gt 95 ]; then
    log "${RED}❌ Temperature too high for testing!${NC}"
    exit 1
fi

# Check if module is already loaded
if lsmod | grep -q dsmil_72dev; then
    log "${YELLOW}Module already loaded, unloading first...${NC}"
    sudo rmmod dsmil-72dev 2>/dev/null || true
    sleep 2
fi

# Load module with SMI access enabled
log ""
log "${GREEN}Loading DSMIL module with SMI access...${NC}"
log "Parameters:"
log "  - force_jrtc1_mode=1 (training mode)"
log "  - thermal_threshold=100 (MIL-SPEC)"
log "  - enable_smi_access=1 (locked token access)"
log ""

sudo insmod "$MODULE_PATH" \
    force_jrtc1_mode=1 \
    thermal_threshold=100 \
    enable_smi_access=1

sleep 2

# Verify module loaded
if lsmod | grep -q dsmil_72dev; then
    log "${GREEN}✅ Module loaded successfully${NC}"
else
    log "${RED}❌ Module failed to load${NC}"
    exit 1
fi

# Check kernel messages
log ""
log "${BLUE}Recent kernel messages:${NC}"
sudo dmesg | tail -20 | grep -i dsmil | while read line; do
    log "  $line"
done

# Check for device files
log ""
log "${BLUE}Checking for device files...${NC}"
if ls /dev/dsmil* 2>/dev/null; then
    log "${GREEN}✅ DSMIL device files found:${NC}"
    ls -la /dev/dsmil* | while read line; do
        log "  $line"
    done
else
    log "${YELLOW}No /dev/dsmil* files (may be normal)${NC}"
fi

# Test accessible token first (0x481 - thermal control)
log ""
log "${BLUE}Testing accessible token 0x481 (thermal control)...${NC}"

# Try to read token value
if command -v smbios-token-ctl &> /dev/null; then
    log "Reading token 0x481..."
    sudo smbios-token-ctl --token-id=0x481 --get 2>&1 | while read line; do
        log "  $line"
    done
else
    log "${YELLOW}smbios-token-ctl not available${NC}"
fi

# Monitor for DSMIL response
log ""
log "${BLUE}Monitoring for DSMIL kernel activity...${NC}"
sudo dmesg | tail -10 | grep -i "dsmil\|smi\|token" | while read line; do
    log "  $line"
done

# Check temperature after operation
post_temp=$(check_thermal)
log ""
log "📊 Post-operation temperature: ${post_temp}°C"
temp_change=$((post_temp - current_temp))
log "   Temperature change: ${temp_change}°C"

# Test SMI access for locked token (if supported)
log ""
log "${BLUE}Testing SMI access for locked token 0x0480...${NC}"
log "${YELLOW}Note: This requires kernel debugfs interface${NC}"

# Check if debugfs is mounted
if [ -d "/sys/kernel/debug" ]; then
    if [ -d "/sys/kernel/debug/dsmil" ]; then
        log "${GREEN}✅ DSMIL debugfs interface found${NC}"
        
        # Try to read locked token via SMI
        if [ -f "/sys/kernel/debug/dsmil/smi_test" ]; then
            log "Testing SMI read of token 0x0480..."
            echo "read 0x0480" | sudo tee /sys/kernel/debug/dsmil/smi_test
            sleep 1
            
            # Check results
            if [ -f "/sys/kernel/debug/dsmil/smi_result" ]; then
                result=$(sudo cat /sys/kernel/debug/dsmil/smi_result)
                log "  SMI Result: $result"
            fi
        else
            log "${YELLOW}SMI test interface not available${NC}"
        fi
    else
        log "${YELLOW}DSMIL debugfs not available${NC}"
    fi
else
    log "${YELLOW}debugfs not mounted${NC}"
fi

# Final kernel messages
log ""
log "${BLUE}Final kernel messages:${NC}"
sudo dmesg | tail -30 | grep -i "dsmil\|smi\|token" | while read line; do
    log "  $line"
done

# Summary
log ""
log "${BLUE}============================================${NC}"
log "${BLUE}Test Summary${NC}"
log "${BLUE}============================================${NC}"
log "Module loaded: ${GREEN}YES${NC}"
log "Initial temp: ${current_temp}°C"
log "Final temp: ${post_temp}°C"
log "Temp change: ${temp_change}°C"

if [ $temp_change -lt 5 ]; then
    log "Thermal status: ${GREEN}SAFE${NC}"
else
    log "Thermal status: ${YELLOW}ELEVATED${NC}"
fi

log ""
log "Log saved to: $LOG_FILE"
log ""
log "${YELLOW}Next steps:${NC}"
log "1. Check dmesg for SMI operations: sudo dmesg | grep -i smi"
log "2. Test thermal token: sudo python3 test_thermal_token.py"
log "3. Run correlation analysis: python3 analyze_token_correlation.py --live"
log "4. Monitor thermal: watch -n1 sensors"
log ""
log "To unload module: sudo rmmod dsmil-72dev"