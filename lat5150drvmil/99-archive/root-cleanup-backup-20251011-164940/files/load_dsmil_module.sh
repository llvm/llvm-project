#!/bin/bash
#
# DSMIL Module Loader with SMBIOS Monitoring
# Safe configuration for token testing
#

set -e

MODULE_PATH="/home/john/LAT5150DRVMIL/01-source/kernel/dsmil-72dev.ko"
LOG_FILE="/home/john/LAT5150DRVMIL/logs/module_load_$(date +%Y%m%d_%H%M%S).log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}DSMIL Module Loader v1.0${NC}"
echo "================================"

# Check if module exists
if [ ! -f "$MODULE_PATH" ]; then
    echo -e "${RED}ERROR: Module not found at $MODULE_PATH${NC}"
    echo "Please build the module first with: cd 01-source/kernel && make"
    exit 1
fi

# Check if already loaded
if lsmod | grep -q dsmil_72dev; then
    echo -e "${YELLOW}Module already loaded. Unloading first...${NC}"
    sudo rmmod dsmil-72dev 2>/dev/null || true
    sleep 1
fi

# Load module with safe parameters
echo -e "${GREEN}Loading DSMIL module with safe parameters:${NC}"
echo "  - JRTC1 mode: Enabled (training variant safety)"
echo "  - Thermal threshold: 100°C (MIL-SPEC range)"
echo "  - SMBIOS monitoring: Enabled"
echo ""

# Load the module
sudo insmod "$MODULE_PATH" \
    force_jrtc1_mode=1 \
    thermal_threshold=100 \
    enable_smbios_monitoring=1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Module loaded successfully${NC}"
    
    # Verify module is loaded
    if lsmod | grep -q dsmil_72dev; then
        echo -e "${GREEN}✅ Module verified in kernel${NC}"
        
        # Check kernel messages
        echo ""
        echo "Recent kernel messages:"
        echo "----------------------"
        sudo dmesg | tail -20 | grep -i dsmil || echo "No DSMIL messages in last 20 lines"
        
        # Check for device files
        echo ""
        echo "Checking for device files:"
        if ls /dev/dsmil* 2>/dev/null; then
            echo -e "${GREEN}✅ DSMIL device files created${NC}"
            ls -la /dev/dsmil*
        else
            echo -e "${YELLOW}No /dev/dsmil* files found (this may be normal)${NC}"
        fi
        
        # Log success
        echo "Module loaded at $(date)" >> "$LOG_FILE"
        echo -e "${GREEN}✅ Module ready for SMBIOS token testing${NC}"
    else
        echo -e "${RED}❌ Module load verification failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Failed to load module${NC}"
    echo "Check dmesg for error details: sudo dmesg | tail"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Run token tests: cd testing && ./run_testbed_suite.sh"
echo "2. Monitor kernel: sudo dmesg -w | grep -i dsmil"
echo "3. Check thermal: watch -n1 sensors"
echo ""
echo "To unload module: sudo rmmod dsmil-72dev"