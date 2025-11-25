#!/bin/bash
################################################################################
# DSMIL Module Reload Script
################################################################################
# Quick script to unload and reload the DSMIL kernel module
#
# Usage:
#   sudo ./reload-dsmil-module.sh
#
# Author: LAT5150DRVMIL AI Platform
# Version: 1.0.0
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}DSMIL Module Reload Utility${NC}"
echo "═══════════════════════════════════════════════════════════════"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}ERROR: This script must be run as root${NC}"
    echo "Please run with: sudo $0"
    exit 1
fi

MODULE_NAMES=("dsmil_84dev" "dsmil_72dev")

echo ""
echo -e "${BOLD}Step 1: Unloading existing DSMIL modules${NC}"

unloaded_any=false
for mod in "${MODULE_NAMES[@]}"; do
    if lsmod | grep -q "^$mod"; then
        echo -e "  Unloading ${CYAN}$mod${NC}..."
        if rmmod "$mod" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Unloaded $mod"
            unloaded_any=true
        else
            echo -e "  ${YELLOW}⚠${NC} Failed to unload $mod via rmmod"
            echo -e "    Trying modprobe -r..."
            if modprobe -r "$mod" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Unloaded $mod"
                unloaded_any=true
            else
                echo -e "  ${RED}✗${NC} Unable to unload $mod"
            fi
        fi
    fi
done

# Check sysfs for any remaining modules
for mod in "${MODULE_NAMES[@]}"; do
    if [[ -d "/sys/module/$mod" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Module $mod still present in /sys/module"
        echo -e "    Attempting removal..."
        rmmod "$mod" 2>/dev/null || true
    fi
done

if [[ "$unloaded_any" = false ]]; then
    echo -e "  ${GREEN}✓${NC} No modules were loaded"
else
    echo -e "  ${GREEN}✓${NC} All modules unloaded"
    # Give kernel time to cleanup
    sleep 1
fi

echo ""
echo -e "${BOLD}Step 2: Loading dsmil_84dev module${NC}"

LOADED=false
if modprobe dsmil_84dev 2>/dev/null; then
    LOADED=true
elif modprobe dsmil_72dev 2>/dev/null; then
    LOADED=true
    echo -e "  ${YELLOW}Note: Loaded legacy dsmil_72dev instead${NC}"
fi

if [ "$LOADED" = true ]; then
    echo -e "  ${GREEN}✓${NC} Module loaded successfully"
else
    echo -e "  ${RED}✗${NC} Failed to load module"
    echo ""
    echo -e "  ${BOLD}Diagnostics:${NC}"
    echo "    Recent dmesg output:"
    dmesg | tail -10 | grep -i dsmil | sed 's/^/      /' || echo "      (no DSMIL messages)"
    echo ""
    echo "  ${BOLD}Possible causes:${NC}"
    echo "    • Platform driver still registered (try: sudo rmmod -f dsmil_72dev)"
    echo "    • Module not installed (run: sudo make install)"
    echo "    • Kernel version mismatch (rebuild module)"
    echo "    • System may need reboot to fully clear driver state"
    exit 1
fi

echo ""
echo -e "${BOLD}Step 3: Verifying module status${NC}"

if lsmod | grep -q "dsmil"; then
    echo -e "  ${GREEN}✓${NC} Module verified in lsmod:"
    lsmod | grep dsmil | sed 's/^/    /'
else
    echo -e "  ${YELLOW}⚠${NC} Module not showing in lsmod (may be normal)"
fi

# Check device nodes
if ls /dev/dsmil* &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Device nodes present:"
    ls -la /dev/dsmil* | sed 's/^/    /'
else
    echo -e "  ${YELLOW}⚠${NC} No /dev/dsmil* nodes (created on first access)"
fi

echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}Module reload complete!${NC}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""
