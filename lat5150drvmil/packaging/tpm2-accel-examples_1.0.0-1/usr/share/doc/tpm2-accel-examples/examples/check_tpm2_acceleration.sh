#!/bin/bash
#
# TPM2 Acceleration Module - Quick Status Check
# Shows current configuration, security level, and hardware status
#

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TPM2 Acceleration Module Status${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if module is loaded
if lsmod | grep -q "^tpm2_accel_early"; then
    echo -e "${GREEN}✅ Module Status: LOADED${NC}"

    # Get module size
    MODULE_SIZE=$(lsmod | grep "^tpm2_accel_early" | awk '{print $2}')
    echo -e "   Size in memory: ${MODULE_SIZE} bytes"

    # Check security level
    if [ -f /sys/module/tpm2_accel_early/parameters/security_level ]; then
        SEC_LEVEL=$(cat /sys/module/tpm2_accel_early/parameters/security_level)
        case $SEC_LEVEL in
            0) echo -e "   Security Level: ${GREEN}UNCLASSIFIED (0)${NC}" ;;
            1) echo -e "   Security Level: ${YELLOW}CONFIDENTIAL (1)${NC}" ;;
            2) echo -e "   Security Level: ${YELLOW}SECRET (2)${NC}" ;;
            3) echo -e "   Security Level: ${RED}TOP SECRET (3)${NC}" ;;
            *) echo -e "   Security Level: Unknown ($SEC_LEVEL)" ;;
        esac
    fi

    # Check debug mode
    if [ -f /sys/module/tpm2_accel_early/parameters/debug_mode ]; then
        DEBUG_MODE=$(cat /sys/module/tpm2_accel_early/parameters/debug_mode)
        if [ "$DEBUG_MODE" = "Y" ]; then
            echo -e "   Debug Mode: ${YELLOW}ENABLED${NC}"
        else
            echo -e "   Debug Mode: ${GREEN}DISABLED${NC}"
        fi
    fi

    # Check early init
    if [ -f /sys/module/tpm2_accel_early/parameters/early_init ]; then
        EARLY_INIT=$(cat /sys/module/tpm2_accel_early/parameters/early_init)
        if [ "$EARLY_INIT" = "Y" ]; then
            echo -e "   Early Init: ${GREEN}ENABLED${NC}"
        else
            echo -e "   Early Init: DISABLED"
        fi
    fi

else
    echo -e "${RED}❌ Module Status: NOT LOADED${NC}"
fi

echo ""

# Check device node
if [ -c /dev/tpm2_accel_early ]; then
    echo -e "${GREEN}✅ Device Node: AVAILABLE${NC}"
    DEVICE_INFO=$(ls -l /dev/tpm2_accel_early | awk '{print $3":"$4, "("$5, $6")"}')
    echo -e "   /dev/tpm2_accel_early"
    echo -e "   Owner: $DEVICE_INFO"
else
    echo -e "${RED}❌ Device Node: NOT FOUND${NC}"
    echo -e "   Expected: /dev/tpm2_accel_early"
fi

echo ""

# Check standard TPM devices
echo -e "${BLUE}Standard TPM Devices:${NC}"
if [ -c /dev/tpm0 ]; then
    echo -e "${GREEN}✅ /dev/tpm0${NC} (Native TPM)"
    if [ -d /sys/class/tpm/tpm0/device/driver ]; then
        TPM_DRIVER=$(readlink /sys/class/tpm/tpm0/device/driver | awk -F'/' '{print $NF}')
        echo -e "   Driver: $TPM_DRIVER"
    fi
else
    echo -e "${YELLOW}⚠️  /dev/tpm0 not found${NC}"
fi

if [ -c /dev/tpmrm0 ]; then
    echo -e "${GREEN}✅ /dev/tpmrm0${NC} (Resource Manager)"
else
    echo -e "   /dev/tpmrm0 not found"
fi

echo ""

# Check systemd service
if systemctl is-enabled tpm2-acceleration-early.service &> /dev/null; then
    echo -e "${GREEN}✅ Systemd Service: ENABLED${NC}"
    echo -e "   Will load on boot"
else
    echo -e "${YELLOW}⚠️  Systemd Service: NOT ENABLED${NC}"
fi

echo ""

# Check configuration files
echo -e "${BLUE}Configuration Files:${NC}"
if [ -f /etc/modules-load.d/tpm2-acceleration.conf ]; then
    echo -e "${GREEN}✅ /etc/modules-load.d/tpm2-acceleration.conf${NC}"
else
    echo -e "${RED}❌ /etc/modules-load.d/tpm2-acceleration.conf${NC}"
fi

if [ -f /etc/modprobe.d/tpm2-acceleration.conf ]; then
    echo -e "${GREEN}✅ /etc/modprobe.d/tpm2-acceleration.conf${NC}"
else
    echo -e "${RED}❌ /etc/modprobe.d/tpm2-acceleration.conf${NC}"
fi

echo ""

# Recent kernel messages
if lsmod | grep -q "^tpm2_accel_early"; then
    echo -e "${BLUE}Recent Kernel Messages (last 5):${NC}"
    sudo dmesg 2>/dev/null | grep tpm2_accel | tail -5 | sed 's/^/   /'

    echo ""

    # Hardware detection summary
    echo -e "${BLUE}Hardware Detection:${NC}"
    if sudo dmesg 2>/dev/null | grep -q "Intel NPU detected"; then
        echo -e "${GREEN}✅ Intel NPU (34.0 TOPS)${NC}"
    else
        echo -e "   Intel NPU: Not detected"
    fi

    if sudo dmesg 2>/dev/null | grep -q "Intel GNA.*detected"; then
        echo -e "${GREEN}✅ Intel GNA 3.5${NC}"
    else
        echo -e "   Intel GNA: Not detected"
    fi

    if sudo dmesg 2>/dev/null | grep -q "Intel ME detected"; then
        echo -e "${GREEN}✅ Intel ME${NC}"
    else
        echo -e "   Intel ME: Not detected"
    fi

    if sudo dmesg 2>/dev/null | grep -q "TPM.*hardware.*detected"; then
        echo -e "${GREEN}✅ TPM 2.0 Hardware${NC}"
    else
        echo -e "   TPM 2.0: Not detected"
    fi

    if sudo dmesg 2>/dev/null | grep -q "Dell SMBIOS.*validated"; then
        echo -e "${GREEN}✅ Dell SMBIOS Integration${NC}"
    else
        echo -e "   Dell SMBIOS: Not validated"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quick Commands:${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "View full logs:"
echo "  sudo dmesg | grep tpm2_accel | less"
echo ""
echo "Change security level:"
echo "  sudo modprobe -r tpm2_accel_early"
echo "  sudo modprobe tpm2_accel_early security_level=1"
echo ""
echo "Enable debug mode:"
echo "  sudo modprobe -r tpm2_accel_early"
echo "  sudo modprobe tpm2_accel_early debug_mode=1"
echo ""
echo "Test standard TPM:"
echo "  tpm2_pcrread"
echo "  tpm2_getrandom 32"
echo ""
echo "Module information:"
echo "  modinfo tpm2_accel_early"
echo ""
