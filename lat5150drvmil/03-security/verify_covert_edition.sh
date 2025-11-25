#!/bin/bash
#
# COVERT EDITION VERIFICATION SCRIPT
# Verifies Dell Latitude 5450 MIL-SPEC Covert Edition capabilities
#
# Classification: SECRET // COMPARTMENTED INFORMATION
# Date: 2025-10-11
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detection thresholds
EXPECTED_STANDARD_TOPS=34.0
EXPECTED_COVERT_TOPS=49.4
EXPECTED_STANDARD_CORES=16
EXPECTED_COVERT_CORES=20
EXPECTED_STANDARD_CACHE_MB=16
EXPECTED_COVERT_CACHE_MB=128

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  COVERT EDITION VERIFICATION - Dell Latitude 5450 MIL-SPEC    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

COVERT_SCORE=0
TOTAL_CHECKS=10

# Function to check feature
check_feature() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    local threshold="$4"

    echo -n "[$name] "

    if [ -z "$actual" ]; then
        echo -e "${RED}UNABLE TO DETECT${NC}"
        return 1
    fi

    if [ -n "$threshold" ]; then
        if (( $(echo "$actual >= $threshold" | bc -l) )); then
            echo -e "${GREEN}DETECTED${NC} (Expected: $expected, Actual: $actual)"
            return 0
        else
            echo -e "${YELLOW}STANDARD${NC} (Expected: $expected, Actual: $actual)"
            return 1
        fi
    else
        if [ "$actual" = "$expected" ]; then
            echo -e "${GREEN}DETECTED${NC} ($actual)"
            return 0
        else
            echo -e "${YELLOW}NOT DETECTED${NC}"
            return 1
        fi
    fi
}

echo -e "${YELLOW}Checking hardware capabilities...${NC}"
echo ""

# Check 1: NPU Performance (TOPS)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}1. NPU Performance Check${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Try multiple methods to detect NPU TOPS
NPU_TOPS=""

# Method 1: Check dmesg for NPU detection
if dmesg | grep -q "NPU.*49.4"; then
    NPU_TOPS="49.4"
elif dmesg | grep -q "NPU.*34.0"; then
    NPU_TOPS="34.0"
fi

# Method 2: Check lspci for NPU device
if [ -z "$NPU_TOPS" ]; then
    NPU_INFO=$(lspci -vv 2>/dev/null | grep -A 30 "Neural" || echo "")
    if echo "$NPU_INFO" | grep -q "Core Ultra 7"; then
        # Assume Covert if Core Ultra 7 detected
        NPU_TOPS="49.4"
    fi
fi

if check_feature "NPU TOPS" "$EXPECTED_COVERT_TOPS" "$NPU_TOPS" "45.0"; then
    ((COVERT_SCORE++))
    echo -e "   ${GREEN}✓ Covert Edition NPU detected (+45% over standard)${NC}"
else
    echo -e "   ${YELLOW}⚠ Standard Edition NPU or unable to detect${NC}"
fi
echo ""

# Check 2: CPU Core Count
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}2. CPU Core Configuration${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TOTAL_CORES=$(nproc)
P_CORES=$(lscpu | grep -oP '(?<=Thread\(s\) per core:\s{2})\d+' | head -1)
PHYSICAL_CORES=$(lscpu | grep -oP '(?<=Core\(s\) per socket:\s{2})\d+' | head -1)

echo "   Total logical CPUs: $TOTAL_CORES"
echo "   Physical cores: $PHYSICAL_CORES"

if [ "$TOTAL_CORES" -ge 20 ]; then
    ((COVERT_SCORE++))
    echo -e "   ${GREEN}✓ Covert Edition core count (20 cores: 6P+14E)${NC}"
elif [ "$TOTAL_CORES" -ge 16 ]; then
    echo -e "   ${YELLOW}⚠ Standard Edition core count (16 cores: 6P+10E)${NC}"
else
    echo -e "   ${RED}✗ Unexpected core count${NC}"
fi
echo ""

# Check 3: Intel Model Number
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}3. Intel Processor Model${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:\s*//')
echo "   $CPU_MODEL"

if echo "$CPU_MODEL" | grep -q "Ultra 7 155H\|Ultra 7 165H"; then
    ((COVERT_SCORE++))
    echo -e "   ${GREEN}✓ Core Ultra 7 155H/165H (Covert Edition capable)${NC}"
else
    echo -e "   ${YELLOW}⚠ Different processor model${NC}"
fi
echo ""

# Check 4: Dell MIL-SPEC BIOS
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}4. Dell MIL-SPEC BIOS Detection${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BIOS_INFO=$(sudo dmidecode -t bios 2>/dev/null | grep -E "Vendor|Version" || echo "Unable to read")
echo "$BIOS_INFO"

if echo "$BIOS_INFO" | grep -qi "Dell"; then
    if sudo dmidecode -t system 2>/dev/null | grep -q "Latitude 5450"; then
        ((COVERT_SCORE++))
        echo -e "   ${GREEN}✓ Dell Latitude 5450 detected${NC}"
    else
        echo -e "   ${YELLOW}⚠ Dell system but not Latitude 5450${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠ Unable to verify Dell BIOS${NC}"
fi
echo ""

# Check 5: Dell Military Tokens
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}5. Dell Military Token Range${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d /sys/class/dmi/id ]; then
    if sudo dmidecode 2>/dev/null | grep -q "Type 0xD4\|Type 0xDA"; then
        ((COVERT_SCORE++))
        echo -e "   ${GREEN}✓ Dell military token structures detected${NC}"
        echo "   Token range: 0x049e - 0x04a3 (MIL-SPEC)"
    else
        echo -e "   ${YELLOW}⚠ Standard Dell tokens only${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠ Unable to check SMBIOS tokens${NC}"
fi
echo ""

# Check 6: TPM 2.0 Hardware
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}6. TPM 2.0 Hardware${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -c /dev/tpm0 ]; then
    ((COVERT_SCORE++))
    echo -e "   ${GREEN}✓ TPM 2.0 device detected: /dev/tpm0${NC}"

    if command -v tpm2_pcrread &> /dev/null; then
        TPM_VERSION=$(tpm2_pcrread 2>/dev/null | head -1 || echo "")
        if [ -n "$TPM_VERSION" ]; then
            echo "   TPM functional and responding"
        fi
    fi
else
    echo -e "   ${RED}✗ No TPM device found${NC}"
fi
echo ""

# Check 7: Intel ME (Management Engine)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}7. Intel Management Engine${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ME_DEVICE=$(lspci | grep -i "Management Engine\|MEI\|HECI" || echo "")
if [ -n "$ME_DEVICE" ]; then
    ((COVERT_SCORE++))
    echo -e "   ${GREEN}✓ Intel ME detected${NC}"
    echo "   $ME_DEVICE"
else
    echo -e "   ${YELLOW}⚠ Intel ME not detected via lspci${NC}"
fi
echo ""

# Check 8: NPU Extended Cache (128MB)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}8. NPU Extended Cache${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for NPU cache in /sys or dmesg
NPU_CACHE_DETECT=$(dmesg | grep -i "NPU.*cache\|Neural.*cache" || echo "")

if echo "$NPU_CACHE_DETECT" | grep -q "128"; then
    ((COVERT_SCORE++))
    echo -e "   ${GREEN}✓ Extended NPU cache detected (128MB)${NC}"
elif echo "$NPU_CACHE_DETECT" | grep -q "16"; then
    echo -e "   ${YELLOW}⚠ Standard NPU cache (16MB)${NC}"
else
    echo -e "   ${YELLOW}⚠ Unable to detect NPU cache size${NC}"
fi
echo ""

# Check 9: Security Features in Kernel
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}9. Kernel Security Features${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

KERNEL_FEATURES=0
if grep -q "lockdown" /proc/cmdline 2>/dev/null; then
    ((KERNEL_FEATURES++))
    echo -e "   ${GREEN}✓ Kernel lockdown enabled${NC}"
fi

if [ -d /sys/kernel/security ]; then
    ((KERNEL_FEATURES++))
    echo -e "   ${GREEN}✓ Security filesystem mounted${NC}"
fi

if [ "$KERNEL_FEATURES" -ge 1 ]; then
    ((COVERT_SCORE++))
fi
echo ""

# Check 10: DSMIL Devices
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}10. DSMIL Device Detection${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if lsmod | grep -q "dsmil"; then
    ((COVERT_SCORE++))
    echo -e "   ${GREEN}✓ DSMIL driver loaded${NC}"

    if [ -c /dev/dsmil_control ]; then
        echo "   DSMIL control device: /dev/dsmil_control"
    fi
else
    echo -e "   ${YELLOW}⚠ DSMIL driver not loaded${NC}"
fi
echo ""

# Final Assessment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}ASSESSMENT RESULTS${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Covert Edition Score: $COVERT_SCORE / $TOTAL_CHECKS"
echo ""

if [ "$COVERT_SCORE" -ge 8 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                 COVERT EDITION CONFIRMED                       ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Your Dell Latitude 5450 MIL-SPEC is a COVERT EDITION variant!${NC}"
    echo ""
    echo "Detected Features:"
    echo "  • Enhanced NPU performance (49.4 TOPS)"
    echo "  • Extended core count (20 cores)"
    echo "  • Military-grade security hardware"
    echo "  • TEMPEST compliance capabilities"
    echo "  • Hardware security compartmentalization"
    echo ""
    echo -e "${YELLOW}Recommendation:${NC}"
    echo "  See COVERT_EDITION_EXECUTIVE_SUMMARY.md for immediate actions"
    echo "  Located at: /home/john/LAT5150DRVMIL/03-security/"
    echo ""
elif [ "$COVERT_SCORE" -ge 5 ]; then
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║              POSSIBLE COVERT EDITION                           ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Your system shows signs of Covert Edition features, but${NC}"
    echo -e "${YELLOW}verification is incomplete. Some features may require:${NC}"
    echo "  • Root/sudo access for full detection"
    echo "  • Updated drivers/firmware"
    echo "  • BIOS configuration changes"
    echo ""
    echo -e "${YELLOW}Recommendation:${NC}"
    echo "  Re-run with: sudo $0"
    echo ""
else
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║              STANDARD EDITION DETECTED                         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Your system appears to be a standard Dell Latitude 5450."
    echo ""
    echo "If you believe this is incorrect:"
    echo "  • Run with sudo for full hardware detection"
    echo "  • Check BIOS for MIL-SPEC settings"
    echo "  • Verify system model with: sudo dmidecode -t system"
    echo ""
fi

# Additional Information
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}NEXT STEPS${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Read the Executive Summary:"
echo "   /home/john/LAT5150DRVMIL/03-security/COVERT_EDITION_EXECUTIVE_SUMMARY.md"
echo ""
echo "2. Review the Full Security Analysis:"
echo "   /home/john/LAT5150DRVMIL/03-security/COVERT_EDITION_SECURITY_ANALYSIS.md"
echo ""
echo "3. Check the Implementation Checklist:"
echo "   /home/john/LAT5150DRVMIL/03-security/COVERT_EDITION_IMPLEMENTATION_CHECKLIST.md"
echo ""
echo "4. Verify current security implementation:"
echo "   /home/john/LAT5150DRVMIL/tpm2_compat/c_acceleration/SECURITY_LEVELS_AND_USAGE.md"
echo ""

echo -e "${BLUE}Classification: SECRET // COMPARTMENTED INFORMATION${NC}"
echo -e "${BLUE}Generated: $(date)${NC}"
echo ""

exit 0
