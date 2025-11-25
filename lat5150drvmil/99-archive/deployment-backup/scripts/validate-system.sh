#!/bin/bash
# Dell MIL-SPEC System Validation Tool
# Pre-installation hardware and software compatibility check

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

pass() { echo -e "${GREEN}✓ PASS${NC}: $1"; ((PASS++)); }
fail() { echo -e "${RED}✗ FAIL${NC}: $1"; ((FAIL++)); }
warn() { echo -e "${YELLOW}⚠ WARN${NC}: $1"; ((WARN++)); }
info() { echo -e "${BLUE}ℹ INFO${NC}: $1"; }

echo "═══════════════════════════════════════════════════════"
echo "  Dell MIL-SPEC Platform Validation"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check 1: Dell Hardware
info "Checking Dell hardware..."
if command -v dmidecode &>/dev/null; then
    VENDOR=$(sudo dmidecode -s system-manufacturer 2>/dev/null || echo "Unknown")
    PRODUCT=$(sudo dmidecode -s system-product-name 2>/dev/null || echo "Unknown")
    
    if [[ "$VENDOR" == "Dell Inc." ]]; then
        if [[ "$PRODUCT" =~ "Latitude 5450" ]]; then
            pass "Dell Latitude 5450 detected"
        else
            warn "Dell hardware but not Latitude 5450 (found: $PRODUCT)"
        fi
    else
        fail "Not Dell hardware (found: $VENDOR)"
    fi
else
    warn "dmidecode not available - cannot verify hardware"
fi

# Check 2: CPU Model
info "Checking CPU model..."
CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
if [[ "$CPU_MODEL" =~ "Core Ultra" ]]; then
    pass "Intel Core Ultra detected (Meteor Lake)"
elif [[ "$CPU_MODEL" =~ "Core" ]]; then
    warn "Intel CPU but not Core Ultra: $CPU_MODEL"
else
    fail "Unsupported CPU: $CPU_MODEL"
fi

# Check 3: Kernel Version
info "Checking kernel version..."
KERNEL_VER=$(uname -r)
KERNEL_MAJ=$(echo $KERNEL_VER | cut -d. -f1)
KERNEL_MIN=$(echo $KERNEL_VER | cut -d. -f2)

if [ "$KERNEL_MAJ" -gt 6 ] || [ "$KERNEL_MAJ" -eq 6 -a "$KERNEL_MIN" -ge 14 ]; then
    pass "Kernel $KERNEL_VER (>= 6.14.0 required)"
elif [ "$KERNEL_MAJ" -eq 6 -a "$KERNEL_MIN" -ge 8 ]; then
    warn "Kernel $KERNEL_VER (6.14.0+ recommended)"
else
    fail "Kernel $KERNEL_VER (6.8.0+ required)"
fi

# Check 4: Kernel Headers
info "Checking kernel headers..."
if [ -d "/lib/modules/$KERNEL_VER/build" ]; then
    pass "Kernel headers installed for $KERNEL_VER"
else
    fail "Kernel headers missing - install linux-headers-$KERNEL_VER"
fi

# Check 5: TPM 2.0
info "Checking TPM 2.0..."
if [ -c /dev/tpm0 ]; then
    pass "TPM 2.0 device present (/dev/tpm0)"
elif [ -c /dev/tpmrm0 ]; then
    pass "TPM 2.0 resource manager present"
else
    fail "No TPM 2.0 device found"
fi

# Check 6: Build tools
info "Checking build tools..."
MISSING_TOOLS=""
for tool in gcc make dkms; do
    if command -v $tool &>/dev/null; then
        pass "$tool installed"
    else
        fail "$tool not installed"
        MISSING_TOOLS="$MISSING_TOOLS $tool"
    fi
done

# Check 7: NPU (optional)
info "Checking Intel NPU (optional)..."
if lspci | grep -qi "1d1d"; then
    pass "Intel NPU detected"
elif lspci | grep -qi "npu"; then
    pass "NPU device found"
else
    warn "Intel NPU not detected (optional for acceleration)"
fi

# Check 8: Memory
info "Checking system memory..."
MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$MEM_GB" -ge 8 ]; then
    pass "Sufficient memory: ${MEM_GB}GB (8GB+ recommended)"
else
    warn "Low memory: ${MEM_GB}GB (8GB+ recommended)"
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Validation Summary"
echo "═══════════════════════════════════════════════════════"
echo -e "  ${GREEN}PASS${NC}: $PASS  ${YELLOW}WARN${NC}: $WARN  ${RED}FAIL${NC}: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ System ready for Dell MIL-SPEC installation${NC}"
    exit 0
else
    echo -e "${RED}✗ System has compatibility issues${NC}"
    echo "  Install missing components and re-run validation"
    exit 1
fi
