#!/bin/bash
# Dell MIL-SPEC Health Check Tool
# Post-installation validation and runtime health monitoring

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    if eval "$2" &>/dev/null; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAIL++))
    fi
}

echo "═══════════════════════════════════════════════════════"
echo "  Dell MIL-SPEC Health Check"
echo "═══════════════════════════════════════════════════════"
echo ""

echo "Module Status:"
check "dsmil-72dev loaded" "lsmod | grep -q dsmil"
check "tpm2_accel_early loaded" "lsmod | grep -q tpm2_accel"

echo ""
echo "Device Nodes:"
check "/dev/dsmil0 exists" "test -c /dev/dsmil0"
check "/dev/tpm2_accel_early exists" "test -c /dev/tpm2_accel_early"
check "/dev/tpm0 exists" "test -c /dev/tpm0"

echo ""
echo "Services:"
check "tpm2-acceleration-early.service" "systemctl is-active tpm2-acceleration-early.service"

echo ""
echo "Configuration:"
check "/etc/modprobe.d/dell-milspec.conf" "test -f /etc/modprobe.d/dell-milspec.conf"
check "/etc/modprobe.d/tpm2-acceleration.conf" "test -f /etc/modprobe.d/tpm2-acceleration.conf"

echo ""
echo "═══════════════════════════════════════════════════════"
echo -e "  Result: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""

[ $FAIL -eq 0 ] && echo "✓ System healthy" && exit 0
echo "✗ Issues detected - check logs: sudo dmesg | tail -50"
exit 1
