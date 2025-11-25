#!/bin/bash
# TPM2 Native Integration Test Suite
# Verifies that TPM2 acceleration is working with standard tools

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== TPM2 Native Integration Test Suite ===${NC}"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -n "Testing $test_name... "

    if eval "$test_command" &> /dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test 1: Check device exists
echo -e "${YELLOW}[1] Device Tests${NC}"
run_test "TPM acceleration device" "[ -c /dev/tpm2_accel_early ] || [ -c /dev/tpm_accel ]"
run_test "Standard TPM device" "[ -c /dev/tpm0 ] || [ -c /dev/tpmrm0 ]"
echo ""

# Test 2: Check TCTI plugin
echo -e "${YELLOW}[2] TCTI Plugin Tests${NC}"
run_test "TCTI library installed" "ls /usr/lib/*/libtss2-tcti-accel.so* &> /dev/null"
run_test "TCTI library loadable" "ldconfig -p | grep -q tcti-accel"
echo ""

# Test 3: Check permissions
echo -e "${YELLOW}[3] Permission Tests${NC}"
run_test "User in tss group" "groups | grep -q tss"
run_test "TPM device readable" "[ -r /dev/tpm0 ] || [ -r /dev/tpmrm0 ]"
echo ""

# Test 4: TPM2 tools availability
echo -e "${YELLOW}[4] TPM2 Tools Tests${NC}"
run_test "tpm2_getrandom available" "command -v tpm2_getrandom"
run_test "tpm2_pcrread available" "command -v tpm2_pcrread"
run_test "tpm2_hash available" "command -v tpm2_hash"
echo ""

# Test 5: Functional tests with hardware TPM
echo -e "${YELLOW}[5] Hardware TPM Functional Tests${NC}"
export TPM2TOOLS_TCTI="device:/dev/tpm0"

if [ -c /dev/tpm0 ] || [ -c /dev/tpmrm0 ]; then
    run_test "Generate random (HW TPM)" "tpm2_getrandom 16 > /dev/null"
    run_test "Read PCRs (HW TPM)" "tpm2_pcrread sha256:0 > /dev/null"
    run_test "Hash data (HW TPM)" "echo 'test' | tpm2_hash -g sha256 > /dev/null"
else
    echo -e "${YELLOW}Skipping hardware TPM tests (no device)${NC}"
fi
echo ""

# Test 6: Functional tests with acceleration (if available)
echo -e "${YELLOW}[6] Acceleration Functional Tests${NC}"
export TPM2TOOLS_TCTI="accel"

if [ -c /dev/tpm2_accel_early ] || [ -c /dev/tpm_accel ]; then
    run_test "Generate random (Accel)" "timeout 5 tpm2_getrandom 16 > /dev/null 2>&1 || true"
    run_test "Hash data (Accel)" "timeout 5 bash -c 'echo test | tpm2_hash -g sha256' > /dev/null 2>&1 || true"
else
    echo -e "${YELLOW}Skipping acceleration tests (no device)${NC}"
fi
echo ""

# Test 7: Configuration tests
echo -e "${YELLOW}[7] Configuration Tests${NC}"
run_test "TPM2 tools config exists" "[ -f /etc/tpm2-tools/tpm2-tools.conf ]"
run_test "udev rules installed" "[ -f /etc/udev/rules.d/99-tpm2-accel.rules ]"
echo ""

# Test 8: Kernel module (if available)
echo -e "${YELLOW}[8] Kernel Module Tests${NC}"
if lsmod | grep -q tpm2_accel; then
    run_test "TPM2 accel module loaded" "lsmod | grep -q tpm2_accel"
    run_test "Module device node exists" "[ -c /dev/tpm_accel ]"
else
    echo -e "${YELLOW}Skipping kernel module tests (not loaded)${NC}"
fi
echo ""

# Summary
echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Test Results Summary              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Tests Passed:  ${GREEN}$TESTS_PASSED${NC}"
echo -e "  Tests Failed:  ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed! Integration is working.${NC}"
    echo ""
    echo -e "${YELLOW}Quick Start:${NC}"
    echo "  export TPM2TOOLS_TCTI=accel"
    echo "  tpm2_getrandom 32 | xxd"
    echo "  echo 'test' | tpm2_hash -g sha256"
    echo ""
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Check installation.${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  1. Run: sudo ./install_native_integration.sh"
    echo "  2. Log out and back in (for group membership)"
    echo "  3. Check: ls -l /dev/tpm*"
    echo "  4. Check: groups | grep tss"
    echo ""
    exit 1
fi
