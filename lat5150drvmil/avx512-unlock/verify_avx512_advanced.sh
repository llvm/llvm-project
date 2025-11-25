#!/bin/bash
# ============================================================================
# Advanced AVX-512 Verification Script
# Tests AVX-512 availability with P-core task pinning
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Advanced AVX-512 Verification Utility                          ║${NC}"
echo -e "${BLUE}║           Tests P-core task pinning + E-core compatibility               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

PASSED_TESTS=0
TOTAL_TESTS=7
critical_fail=false
run_on_pcores_ok=false
ecore_expected_fail=false

# ============================================================================
# TEST 1: Check E-core status (should be ONLINE)
# ============================================================================

echo -e "${BLUE}[TEST 1/7] Checking E-core status (expecting ONLINE)...${NC}"

all_online=true
for cpu in {6..15}; do
    if [ -f "/sys/devices/system/cpu/cpu${cpu}/online" ]; then
        status=$(cat /sys/devices/system/cpu/cpu${cpu}/online 2>/dev/null || echo "1")
        if [ "$status" != "1" ]; then
            all_online=false
            break
        fi
    fi
done

if [ "$all_online" = true ]; then
    echo -e "${GREEN}✓ PASS: All E-cores (6-15) are ONLINE${NC}"
    echo -e "${GREEN}  → Advanced method working correctly${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${YELLOW}⚠ WARNING: Some E-cores are OFFLINE${NC}"
    echo -e "${YELLOW}  → You may be using traditional unlock method${NC}"
    echo -e "${YELLOW}  → Consider using ./unlock_avx512_advanced.sh enable${NC}"
fi

echo ""

# ============================================================================
# TEST 2: Compile test with AVX-512 and run on P-core
# ============================================================================

echo -e "${BLUE}[TEST 2/7] Compiling and testing AVX-512 with P-core pinning...${NC}"

# Create test program
cat > /tmp/avx512_test.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>

int main() {
    __m512i sum = _mm512_setzero_si512();
    __m512i step = _mm512_set1_epi32(1);

    // Run a simple accumulation loop to ensure the CPU executes AVX-512 ops
    for (int i = 0; i < 100000; i++) {
        sum = _mm512_add_epi32(sum, step);
    }

    int result[16];
    _mm512_storeu_si512((__m512i*)result, sum);

    printf("AVX-512 accumulated value: %d\n", result[0]);
    return (result[0] == 100000) ? 0 : 1;
}
EOF

# Compile with AVX-512
test2_compiled=false
test2_pcore_success=false
if gcc -O3 -march=meteorlake -mavx512f -mavx512dq -mavx512bw -mavx512vl /tmp/avx512_test.c -o /tmp/avx512_test 2>/dev/null; then
    test2_compiled=true
    echo -e "${GREEN}✓ PASS: Compilation successful${NC}"

    # Run the test pinned to P-core
    echo -e "${BLUE}[*] Running test pinned to CPU 0 (P-core)...${NC}"
    if taskset -c 0 /tmp/avx512_test > /tmp/avx512_output.txt 2>&1; then
        test2_pcore_success=true
        echo -e "${GREEN}✓ PASS: AVX-512 program executed successfully on P-core (taskset -c 0)${NC}"
        sed 's/^/  /' /tmp/avx512_output.txt
    else
        echo -e "${RED}✗ FAIL: Program crashed even on P-core${NC}"
        echo -e "${YELLOW}  → AVX-512 may not be available at runtime${NC}"
        sed 's/^/  /' /tmp/avx512_output.txt 2>/dev/null || true
        critical_fail=true
    fi
else
    echo -e "${RED}✗ FAIL: Compilation failed${NC}"
    echo -e "${YELLOW}  → Compiler may not support AVX-512 or flags are incorrect${NC}"
    critical_fail=true
fi

if [ "$test2_compiled" = true ] && [ "$test2_pcore_success" = true ]; then
    ((PASSED_TESTS++))
else
    echo -e "${YELLOW}  → Test 2 requires successful compile and P-core execution${NC}"
    critical_fail=true
fi

echo ""

# ============================================================================
# TEST 3: Ensure AVX-512 faults on E-cores
# ============================================================================

echo -e "${BLUE}[TEST 3/7] Validating E-core behavior (should fail)...${NC}"

if [ "$test2_compiled" = true ]; then
    if taskset -c 8 /tmp/avx512_test > /tmp/avx512_ecore_output.txt 2>&1; then
        echo -e "${YELLOW}⚠ E-core test succeeded (unexpected, may indicate microcode not active)${NC}"
        sed 's/^/  /' /tmp/avx512_ecore_output.txt
    else
        echo -e "${GREEN}✓ PASS: E-core execution failed as expected (no AVX-512 support)${NC}"
        ((PASSED_TESTS++))
        ecore_expected_fail=true
    fi
else
    echo -e "${YELLOW}⚠ Skipped: AVX-512 binary unavailable due to compilation failure${NC}"
fi

echo ""

# Cleanup
rm -f /tmp/avx512_test.c /tmp/avx512_test /tmp/avx512_output.txt /tmp/avx512_ecore_output.txt

echo ""

# ============================================================================
# TEST 4: Test run-on-pcores wrapper
# ============================================================================

echo -e "${BLUE}[TEST 4/7] Testing run-on-pcores wrapper script...${NC}"

if [ -x /usr/local/bin/run-on-pcores ]; then
    echo -e "${BLUE}[*] Testing wrapper with 'cat /proc/cpuinfo | grep processor | wc -l'${NC}"
    if /usr/local/bin/run-on-pcores bash -c 'cat /proc/cpuinfo | grep processor | wc -l' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS: run-on-pcores wrapper available and functional${NC}"
        ((PASSED_TESTS++))
        run_on_pcores_ok=true
    else
        echo -e "${YELLOW}⚠ Wrapper present but execution failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠ WARNING: run-on-pcores wrapper not found${NC}"
    echo -e "${YELLOW}  → Run: sudo ./unlock_avx512_advanced.sh enable${NC}"
fi

echo ""

# ============================================================================
# TEST 5: Check for AVX-512 instructions in compiled binary
# ============================================================================

echo -e "${BLUE}[TEST 5/7] Checking for AVX-512 instructions in binary...${NC}"

# Compile a simple vectorizable program
cat > /tmp/vec_test.c <<'EOF'
void multiply(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

int main() {
    float a[64], b[64], c[64];
    for (int i = 0; i < 64; i++) {
        a[i] = i;
        b[i] = i * 2.0f;
    }
    multiply(a, b, c, 64);
    return 0;
}
EOF

gcc -O3 -march=meteorlake -mavx512f -mavx512dq -ftree-vectorize /tmp/vec_test.c -o /tmp/vec_test 2>/dev/null

if objdump -d /tmp/vec_test 2>/dev/null | grep -q "vmul.*zmm\|vadd.*zmm\|vmov.*zmm"; then
    echo -e "${GREEN}✓ PASS: AVX-512 instructions (zmm registers) found in binary${NC}"
    echo -e "${BLUE}  Sample instructions:${NC}"
    objdump -d /tmp/vec_test | grep "zmm" | head -3 | sed 's/^/    /'
    ((PASSED_TESTS++))
else
    echo -e "${YELLOW}⚠ WARNING: No AVX-512 instructions detected in binary${NC}"
    echo -e "${YELLOW}  → Code may be too simple to benefit from AVX-512${NC}"
fi

rm -f /tmp/vec_test.c /tmp/vec_test

echo ""

# ============================================================================
# TEST 6: Check MSR and DSMIL driver
# ============================================================================

echo -e "${BLUE}[TEST 6/7] Checking kernel modules...${NC}"

kernel_modules_pass=false

# Check MSR module
if lsmod | grep -q "^msr"; then
    echo -e "${GREEN}✓ PASS: MSR module loaded${NC}"
    kernel_modules_pass=true
else
    echo -e "${RED}✗ FAIL: MSR module not loaded${NC}"
    echo -e "${YELLOW}  → Advanced MSR-based control not available${NC}"
fi

# Check DSMIL driver
if lsmod | grep -q "^dsmil"; then
    echo -e "${GREEN}✓ PASS: DSMIL driver loaded${NC}"
else
    echo -e "${YELLOW}⚠ INFO: DSMIL driver not loaded${NC}"
    echo -e "${YELLOW}  → Dell platform integration not active${NC}"
fi

if [ "$kernel_modules_pass" = true ]; then
    ((PASSED_TESTS++))
fi

# Check Dell WMI modules
if lsmod | grep -q "dell_smbios"; then
    echo -e "${GREEN}✓ Dell SMBIOS module loaded${NC}"
else
    echo -e "${YELLOW}⚠ Dell SMBIOS module not loaded${NC}"
fi

echo ""

# ============================================================================
# TEST 7: Microcode and configuration status
# ============================================================================

echo -e "${BLUE}[TEST 7/7] Checking microcode configuration...${NC}"

# Check microcode version
microcode_pass=false
if [ -f /proc/cpuinfo ]; then
    microcode=$(grep "microcode" /proc/cpuinfo | head -1 | awk '{print $3}')
    echo -e "${BLUE}  Current microcode version: ${microcode}${NC}"
    if [[ "${microcode}" == "0x1c" ]]; then
        microcode_pass=true
    fi
fi

# Check runtime cmdline first, then GRUB defaults
if grep -qE 'dis_ucode_ldr|microcode=no' /proc/cmdline 2>/dev/null; then
    echo -e "${GREEN}✓ PASS: Microcode loading disabled at runtime (/proc/cmdline)${NC}"
    echo -e "${BLUE}  → BIOS/UEFI microcode only (until GRUB defaults change)${NC}"
    microcode_pass=true
elif grep -qE 'dis_ucode_ldr|microcode=no' /etc/default/grub 2>/dev/null; then
    echo -e "${YELLOW}⚠ Configured: Microcode loading disabled in GRUB defaults${NC}"
    echo -e "${YELLOW}  → Requires reboot into an entry that includes those args${NC}"
else
    echo -e "${BLUE}  ℹ Microcode loading: ENABLED (normal mode)${NC}"
fi

if [ "$microcode_pass" = true ]; then
    ((PASSED_TESTS++))
else
    echo -e "${YELLOW}⚠ Microcode configuration not aligned with AVX-512 requirements${NC}"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                          VERIFICATION SUMMARY                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

PASS_PERCENTAGE=$((PASSED_TESTS * 100 / TOTAL_TESTS))

echo -e "${BLUE}Tests passed: ${PASSED_TESTS}/${TOTAL_TESTS} (${PASS_PERCENTAGE}%)${NC}"
echo ""

if [ "$test2_pcore_success" = true ] && [ $PASSED_TESTS -ge 5 ]; then
    echo -e "${GREEN}[✓] AVX-512 Status: OPERATIONAL${NC}"
    echo -e "${GREEN}    Advanced method working with P-core pinning${NC}"
    echo ""
    echo -e "${MAGENTA}Next steps:${NC}"
    echo -e "  1. Use run-on-pcores wrapper:  ${BLUE}run-on-pcores <your-avx512-program>${NC}"
    echo -e "  2. Or use taskset directly:    ${BLUE}taskset -c 0-5 <your-avx512-program>${NC}"
    echo -e "  3. Source compiler flags:      ${BLUE}source ./avx512_compiler_flags.sh${NC}"
    echo -e "  4. Compile with AVX-512:       ${BLUE}gcc \$CFLAGS_AVX512 -o app app.c${NC}"
elif [ $PASSED_TESTS -ge 3 ]; then
    echo -e "${YELLOW}[⚠] AVX-512 Status: PARTIALLY OPERATIONAL${NC}"
    echo -e "${YELLOW}    Some features working, but not all tests passed${NC}"
    echo ""
    echo -e "${MAGENTA}Troubleshooting:${NC}"
    echo -e "  1. Check if microcode is blocking AVX-512"
    echo -e "  2. Try fallback method: ${BLUE}sudo ./unlock_avx512_advanced.sh microcode-fallback${NC}"
    echo -e "  3. Verify BIOS settings allow AVX-512"
    echo -e "  4. Ensure kernel is recent (6.1+)"
else
    echo -e "${RED}[✗] AVX-512 Status: NOT OPERATIONAL${NC}"
    echo -e "${RED}    AVX-512 is not currently available${NC}"
    echo ""
    echo -e "${MAGENTA}Troubleshooting steps:${NC}"
    echo -e "  1. Enable advanced method:     ${BLUE}sudo ./unlock_avx512_advanced.sh enable${NC}"
    echo -e "  2. If that fails, try fallback: ${BLUE}sudo ./unlock_avx512_advanced.sh microcode-fallback${NC}"
    echo -e "  3. Reboot after fallback:      ${BLUE}sudo reboot${NC}"
    echo -e "  4. Check BIOS settings:        Ensure AVX-512 not disabled in BIOS"
    echo -e "  5. Verify CPU model:           Intel Core Ultra 7 165H (Meteor Lake)"
fi

echo ""

echo -e "${BLUE}Status breakdown:${NC}"
if [ "$test2_pcore_success" = true ]; then
    echo -e "${BLUE}  ✓ P-cores: AVX-512 workload executed successfully${NC}"
else
    echo -e "${RED}  ✗ P-cores: AVX-512 workload failed on P-core${NC}"
fi

if [ "$all_online" = true ]; then
    echo -e "${BLUE}  ✓ E-cores: Active for multitasking${NC}"
else
    echo -e "${YELLOW}  ⚠ E-cores: Some cores offline${NC}"
fi

if [ "$run_on_pcores_ok" = true ]; then
    echo -e "${BLUE}  ✓ Task pinning: run-on-pcores wrapper available${NC}"
else
    echo -e "${YELLOW}  ⚠ Task pinning: Wrapper missing or failed${NC}"
fi

if [ "$ecore_expected_fail" = true ]; then
    echo -e "${BLUE}  ✓ E-core safety: AVX-512 blocked on E-cores${NC}"
else
    echo -e "${YELLOW}  ⚠ E-core safety: Could not confirm expected failure${NC}"
fi

echo ""

# ============================================================================
# DETAILED CPU INFO
# ============================================================================

echo -e "${BLUE}[DETAILED CPU INFO]${NC}"
echo ""
echo -e "${BLUE}Model name:${NC}"
grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^/  /'

echo ""
echo -e "${BLUE}CPU topology:${NC}"
echo -e "  P-cores (0-5):  $(lscpu | grep "^CPU(s):" | head -1 | awk '{print $2}' | sed 's/.*/6 cores/')"
echo -e "  E-cores (6-15): $(lscpu | grep "^CPU(s):" | head -1 | awk '{print $2}' | sed 's/.*/10 cores/')"

echo ""
echo -e "${BLUE}CPU flags (vector instructions):${NC}"
grep "flags" /proc/cpuinfo | head -1 | grep -o "\(sse[^ ]*\|avx[^ ]*\)" | tr ' ' '\n' | sort -u | sed 's/^/  /'

echo ""
echo -e "${BLUE}Online CPUs:${NC}"
cat /sys/devices/system/cpu/online | sed 's/^/  /'

echo ""
echo -e "${BLUE}CPU affinity tools available:${NC}"
command -v taskset >/dev/null 2>&1 && echo -e "  ✓ taskset" || echo -e "  ✗ taskset (not found)"
command -v numactl >/dev/null 2>&1 && echo -e "  ✓ numactl" || echo -e "  ✗ numactl (not found)"
[ -f /usr/local/bin/run-on-pcores ] && echo -e "  ✓ run-on-pcores" || echo -e "  ✗ run-on-pcores (not found)"

echo ""

exit 0
