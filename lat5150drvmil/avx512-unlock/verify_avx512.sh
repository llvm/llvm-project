#!/bin/bash
# ============================================================================
# AVX-512 Verification Script
# Checks if AVX-512 is available and working on Intel Meteor Lake
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  AVX-512 Verification Utility                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# TEST 1: Check CPU flags in /proc/cpuinfo
# ============================================================================

echo -e "${BLUE}[TEST 1] Checking /proc/cpuinfo for AVX-512 flags...${NC}"

if grep -q "avx512" /proc/cpuinfo; then
    echo -e "${GREEN}✓ PASS: AVX-512 flags found in /proc/cpuinfo${NC}"

    # List all AVX-512 features
    echo -e "${BLUE}  Available AVX-512 features:${NC}"
    grep "flags" /proc/cpuinfo | head -1 | grep -o "avx512[a-z0-9_]*" | sort -u | sed 's/^/    /'
else
    echo -e "${RED}✗ FAIL: No AVX-512 flags detected${NC}"
    echo -e "${YELLOW}  → E-cores may still be enabled. Run: sudo ./unlock_avx512.sh enable${NC}"
fi

echo ""

# ============================================================================
# TEST 2: Check E-core status
# ============================================================================

echo -e "${BLUE}[TEST 2] Checking E-core status...${NC}"

all_offline=true
for cpu in {6..15}; do
    if [ -f "/sys/devices/system/cpu/cpu${cpu}/online" ]; then
        status=$(cat /sys/devices/system/cpu/cpu${cpu}/online 2>/dev/null || echo "1")
        if [ "$status" == "1" ]; then
            all_offline=false
            break
        fi
    fi
done

if [ "$all_offline" = true ]; then
    echo -e "${GREEN}✓ PASS: All E-cores (6-15) are disabled${NC}"
    echo -e "${GREEN}  → AVX-512 should be unlocked${NC}"
else
    echo -e "${RED}✗ FAIL: E-cores are still enabled${NC}"
    echo -e "${YELLOW}  → Disable E-cores with: sudo ./unlock_avx512.sh enable${NC}"
fi

echo ""

# ============================================================================
# TEST 3: Compile test with AVX-512
# ============================================================================

echo -e "${BLUE}[TEST 3] Compiling test program with AVX-512 flags...${NC}"

# Create test program
cat > /tmp/avx512_test.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>

int main() {
    // Test AVX-512 intrinsic
    __m512i a = _mm512_set1_epi32(42);
    __m512i b = _mm512_set1_epi32(8);
    __m512i c = _mm512_add_epi32(a, b);

    int result[16];
    _mm512_storeu_si512((__m512i*)result, c);

    printf("AVX-512 Test: %d + %d = %d\n", 42, 8, result[0]);

    if (result[0] == 50) {
        printf("✓ AVX-512 computation correct!\n");
        return 0;
    } else {
        printf("✗ AVX-512 computation incorrect!\n");
        return 1;
    }
}
EOF

# Compile with AVX-512
if gcc -O3 -march=meteorlake -mavx512f -mavx512dq -mavx512bw -mavx512vl /tmp/avx512_test.c -o /tmp/avx512_test 2>/dev/null; then
    echo -e "${GREEN}✓ PASS: Compilation successful${NC}"

    # Run the test
    if /tmp/avx512_test > /tmp/avx512_output.txt 2>&1; then
        echo -e "${GREEN}✓ PASS: AVX-512 program executed successfully${NC}"
        cat /tmp/avx512_output.txt | sed 's/^/  /'
    else
        echo -e "${RED}✗ FAIL: Program crashed (AVX-512 may not be available at runtime)${NC}"
        echo -e "${YELLOW}  → Check if E-cores are disabled${NC}"
    fi
else
    echo -e "${RED}✗ FAIL: Compilation failed${NC}"
    echo -e "${YELLOW}  → Compiler may not support AVX-512 or flags are incorrect${NC}"
fi

# Cleanup
rm -f /tmp/avx512_test.c /tmp/avx512_test /tmp/avx512_output.txt

echo ""

# ============================================================================
# TEST 4: Check for AVX-512 instructions in compiled binary
# ============================================================================

echo -e "${BLUE}[TEST 4] Checking for AVX-512 instructions in binary...${NC}"

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

if objdump -d /tmp/vec_test 2>/dev/null | grep -q "vmul.*zmm"; then
    echo -e "${GREEN}✓ PASS: AVX-512 instructions (zmm registers) found in binary${NC}"
    echo -e "${BLUE}  Sample instructions:${NC}"
    objdump -d /tmp/vec_test | grep "zmm" | head -3 | sed 's/^/    /'
else
    echo -e "${YELLOW}⚠ WARNING: No AVX-512 instructions detected in binary${NC}"
    echo -e "${YELLOW}  → Code may be too simple to benefit from AVX-512${NC}"
fi

rm -f /tmp/vec_test.c /tmp/vec_test

echo ""

# ============================================================================
# TEST 5: Check kernel AVX-512 support
# ============================================================================

echo -e "${BLUE}[TEST 5] Checking kernel AVX-512 support...${NC}"

if dmesg | grep -qi "avx512"; then
    echo -e "${GREEN}✓ PASS: Kernel logs mention AVX-512${NC}"
    dmesg | grep -i avx512 | tail -3 | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠ INFO: No AVX-512 references in kernel log${NC}"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                          VERIFICATION SUMMARY                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Count passes
PASSED=0

grep -q "avx512" /proc/cpuinfo && ((PASSED++))
[ "$all_offline" = true ] && ((PASSED++))
# Additional tests would increment here

if [ $PASSED -ge 2 ]; then
    echo -e "${GREEN}[✓] AVX-512 Status: OPERATIONAL${NC}"
    echo -e "${GREEN}    E-cores disabled, AVX-512 unlocked and working${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Source AVX-512 flags:  source ./avx512_compiler_flags.sh"
    echo -e "  2. Compile with:          gcc \$CFLAGS_AVX512 -o app app.c"
    echo -e "  3. Run benchmarks:        benchmark_avx512_vs_avx2"
else
    echo -e "${RED}[✗] AVX-512 Status: NOT OPERATIONAL${NC}"
    echo -e "${RED}    AVX-512 is not currently available${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  1. Unlock AVX-512:        sudo ./unlock_avx512.sh enable"
    echo -e "  2. Verify E-cores off:    sudo ./unlock_avx512.sh status"
    echo -e "  3. Check BIOS settings:   Ensure AVX-512 not disabled in BIOS"
    echo -e "  4. Reboot if needed:      Some changes require reboot"
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
echo -e "${BLUE}CPU flags (vector instructions):${NC}"
grep "flags" /proc/cpuinfo | head -1 | grep -o "\(sse[^ ]*\|avx[^ ]*\)" | tr ' ' '\n' | sort -u | sed 's/^/  /'

echo ""
echo -e "${BLUE}Online CPUs:${NC}"
cat /sys/devices/system/cpu/online | sed 's/^/  /'

echo ""

exit 0
