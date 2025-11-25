#!/bin/bash
# Verify DSMIL package installation and system readiness
# Usage: ./verify-installation.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0

print_header() {
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}\n"
}

print_test() { echo -e "${YELLOW}[TEST]${NC} $1"; }
print_pass() { echo -e "${GREEN}  ✓ PASS${NC} $1"; ((PASSED++)); }
print_fail() { echo -e "${RED}  ✗ FAIL${NC} $1"; ((FAILED++)); }
print_info() { echo -e "${BLUE}  ℹ INFO${NC} $1"; }

print_header "DSMIL Installation Verification"

# Test 1: Check if packages are installed
print_test "Checking installed packages..."
for pkg in dsmil-platform dell-milspec-tools tpm2-accel-examples dsmil-complete; do
    if dpkg -l | grep -q "^ii.*$pkg"; then
        VERSION=$(dpkg -l | grep "^ii.*$pkg" | awk '{print $3}')
        print_pass "$pkg ($VERSION) is installed"
    else
        print_fail "$pkg is NOT installed"
    fi
done

# Test 2: Check executables
print_test "Checking installed executables..."
COMMANDS=(
    "dsmil-status"
    "dsmil-test"
    "milspec-control"
    "milspec-monitor"
    "tpm2-accel-status"
    "milspec-emergency-stop"
)

for cmd in "${COMMANDS[@]}"; do
    if command -v "$cmd" &> /dev/null; then
        LOCATION=$(which "$cmd")
        print_pass "$cmd found at $LOCATION"
    else
        print_fail "$cmd not found in PATH"
    fi
done

# Test 3: Check Python dependencies
print_test "Checking Python environment..."
if python3 --version &> /dev/null; then
    PYVER=$(python3 --version)
    print_pass "Python available: $PYVER"
else
    print_fail "Python 3 not found"
fi

# Test 4: Check build prerequisites
print_test "Checking build prerequisites..."
for tool in gcc make dpkg-deb; do
    if command -v "$tool" &> /dev/null; then
        VER=$($tool --version 2>&1 | head -1)
        print_pass "$tool available: $VER"
    else
        print_fail "$tool not found (needed for building)"
    fi
done

# Test 5: Check kernel headers
print_test "Checking kernel headers..."
KERNEL=$(uname -r)
if [ -d "/usr/src/linux-headers-$KERNEL" ]; then
    print_pass "Kernel headers found for $KERNEL"
else
    print_fail "Kernel headers missing for $KERNEL"
    print_info "Install with: sudo apt-get install linux-headers-$KERNEL"
fi

# Test 6: Check Rust availability
print_test "Checking Rust toolchain..."
if command -v rustc &> /dev/null; then
    if rustc --version &> /dev/null; then
        RUSTVER=$(rustc --version)
        print_pass "Rust available: $RUSTVER"
    else
        print_fail "Rust installed but not working (no toolchain)"
        print_info "This is OK - build will use C stubs"
    fi
else
    print_info "Rust not found (this is OK - build will use C stubs)"
fi

# Test 7: Check documentation
print_test "Checking documentation..."
DOCS=(
    "/usr/share/doc/tpm2-accel-examples"
    "/usr/share/dell-milspec"
)

for doc in "${DOCS[@]}"; do
    if [ -d "$doc" ]; then
        COUNT=$(find "$doc" -type f | wc -l)
        print_pass "Documentation found: $doc ($COUNT files)"
    else
        print_fail "Documentation missing: $doc"
    fi
done

# Test 8: Check if dsmil.py exists
print_test "Checking dsmil.py build system..."
if [ -f "../dsmil.py" ] || [ -f "dsmil.py" ]; then
    print_pass "dsmil.py found"
else
    print_fail "dsmil.py not found"
fi

# Test 9: Check if kernel modules can be loaded
print_test "Checking kernel module support..."
if [ -w "/dev" ]; then
    print_pass "Can access /dev (needed for driver loading)"
else
    print_fail "Cannot access /dev"
fi

# Test 10: Check system permissions
print_test "Checking system permissions..."
if [ "$EUID" -eq 0 ]; then
    print_pass "Running as root (correct for driver operations)"
else
    print_info "Running as user (sudo needed for driver operations)"
fi

# Summary
print_header "Verification Summary"
TOTAL=$((PASSED + FAILED))
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo -e "Total:  $TOTAL"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! System is ready.${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Build kernel drivers: sudo python3 dsmil.py build-auto"
    echo "  2. Load drivers: sudo python3 dsmil.py load-all"
    echo "  3. Check status: sudo python3 dsmil.py status"
    exit 0
else
    echo -e "${YELLOW}⚠ Some checks failed. Review failures above.${NC}"
    echo ""
    echo -e "${BLUE}Common fixes:${NC}"
    echo "  • Install packages: sudo ./install-all-debs.sh"
    echo "  • Install kernel headers: sudo apt-get install linux-headers-\$(uname -r)"
    echo "  • Install build tools: sudo apt-get install build-essential dpkg-dev"
    exit 1
fi
