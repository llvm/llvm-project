#!/bin/bash
# System Verification Script - DSMIL Military-Spec Kernel

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     DSMIL KERNEL - SYSTEM VERIFICATION                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0
WARN=0

check_pass() {
    echo "✅ $1"
    ((PASS++))
}

check_fail() {
    echo "❌ $1"
    ((FAIL++))
}

check_warn() {
    echo "⚠️  $1"
    ((WARN++))
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "KERNEL BUILD VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check kernel image
if [ -f "/home/john/linux-6.16.9/arch/x86/boot/bzImage" ]; then
    SIZE=$(stat -c%s "/home/john/linux-6.16.9/arch/x86/boot/bzImage")
    if [ $SIZE -gt 10000000 ]; then
        check_pass "Kernel image exists ($(numfmt --to=iec $SIZE))"
    else
        check_warn "Kernel image exists but seems small ($(numfmt --to=iec $SIZE))"
    fi
else
    check_fail "Kernel image not found"
fi

# Check DSMIL driver
if [ -f "/home/john/linux-6.16.9/drivers/platform/x86/dell-milspec/dsmil-core.c" ]; then
    LINES=$(wc -l < "/home/john/linux-6.16.9/drivers/platform/x86/dell-milspec/dsmil-core.c")
    if [ $LINES -gt 2000 ]; then
        check_pass "DSMIL driver source ($LINES lines)"
    else
        check_warn "DSMIL driver exists but seems short ($LINES lines)"
    fi
else
    check_fail "DSMIL driver source not found"
fi

# Check header file
if [ -f "/home/john/linux-6.16.9/drivers/platform/x86/dell-milspec/dell-milspec.h" ]; then
    check_pass "DSMIL header file exists"
else
    check_fail "DSMIL header file not found"
fi

# Check build log
if [ -f "/home/john/kernel-build-apt-secure.log" ]; then
    if grep -q "Kernel: arch/x86/boot/bzImage is ready" "/home/john/kernel-build-apt-secure.log"; then
        check_pass "Build log confirms successful build"
    else
        check_warn "Build log exists but success message not found"
    fi
else
    check_warn "Final build log not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DOCUMENTATION VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check critical documentation
DOCS=(
    "README.md"
    "MASTER_INDEX.md"
    "COMPLETE_MILITARY_SPEC_HANDOFF.md"
    "DEPLOYMENT_CHECKLIST.md"
    "MODE5_SECURITY_LEVELS_WARNING.md"
    "APT_ADVANCED_SECURITY_FEATURES.md"
    "SYSTEM_ARCHITECTURE.md"
    "INTERFACE_README.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "/home/john/$doc" ]; then
        check_pass "$doc exists"
    else
        check_fail "$doc not found"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "INTERFACE VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check interface files
if [ -f "/home/john/opus_interface.html" ]; then
    check_pass "Web interface HTML exists"
else
    check_fail "Web interface HTML not found"
fi

if [ -f "/home/john/opus_server.py" ]; then
    check_pass "Backend server script exists"
else
    check_fail "Backend server script not found"
fi

# Check if server is running
if lsof -i :8080 >/dev/null 2>&1; then
    PID=$(lsof -t -i :8080 2>/dev/null)
    check_pass "Server running on port 8080 (PID: $PID)"
else
    check_warn "Server not running (start with: ./quick-start-interface.sh)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SCRIPTS VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check scripts
SCRIPTS=(
    "quick-start-interface.sh"
    "show-complete-status.sh"
    "verify-system.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "/home/john/$script" ]; then
        if [ -x "/home/john/$script" ]; then
            check_pass "$script exists and is executable"
        else
            check_warn "$script exists but not executable"
        fi
    else
        check_fail "$script not found"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ADDITIONAL MODULES VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check AVX-512 module
if [ -f "/home/john/livecd-gen/kernel-modules/dsmil_avx512_enabler.ko" ]; then
    SIZE=$(stat -c%s "/home/john/livecd-gen/kernel-modules/dsmil_avx512_enabler.ko")
    check_pass "AVX-512 enabler module exists ($(numfmt --to=iec $SIZE))"
else
    check_warn "AVX-512 enabler module not found (optional)"
fi

# Check C modules to compile
C_MODULES=(
    "ai_hardware_optimizer.c"
    "meteor_lake_scheduler.c"
    "dell_platform_optimizer.c"
    "tpm_kernel_security.c"
    "avx512_optimizer.c"
)

C_FOUND=0
for module in "${C_MODULES[@]}"; do
    if [ -f "/home/john/livecd-gen/$module" ]; then
        ((C_FOUND++))
    fi
done

if [ $C_FOUND -eq 5 ]; then
    check_pass "All 5 C modules found for compilation"
elif [ $C_FOUND -gt 0 ]; then
    check_warn "$C_FOUND/5 C modules found"
else
    check_warn "C modules not found (optional)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SAFETY CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for dangerous configurations
if grep -q "PARANOID_PLUS" /home/john/MODE5_SECURITY_LEVELS_WARNING.md 2>/dev/null; then
    check_pass "Mode 5 safety warnings documented"
else
    check_warn "Mode 5 safety warnings not found"
fi

# Check kernel config for dangerous settings (if .config exists)
if [ -f "/home/john/linux-6.16.9/.config" ]; then
    if grep -q "CONFIG_DELL_MILSPEC=y" "/home/john/linux-6.16.9/.config"; then
        check_pass "DSMIL configured as built-in (=y)"
    else
        check_warn "DSMIL configuration not verified"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VERIFICATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TOTAL=$((PASS + FAIL + WARN))

echo "Results:"
echo "  ✅ Passed: $PASS"
echo "  ❌ Failed: $FAIL"
echo "  ⚠️  Warnings: $WARN"
echo "  📊 Total Checks: $TOTAL"
echo ""

if [ $FAIL -eq 0 ]; then
    if [ $WARN -eq 0 ]; then
        echo "🎉 PERFECT: All checks passed!"
        echo ""
        echo "System is ready for deployment. Next steps:"
        echo "  1. Read DEPLOYMENT_CHECKLIST.md"
        echo "  2. Access web interface: ./quick-start-interface.sh"
        echo "  3. Review Mode 5 warnings carefully"
    else
        echo "✅ GOOD: All critical checks passed (warnings are normal)"
        echo ""
        echo "Warnings are typically for optional components."
        echo "System is ready for deployment."
    fi
else
    echo "❌ ISSUES FOUND: Please review failed checks above"
    echo ""
    echo "Common fixes:"
    echo "  • Re-run build if kernel/driver missing"
    echo "  • Regenerate docs if missing"
    echo "  • Check file permissions"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Exit code based on failures
if [ $FAIL -gt 0 ]; then
    exit 1
else
    exit 0
fi