#!/bin/bash
################################################################################
# DSMIL FMA Instruction Mitigation Verification Script
################################################################################
# Checks if the Rust library was built without FMA instructions that cause
# objtool decoding failures
#
# Usage:
#   ./check-fma-mitigation.sh
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

echo -e "${CYAN}${BOLD}DSMIL FMA Instruction Mitigation Check${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

RUST_DIR="rust"
RUST_LIB="$RUST_DIR/libdsmil_rust.a"
RUST_OBJ="$RUST_DIR/libdsmil_rust.o"

# Check 1: Rust library exists
echo -e "${BOLD}Check 1: Rust library existence${NC}"
if [[ -f "$RUST_LIB" ]]; then
    LIB_SIZE=$(stat -c%s "$RUST_LIB")
    echo -e "  ${GREEN}✓${NC} Rust library found: $RUST_LIB ($(numfmt --to=iec-i --suffix=B $LIB_SIZE))"
else
    echo -e "  ${RED}✗${NC} Rust library not found: $RUST_LIB"
    echo "  Run: make rust-lib"
    exit 1
fi
echo ""

# Check 2: Rust object file exists
echo -e "${BOLD}Check 2: Rust object file existence${NC}"
if [[ -f "$RUST_OBJ" ]]; then
    OBJ_SIZE=$(stat -c%s "$RUST_OBJ")
    echo -e "  ${GREEN}✓${NC} Rust object found: $RUST_OBJ ($(numfmt --to=iec-i --suffix=B $OBJ_SIZE))"
else
    echo -e "  ${YELLOW}⚠${NC} Rust object not found: $RUST_OBJ"
    echo "  This is normal before first build"
fi
echo ""

# Check 3: Scan for FMA symbols in library
echo -e "${BOLD}Check 3: FMA symbol detection in library${NC}"
FMA_SYMBOLS=$(nm "$RUST_LIB" 2>/dev/null | grep -i "fma" || true)
if [[ -z "$FMA_SYMBOLS" ]]; then
    echo -e "  ${GREEN}✓${NC} No FMA symbols detected in library"
else
    echo -e "  ${YELLOW}⚠${NC} FMA symbols found in library:"
    echo "$FMA_SYMBOLS" | sed 's/^/    /'
    echo ""
    echo "  This may cause objtool failures"
fi
echo ""

# Check 4: Scan for FMA symbols in object file
if [[ -f "$RUST_OBJ" ]]; then
    echo -e "${BOLD}Check 4: FMA symbol detection in object${NC}"
    FMA_OBJ_SYMBOLS=$(nm "$RUST_OBJ" 2>/dev/null | grep -i "fma" || true)
    if [[ -z "$FMA_OBJ_SYMBOLS" ]]; then
        echo -e "  ${GREEN}✓${NC} No FMA symbols detected in object file"
    else
        echo -e "  ${YELLOW}⚠${NC} FMA symbols found in object file:"
        echo "$FMA_OBJ_SYMBOLS" | sed 's/^/    /'
        echo ""
        echo "  This may cause objtool failures"
    fi
    echo ""
fi

# Check 5: Rust source component
echo -e "${BOLD}Check 5: rust-src component${NC}"
if command -v rustup &> /dev/null; then
    if rustup component list 2>/dev/null | grep -q "rust-src (installed)"; then
        echo -e "  ${GREEN}✓${NC} rust-src component installed"
        echo "  Build can use -Z build-std to rebuild compiler_builtins"
    else
        echo -e "  ${YELLOW}⚠${NC} rust-src component NOT installed"
        echo "  Install with: rustup component add rust-src"
        echo "  Without this, pre-compiled stdlib (with FMA) will be used"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} rustup not found, cannot check rust-src"
fi
echo ""

# Check 6: Cargo config
echo -e "${BOLD}Check 6: Cargo configuration${NC}"
CARGO_CONFIG="$RUST_DIR/.cargo/config.toml"
if [[ -f "$CARGO_CONFIG" ]]; then
    echo -e "  ${GREEN}✓${NC} Cargo config found: $CARGO_CONFIG"
    if grep -q "build-std" "$CARGO_CONFIG"; then
        echo -e "  ${GREEN}✓${NC} build-std configuration present"
    else
        echo -e "  ${YELLOW}⚠${NC} build-std not configured"
    fi
    if grep -q "\-fma" "$CARGO_CONFIG"; then
        echo -e "  ${GREEN}✓${NC} FMA disabling flags present"
    else
        echo -e "  ${RED}✗${NC} FMA disabling flags missing"
    fi
else
    echo -e "  ${RED}✗${NC} Cargo config not found: $CARGO_CONFIG"
fi
echo ""

# Check 7: Disassemble and check for FMA instructions
echo -e "${BOLD}Check 7: FMA instruction scan (detailed)${NC}"
if command -v objdump &> /dev/null; then
    if [[ -f "$RUST_OBJ" ]]; then
        FMA_INSTRS=$(objdump -d "$RUST_OBJ" 2>/dev/null | grep -E "vfmadd|vfmsub|vfnmadd|vfnmsub|fma" || true)
        if [[ -z "$FMA_INSTRS" ]]; then
            echo -e "  ${GREEN}✓${NC} No FMA instructions found in disassembly"
        else
            echo -e "  ${RED}✗${NC} FMA instructions detected:"
            echo "$FMA_INSTRS" | head -10 | sed 's/^/    /'
            FMA_COUNT=$(echo "$FMA_INSTRS" | wc -l)
            if [[ $FMA_COUNT -gt 10 ]]; then
                echo -e "    ... and $((FMA_COUNT - 10)) more"
            fi
            echo ""
            echo "  ${BOLD}These instructions will cause objtool to fail!${NC}"
        fi
    else
        echo "  ${YELLOW}⚠${NC} Object file not found, skipping disassembly"
    fi
else
    echo "  ${YELLOW}⚠${NC} objdump not found, skipping instruction scan"
fi
echo ""

# Check 8: OBJECT_FILES_NON_STANDARD markers
echo -e "${BOLD}Check 8: Objtool bypass configuration${NC}"
if grep -q "OBJECT_FILES_NON_STANDARD_dsmil-84dev.o" Makefile; then
    echo -e "  ${GREEN}✓${NC} Final module marked as OBJECT_FILES_NON_STANDARD"
else
    echo -e "  ${RED}✗${NC} Final module NOT marked to skip objtool"
fi

if grep -q "OBJECT_FILES_NON_STANDARD_rust/libdsmil_rust.o" Makefile; then
    echo -e "  ${GREEN}✓${NC} Rust object marked as OBJECT_FILES_NON_STANDARD"
else
    echo -e "  ${YELLOW}⚠${NC} Rust object not explicitly marked"
fi
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════════"
echo -e "${BOLD}Summary & Recommendations${NC}"
echo ""

if command -v rustup &> /dev/null && rustup component list 2>/dev/null | grep -q "rust-src (installed)"; then
    RUST_SRC_OK=true
else
    RUST_SRC_OK=false
fi

if [[ -f "$CARGO_CONFIG" ]] && grep -q "build-std" "$CARGO_CONFIG" && grep -q "\-fma" "$CARGO_CONFIG"; then
    CARGO_CONFIG_OK=true
else
    CARGO_CONFIG_OK=false
fi

if command -v objdump &> /dev/null && [[ -f "$RUST_OBJ" ]]; then
    FMA_INSTRS=$(objdump -d "$RUST_OBJ" 2>/dev/null | grep -E "vfmadd|vfmsub|vfnmadd|vfnmsub|fma" || true)
    if [[ -z "$FMA_INSTRS" ]]; then
        NO_FMA_INSTRS=true
    else
        NO_FMA_INSTRS=false
    fi
else
    NO_FMA_INSTRS="unknown"
fi

if [[ "$RUST_SRC_OK" == true ]] && [[ "$CARGO_CONFIG_OK" == true ]] && [[ "$NO_FMA_INSTRS" == true ]]; then
    echo -e "${GREEN}${BOLD}✓ FMA mitigation is properly configured and working${NC}"
    echo ""
    echo "Your build should succeed without objtool failures."
elif [[ "$NO_FMA_INSTRS" == true ]]; then
    echo -e "${GREEN}${BOLD}✓ No FMA instructions detected (mitigation working)${NC}"
    echo ""
    echo "Build should succeed, but configuration could be improved."
elif [[ "$NO_FMA_INSTRS" == "unknown" ]]; then
    echo -e "${YELLOW}${BOLD}⚠ Cannot verify FMA instructions (object file not built yet)${NC}"
    echo ""
    echo "Run 'make all' and then re-run this script to verify."
else
    echo -e "${RED}${BOLD}✗ FMA mitigation is NOT working properly${NC}"
    echo ""
    echo "Recommended actions:"
    echo ""
fi

if [[ "$RUST_SRC_OK" == false ]]; then
    echo "1. Install rust-src component:"
    echo "   ${CYAN}rustup component add rust-src${NC}"
    echo ""
fi

if [[ "$CARGO_CONFIG_OK" == false ]]; then
    echo "2. Ensure Cargo config is present:"
    echo "   ${CYAN}cat $CARGO_CONFIG${NC}"
    echo ""
fi

if [[ "$NO_FMA_INSTRS" == false ]]; then
    echo "3. Rebuild with build-std:"
    echo "   ${CYAN}cd rust && make clean && make all${NC}"
    echo ""
fi

echo "4. If build still fails, the Makefile will fall back to non-Rust mode"
echo "   automatically."
echo ""

echo "For more information, see: MODULE_RELOAD_GUIDE.md"
echo ""
