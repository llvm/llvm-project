#!/bin/bash
# Hardware Validation Script for Claude-Backups Integration
# Checks availability of NPU, GPU, AVX2, and other hardware features

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DSMIL AI - Hardware Validation for Advanced Integration   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

TOTAL_SCORE=0
MAX_SCORE=0

# ============================================================================
# 1. CHECK INTEL NPU
# ============================================================================

echo -e "${BLUE}[1/10]${NC} Checking Intel NPU (Neural Processing Unit)..."
MAX_SCORE=$((MAX_SCORE + 20))

NPU_FOUND=false
NPU_DEVICE=""
NPU_TOPS=0

# Check PCI devices for NPU
if lspci | grep -qi "0b.0.*system peripheral.*intel"; then
    NPU_DEVICE=$(lspci | grep "0b.0")
    NPU_FOUND=true
    echo -e "${GREEN}✓${NC} Intel NPU detected: $NPU_DEVICE"

    # Try to determine TOPS from device ID or model
    if echo "$NPU_DEVICE" | grep -qi "3720"; then
        NPU_TOPS=34
        echo -e "${GREEN}  NPU 3720 detected: 34.0 TOPS (standard), 49.4 TOPS (military mode target)${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 20))
    else
        NPU_TOPS=11
        echo -e "${YELLOW}  NPU detected but model unclear: estimated 11 TOPS${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 15))
    fi
else
    echo -e "${RED}✗${NC} Intel NPU not detected"
    echo -e "${YELLOW}  NPU is critical for claude-backups voice UI and real-time inference${NC}"
fi

echo ""

# ============================================================================
# 2. CHECK INTEL GPU
# ============================================================================

echo -e "${BLUE}[2/10]${NC} Checking Intel GPU..."
MAX_SCORE=$((MAX_SCORE + 15))

GPU_FOUND=false
GPU_TOPS=0

if lspci | grep -i vga | grep -qi intel; then
    GPU_DEVICE=$(lspci | grep -i vga | grep -i intel)
    GPU_FOUND=true
    echo -e "${GREEN}✓${NC} Intel GPU detected: $GPU_DEVICE"

    # Check for Intel Arc
    if echo "$GPU_DEVICE" | grep -qi "arc\|7d55"; then
        GPU_TOPS=76
        echo -e "${GREEN}  Intel Arc detected: ~76-106 TOPS${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 15))
    else
        GPU_TOPS=10
        echo -e "${YELLOW}  Intel integrated GPU: ~10-20 TOPS estimated${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 10))
    fi
else
    echo -e "${RED}✗${NC} Intel GPU not detected"
fi

echo ""

# ============================================================================
# 3. CHECK AVX2/AVX512 SUPPORT
# ============================================================================

echo -e "${BLUE}[3/10]${NC} Checking CPU SIMD Extensions (for shadowgit)..."
MAX_SCORE=$((MAX_SCORE + 10))

AVX2_SUPPORTED=false
AVX512_SUPPORTED=false

if grep -q avx2 /proc/cpuinfo; then
    AVX2_SUPPORTED=true
    echo -e "${GREEN}✓${NC} AVX2 supported (3-5x git acceleration possible)"
    TOTAL_SCORE=$((TOTAL_SCORE + 5))
fi

if grep -q avx512 /proc/cpuinfo; then
    AVX512_SUPPORTED=true
    echo -e "${GREEN}✓${NC} AVX512 supported (5-10x git acceleration possible)"
    TOTAL_SCORE=$((TOTAL_SCORE + 10))
elif $AVX2_SUPPORTED; then
    echo -e "${YELLOW}⚠${NC} AVX512 not supported, but AVX2 available"
else
    echo -e "${RED}✗${NC} No AVX2 or AVX512 support (limited shadowgit benefits)"
fi

echo ""

# ============================================================================
# 4. CHECK CPU CORES
# ============================================================================

echo -e "${BLUE}[4/10]${NC} Checking CPU cores (for parallel agent execution)..."
MAX_SCORE=$((MAX_SCORE + 10))

CPU_CORES=$(nproc)
echo -e "${GREEN}✓${NC} $CPU_CORES CPU cores detected"

if [ "$CPU_CORES" -ge 16 ]; then
    echo -e "${GREEN}  Excellent for 98-agent parallel execution (20 cores recommended)${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 10))
elif [ "$CPU_CORES" -ge 8 ]; then
    echo -e "${YELLOW}  Good for agent execution (16+ cores recommended)${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 7))
else
    echo -e "${YELLOW}  Limited for 98-agent system (consider agent count reduction)${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 5))
fi

echo ""

# ============================================================================
# 5. CHECK MEMORY
# ============================================================================

echo -e "${BLUE}[5/10]${NC} Checking system memory..."
MAX_SCORE=$((MAX_SCORE + 10))

TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo -e "${GREEN}✓${NC} ${TOTAL_MEM_GB}GB total memory"

if [ "$TOTAL_MEM_GB" -ge 32 ]; then
    echo -e "${GREEN}  Excellent for large models and 98-agent system${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 10))
elif [ "$TOTAL_MEM_GB" -ge 16 ]; then
    echo -e "${YELLOW}  Adequate for most operations${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 7))
else
    echo -e "${YELLOW}  Limited (may need to reduce concurrent agents)${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 5))
fi

echo ""

# ============================================================================
# 6. CHECK AUDIO DEVICES (for voice UI)
# ============================================================================

echo -e "${BLUE}[6/10]${NC} Checking audio devices (for voice UI)..."
MAX_SCORE=$((MAX_SCORE + 5))

if command -v arecord &> /dev/null; then
    AUDIO_DEVICES=$(arecord -l 2>/dev/null | grep -c "card" || echo "0")
    if [ "$AUDIO_DEVICES" -gt 0 ]; then
        echo -e "${GREEN}✓${NC} $AUDIO_DEVICES audio input device(s) found"
        echo -e "${GREEN}  Voice UI integration possible${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 5))
    else
        echo -e "${YELLOW}⚠${NC} No audio input devices detected"
        echo -e "${YELLOW}  Voice UI will not work without microphone${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} Audio tools not installed (install alsa-utils)"
fi

echo ""

# ============================================================================
# 7. CHECK OPENVINO INSTALLATION
# ============================================================================

echo -e "${BLUE}[7/10]${NC} Checking OpenVINO installation..."
MAX_SCORE=$((MAX_SCORE + 10))

if python3 -c "import openvino" 2>/dev/null; then
    OPENVINO_VERSION=$(python3 -c "import openvino; print(openvino.__version__)" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓${NC} OpenVINO installed: version $OPENVINO_VERSION"
    TOTAL_SCORE=$((TOTAL_SCORE + 10))
else
    echo -e "${YELLOW}⚠${NC} OpenVINO not installed"
    echo -e "${YELLOW}  Required for NPU backend integration${NC}"
    echo -e "${YELLOW}  Install: pip3 install openvino${NC}"
fi

echo ""

# ============================================================================
# 8. CHECK REDIS (for agent messaging)
# ============================================================================

echo -e "${BLUE}[8/10]${NC} Checking Redis (for agent communication)..."
MAX_SCORE=$((MAX_SCORE + 10))

if command -v redis-cli &> /dev/null; then
    if redis-cli ping 2>/dev/null | grep -q PONG; then
        echo -e "${GREEN}✓${NC} Redis server running and responsive"
        TOTAL_SCORE=$((TOTAL_SCORE + 10))
    else
        echo -e "${YELLOW}⚠${NC} Redis installed but not running"
        echo -e "${YELLOW}  Start: sudo systemctl start redis-server${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 5))
    fi
else
    echo -e "${YELLOW}⚠${NC} Redis not installed"
    echo -e "${YELLOW}  Required for 98-agent message queue${NC}"
    echo -e "${YELLOW}  Install: sudo apt-get install redis-server${NC}"
fi

echo ""

# ============================================================================
# 9. CHECK RUST TOOLCHAIN (for shadowgit)
# ============================================================================

echo -e "${BLUE}[9/10]${NC} Checking Rust toolchain (for shadowgit compilation)..."
MAX_SCORE=$((MAX_SCORE + 5))

if command -v cargo &> /dev/null; then
    RUST_VERSION=$(cargo --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Rust installed: $RUST_VERSION"
    echo -e "${GREEN}  Can compile shadowgit from source if needed${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 5))
else
    echo -e "${YELLOW}⚠${NC} Rust not installed"
    echo -e "${YELLOW}  Optional for shadowgit compilation${NC}"
    echo -e "${YELLOW}  Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh${NC}"
fi

echo ""

# ============================================================================
# 10. CHECK DSMIL FRAMEWORK
# ============================================================================

echo -e "${BLUE}[10/10]${NC} Checking DSMIL framework status..."
MAX_SCORE=$((MAX_SCORE + 5))

if [ -f "/home/user/LAT5150DRVMIL/DSMIL_UNIVERSAL_FRAMEWORK.py" ]; then
    echo -e "${GREEN}✓${NC} DSMIL framework found"
    TOTAL_SCORE=$((TOTAL_SCORE + 5))

    # Try to get device count
    DSMIL_STATUS=$(python3 /home/user/LAT5150DRVMIL/DSMIL_UNIVERSAL_FRAMEWORK.py --status 2>/dev/null || echo "")
    if echo "$DSMIL_STATUS" | grep -q "devices"; then
        DEVICE_COUNT=$(echo "$DSMIL_STATUS" | grep -oP '\d+/\d+' | head -1)
        echo -e "${GREEN}  DSMIL status: $DEVICE_COUNT devices accessible${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} DSMIL framework not found at expected location"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    HARDWARE CAPABILITY SUMMARY                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Calculate total TOPS
TOTAL_TOPS=$(($NPU_TOPS + $GPU_TOPS))
echo -e "${GREEN}Total Compute Capacity:${NC}"
if $NPU_FOUND; then
    echo -e "  NPU:   ${NPU_TOPS} TOPS"
fi
if $GPU_FOUND; then
    echo -e "  GPU:   ${GPU_TOPS} TOPS"
fi
echo -e "  ${BLUE}Total: ${TOTAL_TOPS} TOPS${NC}"
echo ""

# Overall score
PERCENTAGE=$((TOTAL_SCORE * 100 / MAX_SCORE))

echo -e "${GREEN}Overall Readiness Score: $TOTAL_SCORE/$MAX_SCORE ($PERCENTAGE%)${NC}"
echo ""

if [ $PERCENTAGE -ge 80 ]; then
    echo -e "${GREEN}✓ EXCELLENT${NC} - System ready for full claude-backups integration"
    echo -e "  All advanced features can be implemented"
    READINESS="EXCELLENT"
elif [ $PERCENTAGE -ge 60 ]; then
    echo -e "${YELLOW}⚠ GOOD${NC} - Most features can be implemented"
    echo -e "  Some limitations with missing components"
    READINESS="GOOD"
elif [ $PERCENTAGE -ge 40 ]; then
    echo -e "${YELLOW}⚠ FAIR${NC} - Core features possible, advanced features limited"
    echo -e "  Consider installing missing components"
    READINESS="FAIR"
else
    echo -e "${RED}✗ LIMITED${NC} - Many features will not be available"
    echo -e "  Significant hardware/software upgrades needed"
    READINESS="LIMITED"
fi

echo ""

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      RECOMMENDATIONS                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Priority 1 (Critical):${NC}"
if ! $NPU_FOUND; then
    echo -e "  ❌ Install Intel NPU drivers"
    echo -e "     Impact: No NPU acceleration, voice UI limited"
fi

if ! python3 -c "import openvino" 2>/dev/null; then
    echo -e "  ❌ Install OpenVINO: pip3 install openvino"
    echo -e "     Impact: Cannot use NPU/GPU backends"
fi

if ! command -v redis-cli &> /dev/null || ! redis-cli ping 2>/dev/null | grep -q PONG; then
    echo -e "  ❌ Install/start Redis: sudo apt-get install redis-server"
    echo -e "     Impact: 98-agent communication will not work"
fi

echo ""

echo -e "${YELLOW}Priority 2 (Recommended):${NC}"
if ! $AVX512_SUPPORTED && ! $AVX2_SUPPORTED; then
    echo -e "  ⚠ CPU lacks AVX2/AVX512 support"
    echo -e "     Impact: Limited shadowgit acceleration benefits"
fi

if ! command -v cargo &> /dev/null; then
    echo -e "  ⚠ Install Rust for shadowgit compilation (optional)"
    echo -e "     Impact: Cannot compile shadowgit from source"
fi

if ! command -v arecord &> /dev/null || [ "$AUDIO_DEVICES" -eq 0 ]; then
    echo -e "  ⚠ No audio devices for voice UI"
    echo -e "     Impact: Voice interface will not work"
fi

echo ""

echo -e "${BLUE}Priority 3 (Nice to Have):${NC}"
if [ "$CPU_CORES" -lt 16 ]; then
    echo -e "  ℹ Consider reducing agent count if <16 cores"
    echo -e "     Impact: 98-agent system may be slow"
fi

if [ "$TOTAL_MEM_GB" -lt 32 ]; then
    echo -e "  ℹ Consider memory upgrade for large models"
    echo -e "     Impact: May need to use smaller models"
fi

echo ""

# ============================================================================
# NEXT STEPS
# ============================================================================

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        NEXT STEPS                             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$READINESS" = "EXCELLENT" ] || [ "$READINESS" = "GOOD" ]; then
    echo -e "${GREEN}Recommended Integration Path:${NC}"
    echo ""
    echo "1. Install missing dependencies (see recommendations above)"
    echo "2. Review CLAUDE_BACKUPS_INTEGRATION_ANALYSIS.md"
    echo "3. Begin Phase 1: NPU Activation (8-12 hours)"
    echo "   - Install OpenVINO"
    echo "   - Test NPU inference"
    echo "   - Add NPU to GUI dashboard"
    echo ""
    echo "4. Phase 2: 98-Agent Expansion (20-30 hours)"
    echo "   - Design agent architecture"
    echo "   - Implement communication protocol"
    echo "   - Test coordination"
    echo ""
    echo "5. Phase 3-6: Advanced features"
    echo "   - Shadowgit integration"
    echo "   - Voice UI"
    echo "   - Hook system"
    echo "   - Heterogeneous execution"
else
    echo -e "${YELLOW}Recommended Actions:${NC}"
    echo ""
    echo "1. Address all Priority 1 (Critical) items first"
    echo "2. Re-run this validation script"
    echo "3. Once score >60%, begin integration process"
fi

echo ""

# Save report
REPORT_FILE="/home/user/LAT5150DRVMIL/02-ai-engine/hardware_validation_report.json"

cat > "$REPORT_FILE" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "score": {
    "total": $TOTAL_SCORE,
    "max": $MAX_SCORE,
    "percentage": $PERCENTAGE,
    "readiness": "$READINESS"
  },
  "hardware": {
    "npu": {
      "found": $NPU_FOUND,
      "tops": $NPU_TOPS,
      "device": "$NPU_DEVICE"
    },
    "gpu": {
      "found": $GPU_FOUND,
      "tops": $GPU_TOPS
    },
    "cpu": {
      "cores": $CPU_CORES,
      "avx2": $AVX2_SUPPORTED,
      "avx512": $AVX512_SUPPORTED
    },
    "memory_gb": $TOTAL_MEM_GB,
    "total_tops": $TOTAL_TOPS
  },
  "software": {
    "openvino_installed": $(python3 -c "import openvino; print('true')" 2>/dev/null || echo "false"),
    "redis_running": $(redis-cli ping 2>/dev/null | grep -q PONG && echo "true" || echo "false"),
    "rust_installed": $(command -v cargo &> /dev/null && echo "true" || echo "false")
  }
}
EOF

echo -e "${GREEN}✓ Validation report saved to:${NC} $REPORT_FILE"
echo ""

exit 0
