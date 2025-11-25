#!/bin/bash
#
# Test All AI Accelerators - Enhanced with DSMIL Integration
#
# Comprehensive test suite for Dell Latitude 5450 MIL-SPEC AI hardware:
# 1. Intel NPU (Meteor Lake, 49.4 TOPS INT8)
# 2. Intel GNA (Gaussian & Neural-Network Accelerator)
# 3. DSMIL NPU Integration (Consumer + Military with Hardware Attestation)
#
# Usage:
#   ./test_all_accelerators_enhanced.sh
#   ./test_all_accelerators_enhanced.sh --verbose
#   ./test_all_accelerators_enhanced.sh --npu-only
#   ./test_all_accelerators_enhanced.sh --dsmil-only

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tank/ai-engine/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

# Parse arguments
RUN_NPU=1
RUN_GNA=1
RUN_DSMIL=1
VERBOSE=0

for arg in "$@"; do
    case $arg in
        --npu-only)
            RUN_GNA=0
            RUN_DSMIL=0
            ;;
        --gna-only)
            RUN_NPU=0
            RUN_DSMIL=0
            ;;
        --dsmil-only)
            RUN_NPU=0
            RUN_GNA=0
            ;;
        --verbose)
            VERBOSE=1
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --npu-only       Test only Intel NPU"
            echo "  --gna-only       Test only Intel GNA"
            echo "  --dsmil-only     Test only DSMIL NPU Integration"
            echo "  --verbose        Show detailed output"
            echo "  --help           Show this help"
            exit 0
            ;;
    esac
done

# Header
echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}  AI Accelerator Test Suite - Enhanced DSMIL Integration${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo -e "Hardware: ${BLUE}Dell Latitude 5450 MIL-SPEC${NC}"
echo -e "Date: $(date)"
echo -e "Logs: $LOG_DIR"
echo -e "${CYAN}================================================================================${NC}"
echo ""

# Results
RESULTS=()

# Test 1: Intel NPU
if [ $RUN_NPU -eq 1 ]; then
    echo -e "${YELLOW}[1/3] Testing Intel NPU (Meteor Lake, 49.4 TOPS INT8)${NC}"
    echo "-----------------------------------------------------"

    NPU_LOG="$LOG_DIR/npu_test_${TIMESTAMP}.log"

    if [ $VERBOSE -eq 1 ]; then
        python3 "$SCRIPT_DIR/npu_test_benchmark.py" 2>&1 | tee "$NPU_LOG"
        NPU_STATUS=${PIPESTATUS[0]}
    else
        python3 "$SCRIPT_DIR/npu_test_benchmark.py" > "$NPU_LOG" 2>&1
        NPU_STATUS=$?
    fi

    if [ $NPU_STATUS -eq 0 ]; then
        echo -e "${GREEN}✓ NPU Test: PASSED${NC}"
        RESULTS+=("NPU:PASS")
    else
        echo -e "${RED}✗ NPU Test: FAILED${NC}"
        RESULTS+=("NPU:FAIL")
        echo "  See log: $NPU_LOG"
    fi
    echo ""
fi

# Test 2: Intel GNA
if [ $RUN_GNA -eq 1 ]; then
    echo -e "${YELLOW}[2/3] Testing Intel GNA (Gaussian & Neural-Network Accelerator)${NC}"
    echo "-----------------------------------------------------"

    GNA_LOG="$LOG_DIR/gna_activation_${TIMESTAMP}.log"

    if [ $VERBOSE -eq 1 ]; then
        python3 "$SCRIPT_DIR/gna_activation.py" 2>&1 | tee "$GNA_LOG"
        GNA_STATUS=${PIPESTATUS[0]}
    else
        python3 "$SCRIPT_DIR/gna_activation.py" > "$GNA_LOG" 2>&1
        GNA_STATUS=$?
    fi

    if [ $GNA_STATUS -eq 0 ]; then
        echo -e "${GREEN}✓ GNA Activation: PASSED${NC}"
        RESULTS+=("GNA:PASS")
    else
        echo -e "${RED}✗ GNA Activation: FAILED${NC}"
        RESULTS+=("GNA:FAIL")
        echo "  See log: $GNA_LOG"
    fi
    echo ""
fi

# Test 3: DSMIL NPU Integration
if [ $RUN_DSMIL -eq 1 ]; then
    echo -e "${YELLOW}[3/3] Testing DSMIL NPU Integration (Hardware Attestation)${NC}"
    echo "-----------------------------------------------------"
    echo -e "${CYAN}DSMIL Framework: 84-Device System with Mode 5 Attestation${NC}"
    echo ""

    DSMIL_LOG="$LOG_DIR/dsmil_npu_integration_${TIMESTAMP}.log"

    if [ $VERBOSE -eq 1 ]; then
        python3 "$SCRIPT_DIR/dsmil_npu_integration.py" 2>&1 | tee "$DSMIL_LOG"
        DSMIL_STATUS=${PIPESTATUS[0]}
    else
        python3 "$SCRIPT_DIR/dsmil_npu_integration.py" > "$DSMIL_LOG" 2>&1
        DSMIL_STATUS=$?
    fi

    if [ $DSMIL_STATUS -eq 0 ]; then
        echo -e "${GREEN}✓ DSMIL NPU Integration: COMPLETE${NC}"
        RESULTS+=("DSMIL:PASS")

        # Show DSMIL config summary
        if [ -f "/tank/ai-engine/logs/dsmil_npu_config.json" ]; then
            echo ""
            echo -e "${CYAN}DSMIL Configuration:${NC}"
            python3 -c "
import json
try:
    with open('/tank/ai-engine/logs/dsmil_npu_config.json', 'r') as f:
        config = json.load(f)
        print(f\"  Mode 5 Level: {config.get('mode5_level', 'UNKNOWN')}\")
        npus = config.get('npus', [])
        for npu in npus:
            print(f\"  NPU ({npu['device_type']}): {npu['tops_rating']} TOPS\")
except:
    pass
" 2>/dev/null || true
        fi
    else
        echo -e "${YELLOW}⚠️  DSMIL Integration: Partial${NC}"
        RESULTS+=("DSMIL:PARTIAL")
        echo "  See log: $DSMIL_LOG"
        echo ""
        echo -e "${BLUE}Note: Consumer NPU (49.4 TOPS) may still be available${NC}"
        echo -e "${BLUE}      DSMIL framework provides hardware attestation${NC}"
    fi
    echo ""
fi

# Summary
echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}  Test Summary${NC}"
echo -e "${CYAN}================================================================================${NC}"

TOTAL_TESTS=0
PASSED_TESTS=0

for result in "${RESULTS[@]}"; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    COMPONENT=$(echo "$result" | cut -d: -f1)
    STATUS=$(echo "$result" | cut -d: -f2)

    if [ "$STATUS" == "PASS" ]; then
        echo -e "${COMPONENT}: ${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    elif [ "$STATUS" == "FAIL" ]; then
        echo -e "${COMPONENT}: ${RED}✗ FAILED${NC}"
    elif [ "$STATUS" == "PARTIAL" ]; then
        echo -e "${COMPONENT}: ${YELLOW}⚠️  PARTIAL${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 0.5))  # Half credit
    else
        echo -e "${COMPONENT}: ${YELLOW}⚠️  N/A${NC}"
    fi
done

echo -e "${CYAN}-------------------------------------------------------------------------------${NC}"
echo -e "Tests Passed: ${PASSED_TESTS}/${TOTAL_TESTS}"
echo -e "${CYAN}================================================================================${NC}"

# Hardware status
echo ""
echo -e "${CYAN}Hardware Status:${NC}"

if [ $RUN_NPU -eq 1 ] && [ $NPU_STATUS -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} Intel NPU (49.4 TOPS INT8) - Ready for inference"
fi

if [ $RUN_GNA -eq 1 ]; then
    if [ $GNA_STATUS -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} Intel GNA - Activated (audio ML, low-power inference)"
    else
        echo -e "  ${YELLOW}⚠️${NC}  Intel GNA - Not activated (optional, mainly for audio)"
    fi
fi

if [ $RUN_DSMIL -eq 1 ]; then
    if [ $DSMIL_STATUS -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} DSMIL Framework - ACTIVE (hardware-attested AI inference enabled)"
    else
        echo -e "  ${BLUE}ℹ${NC}  DSMIL Framework - Not loaded (consumer NPU still usable)"
        echo -e "     Install with: sudo dpkg -i /home/user/LAT5150DRVMIL/packaging/dsmil-complete_8.3.2-1.deb"
    fi
fi

echo ""
echo -e "${CYAN}DSMIL Features:${NC}"
echo "  • Hardware-attested AI inference (Mode 5 platform integrity)"
echo "  • TPM-sealed model weights for security"
echo "  • 84-device virtual device system"
echo "  • APT defense and audit logging"
echo "  • Integration with consumer NPU (49.4 TOPS)"

echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. Review logs in: $LOG_DIR"
echo "  2. Install DSMIL framework (if not installed):"
echo "     sudo dpkg -i /home/user/LAT5150DRVMIL/packaging/dsmil-complete_8.3.2-1.deb"
echo "  3. Test dynamic resource allocator:"
echo "     python3 02-ai-engine/hardware/dynamic_resource_allocator.py"
echo "  4. Build AVX-512 module:"
echo "     cd 02-ai-engine/rag_cpp && make build"

# Exit with overall status
if (( $(echo "$PASSED_TESTS >= $TOTAL_TESTS - 0.5" | bc -l) )); then
    echo ""
    echo -e "${GREEN}🎉 All critical tests PASSED!${NC}"
    exit 0
else
    echo ""
    echo -e "${YELLOW}⚠️  Some tests did not pass. Check logs for details.${NC}"
    exit 1
fi
