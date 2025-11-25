#!/bin/bash
#
# Test All AI Accelerators
#
# Comprehensive test suite for Dell Latitude 5450 MIL-SPEC AI hardware:
# 1. Intel NPU (Meteor Lake, 49.4 TOPS INT8)
# 2. Intel GNA (Gaussian & Neural-Network Accelerator)
# 3. Military NPU (if present, requires DSMIL driver)
#
# Usage:
#   ./test_all_accelerators.sh
#   ./test_all_accelerators.sh --verbose
#   ./test_all_accelerators.sh --npu-only
#   ./test_all_accelerators.sh --gna-only
#   ./test_all_accelerators.sh --military-only

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
RUN_MILITARY=1
VERBOSE=0

for arg in "$@"; do
    case $arg in
        --npu-only)
            RUN_GNA=0
            RUN_MILITARY=0
            ;;
        --gna-only)
            RUN_NPU=0
            RUN_MILITARY=0
            ;;
        --military-only)
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
            echo "  --military-only  Test only Military NPU"
            echo "  --verbose        Show detailed output"
            echo "  --help           Show this help"
            exit 0
            ;;
    esac
done

# Header
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}  AI Accelerator Test Suite${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo -e "Hardware: Dell Latitude 5450 MIL-SPEC"
echo -e "Date: $(date)"
echo -e "Logs: $LOG_DIR"
echo -e "${BLUE}================================================================================${NC}"
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

# Test 3: Military NPU (DSMIL)
if [ $RUN_MILITARY -eq 1 ]; then
    echo -e "${YELLOW}[3/3] Testing Military NPU (DSMIL Driver)${NC}"
    echo "-----------------------------------------------------"

    MILITARY_LOG="$LOG_DIR/military_npu_${TIMESTAMP}.log"

    if [ $VERBOSE -eq 1 ]; then
        python3 "$SCRIPT_DIR/military_npu_dsmil_loader.py" 2>&1 | tee "$MILITARY_LOG"
        MILITARY_STATUS=${PIPESTATUS[0]}
    else
        python3 "$SCRIPT_DIR/military_npu_dsmil_loader.py" > "$MILITARY_LOG" 2>&1
        MILITARY_STATUS=$?
    fi

    if [ $MILITARY_STATUS -eq 0 ]; then
        echo -e "${GREEN}✓ Military NPU: ACTIVATED${NC}"
        RESULTS+=("MILITARY:PASS")
    else
        echo -e "${YELLOW}⚠️  Military NPU: NOT DETECTED (or driver not loaded)${NC}"
        RESULTS+=("MILITARY:N/A")
        echo "  See log: $MILITARY_LOG"
    fi
    echo ""
fi

# Summary
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}================================================================================${NC}"

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
    else
        echo -e "${COMPONENT}: ${YELLOW}⚠️  N/A${NC}"
    fi
done

echo -e "${BLUE}-------------------------------------------------------------------------------${NC}"
echo -e "Tests Passed: ${PASSED_TESTS}/${TOTAL_TESTS}"
echo -e "${BLUE}================================================================================${NC}"

# Hardware recommendation
echo ""
echo -e "${BLUE}Hardware Status:${NC}"

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

if [ $RUN_MILITARY -eq 1 ]; then
    if [ $MILITARY_STATUS -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} Military NPU - ACTIVATED (enhanced capabilities available)"
    else
        echo -e "  ${BLUE}ℹ${NC}  Military NPU - Not detected (consumer NPU available)"
    fi
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Review logs in: $LOG_DIR"
echo "  2. Test dynamic resource allocator:"
echo "     python3 02-ai-engine/hardware/dynamic_resource_allocator.py"
echo "  3. Build AVX-512 module:"
echo "     cd 02-ai-engine/rag_cpp && make build"

# Exit with overall status
if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests PASSED!${NC}"
    exit 0
else
    echo ""
    echo -e "${YELLOW}⚠️  Some tests did not pass. Check logs for details.${NC}"
    exit 1
fi
