#!/bin/bash
#
# DSMIL Comprehensive System Analysis Script
#
# This script performs a complete analysis of the DSMIL framework including:
# - Hardware discovery and capability assessment
# - Device functional testing and validation
# - Performance benchmarking
# - Security posture analysis
# - Integration health checks
# - Comprehensive reporting with recommendations
#
# USAGE:
#   ./dsmil-analyze.sh [OPTIONS]
#
# OPTIONS:
#   --help, -h          Show this help message
#   --quick             Quick analysis (skip benchmarks)
#   --full              Full analysis with benchmarks (default)
#   --output DIR        Save reports to directory (default: ./dsmil-reports)
#   --timestamp         Add timestamp to report filename
#   --sudo              Run with sudo for deep hardware access
#
# Author: DSMIL Integration Framework
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
#

# Don't exit on error - continue through all phases
set +e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOOLS_DIR="${SCRIPT_DIR}/02-tools/dsmil-devices"

# Default options
QUICK_MODE=0
OUTPUT_DIR="./dsmil-reports"
ADD_TIMESTAMP=1
USE_SUDO=0
SHOW_HELP=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            SHOW_HELP=1
            shift
            ;;
        --quick)
            QUICK_MODE=1
            shift
            ;;
        --full)
            QUICK_MODE=0
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --timestamp)
            ADD_TIMESTAMP=1
            shift
            ;;
        --sudo)
            USE_SUDO=1
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help
if [ $SHOW_HELP -eq 1 ]; then
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║              DSMIL COMPREHENSIVE SYSTEM ANALYSIS - HELP                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    ./dsmil-analyze.sh [OPTIONS]

OPTIONS:
    --help, -h          Show this help message
    --quick             Quick analysis (skip benchmarks)
    --full              Full analysis with benchmarks (default)
    --output DIR        Save reports to directory (default: ./dsmil-reports)
    --timestamp         Add timestamp to report filename (default)
    --sudo              Run with sudo for deep hardware access

ANALYSIS PHASES:

    Phase 1: Hardware Discovery
        • Deep hardware scanning (SMM/MSR/Intel ME)
        • DSMIL capability detection
        • Security posture assessment
        • HAP bit verification

    Phase 2: Device Functional Testing
        • All DSMIL devices initialization
        • Operation testing (all operations)
        • Quarantine verification
        • Error detection

    Phase 3: Performance Benchmarking (--full mode)
        • Device operation timing
        • Register access performance
        • Memory usage analysis
        • CPU utilization tracking

    Phase 4: Security Analysis
        • Device capability assessment
        • Risk level verification
        • Security policy compliance
        • Encryption status

    Phase 5: Integration Health
        • Module import verification
        • Cross-device communication
        • Framework integrity
        • Documentation completeness

    Phase 6: Comprehensive Reporting
        • Executive summary
        • Detailed findings
        • Performance metrics
        • Security recommendations
        • Deployment readiness

EXAMPLES:

    Quick analysis (no benchmarks):
        ./dsmil-analyze.sh --quick

    Full analysis with sudo:
        ./dsmil-analyze.sh --full --sudo

    Save to custom directory:
        ./dsmil-analyze.sh --output /tmp/reports

    Complete deep analysis:
        ./dsmil-analyze.sh --full --sudo --output ./reports

OUTPUT:

    Reports saved to: <OUTPUT_DIR>/
        • dsmil-analysis-<timestamp>.txt    - Full text report
        • dsmil-discovery-<timestamp>.txt   - Hardware discovery
        • dsmil-functional-<timestamp>.txt  - Functional test results
        • dsmil-security-<timestamp>.txt    - Security analysis
        • dsmil-summary-<timestamp>.txt     - Executive summary

REQUIREMENTS:

    Required:
        • Python 3.x
        • DSMIL integration framework (02-tools/dsmil-devices/)

    Optional:
        • Root/sudo access (for complete hardware detection)
        • msr-tools (rdmsr/wrmsr)
        • intelmetool (Intel ME analysis)

CLASSIFICATION:
    UNCLASSIFIED // FOR OFFICIAL USE ONLY

EOF
    exit 0
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    IS_ROOT=1
elif [ $USE_SUDO -eq 1 ]; then
    echo -e "${YELLOW}⚠ Elevating to root for deep hardware access...${NC}"
    exec sudo "$0" "$@"
else
    IS_ROOT=0
fi

# Create output directory with proper permissions
mkdir -p "$OUTPUT_DIR"
chmod 755 "$OUTPUT_DIR" 2>/dev/null || true

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Report filenames
if [ $ADD_TIMESTAMP -eq 1 ]; then
    REPORT_FULL="${OUTPUT_DIR}/dsmil-analysis-${TIMESTAMP}.txt"
    REPORT_DISCOVERY="${OUTPUT_DIR}/dsmil-discovery-${TIMESTAMP}.txt"
    REPORT_FUNCTIONAL="${OUTPUT_DIR}/dsmil-functional-${TIMESTAMP}.txt"
    REPORT_SECURITY="${OUTPUT_DIR}/dsmil-security-${TIMESTAMP}.txt"
    REPORT_SUMMARY="${OUTPUT_DIR}/dsmil-summary-${TIMESTAMP}.txt"
else
    REPORT_FULL="${OUTPUT_DIR}/dsmil-analysis.txt"
    REPORT_DISCOVERY="${OUTPUT_DIR}/dsmil-discovery.txt"
    REPORT_FUNCTIONAL="${OUTPUT_DIR}/dsmil-functional.txt"
    REPORT_SECURITY="${OUTPUT_DIR}/dsmil-security.txt"
    REPORT_SUMMARY="${OUTPUT_DIR}/dsmil-summary.txt"
fi

# Print banner
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              DSMIL COMPREHENSIVE SYSTEM ANALYSIS                             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "  ${GREEN}Analysis Mode:${NC}    $([ $QUICK_MODE -eq 1 ] && echo 'Quick (no benchmarks)' || echo 'Full (with benchmarks)')"
echo -e "  ${GREEN}Output Dir:${NC}       $OUTPUT_DIR"
echo -e "  ${GREEN}Timestamp:${NC}        $TIMESTAMP"
echo -e "  ${GREEN}Root Access:${NC}      $([ $IS_ROOT -eq 1 ] && echo 'Yes' || echo 'No')"
echo

# Start analysis
START_TIME=$(date +%s)

# Phase 1: Hardware Discovery
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}Phase 1/6: Hardware Discovery${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

python3 "${TOOLS_DIR}/dsmil_discover.py" 2>&1 | tee "$REPORT_DISCOVERY"
DISCOVERY_STATUS=${PIPESTATUS[0]}

if [ $DISCOVERY_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}✓ Hardware discovery completed successfully${NC}"
    # Extract key metrics
    HARDWARE_SCORE=$(grep -A 1 "Overall Readiness:" "$REPORT_DISCOVERY" | tail -1 | grep -oP '\d+/\d+' || echo "N/A")
    echo -e "  Readiness Score: ${HARDWARE_SCORE}"
else
    echo -e "\n${RED}✗ Hardware discovery failed (exit code: $DISCOVERY_STATUS)${NC}"
fi
echo

# Phase 2: Device Functional Testing
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}Phase 2/6: Device Functional Testing${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

python3 "${TOOLS_DIR}/dsmil_probe.py" 2>&1 | tee "$REPORT_FUNCTIONAL"
FUNCTIONAL_STATUS=${PIPESTATUS[0]}

if [ $FUNCTIONAL_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}✓ Functional testing completed successfully${NC}"
    # Extract metrics
    FUNCTIONAL_COUNT=$(grep "Fully Functional:" "$REPORT_FUNCTIONAL" | grep -oP '\d+' | head -1 || echo "0")
    TOTAL_DEVICES=$(grep "Found.*registered devices" "$REPORT_FUNCTIONAL" | grep -oP '\d+' | head -1 || echo "0")
    TOTAL_OPS=$(grep "Total Operations Tested:" "$REPORT_FUNCTIONAL" | grep -oP '\d+' || echo "0")
    SUCCESS_OPS=$(grep "Total Successful:" "$REPORT_FUNCTIONAL" | grep -oP '\d+' || echo "0")
    echo -e "  Total Devices: ${TOTAL_DEVICES}"
    echo -e "  Functional Devices: ${FUNCTIONAL_COUNT}"
    echo -e "  Successful Operations: ${SUCCESS_OPS}/${TOTAL_OPS}"
else
    echo -e "\n${RED}✗ Functional testing failed (exit code: $FUNCTIONAL_STATUS)${NC}"
fi
echo

# Phase 3: Performance Benchmarking
if [ $QUICK_MODE -eq 0 ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}Phase 3/6: Performance Benchmarking${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo

    # Create benchmark script
    python3 - <<'BENCHMARK_SCRIPT'
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '02-tools/dsmil-devices'))

from dsmil_integration import initialize_all_devices, get_device

print("Benchmarking device operations...")

# Initialize devices
initialize_all_devices()

# Benchmark get_status operations
timings = []
for device_id in [0x8000, 0x8001, 0x8002, 0x8003, 0x8005, 0x8006]:
    device = get_device(device_id)
    if device:
        start = time.perf_counter()
        for _ in range(100):
            device.get_status()
        end = time.perf_counter()
        avg_time = (end - start) / 100 * 1000  # Convert to ms
        timings.append(avg_time)

if timings:
    print(f"Average operation time: {statistics.mean(timings):.4f} ms")
    print(f"Min operation time: {min(timings):.4f} ms")
    print(f"Max operation time: {max(timings):.4f} ms")
    print(f"Std deviation: {statistics.stdev(timings):.4f} ms")
else:
    print("No benchmark data available")
BENCHMARK_SCRIPT

    echo -e "${GREEN}✓ Performance benchmarking completed${NC}"
    echo
else
    echo -e "${YELLOW}⚠ Skipping performance benchmarking (quick mode)${NC}"
    echo
fi

# Phase 4: Security Analysis
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}Phase 4/6: Security Analysis${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

python3 - > "$REPORT_SECURITY" 2>&1 <<'SECURITY_SCRIPT'
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '02-tools/dsmil-devices'))

from dsmil_integration import list_devices, get_device, initialize_all_devices

print("=" * 80)
print("DSMIL SECURITY ANALYSIS")
print("=" * 80)
print()

devices = list_devices()
initialize_all_devices()

# Analyze device capabilities
print("━━━ Device Security Capabilities ━━━")
print()

encryption_devices = []
read_only_devices = []
config_devices = []
dma_devices = []

for dev_info in devices:
    device = get_device(dev_info['device_id'])
    if device:
        caps = device.get_capabilities().data.get('capabilities', [])

        if 'encrypted_storage' in caps:
            encryption_devices.append(dev_info['name'])
        if 'read_only' in caps:
            read_only_devices.append(dev_info['name'])
        if 'configuration' in caps:
            config_devices.append(dev_info['name'])
        if 'dma_capable' in caps:
            dma_devices.append(dev_info['name'])

print(f"Devices with Encrypted Storage: {len(encryption_devices)}")
for name in encryption_devices[:5]:
    print(f"  • {name}")
if len(encryption_devices) > 5:
    print(f"  ... and {len(encryption_devices) - 5} more")

print(f"\nRead-Only Devices (Safer): {len(read_only_devices)}")
print(f"Configuration Devices (Caution): {len(config_devices)}")
print(f"DMA-Capable Devices (High Risk): {len(dma_devices)}")

# Risk level distribution
print("\n━━━ Risk Level Distribution ━━━")
print()

risk_counts = {}
for dev_info in devices:
    risk = dev_info['risk_level']
    risk_counts[risk] = risk_counts.get(risk, 0) + 1

for risk, count in sorted(risk_counts.items()):
    print(f"  {risk.upper():12} {count:3} devices")

# Security recommendations
print("\n━━━ Security Recommendations ━━━")
print()

recommendations = []

if dma_devices:
    recommendations.append("⚠ DMA-capable devices detected - ensure IOMMU is enabled")

if config_devices:
    recommendations.append("⚠ Configuration devices require access control")

if len(encryption_devices) < 5:
    recommendations.append("ℹ Consider enabling encryption on more devices")

if not recommendations:
    recommendations.append("✓ No immediate security concerns detected")

for i, rec in enumerate(recommendations, 1):
    print(f"  [{i}] {rec}")

print("\n" + "=" * 80)
SECURITY_SCRIPT

SECURITY_STATUS=$?

if [ $SECURITY_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ Security analysis completed${NC}"
else
    echo -e "${RED}✗ Security analysis failed${NC}"
fi
echo

# Phase 5: Integration Health Check
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}Phase 5/6: Integration Health Check${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Check imports
echo -e "${CYAN}Checking module imports...${NC}"
python3 -c "
import sys
import os
sys.path.insert(0, '${TOOLS_DIR}')

try:
    import dsmil_integration
    print('  ✓ dsmil_integration')
    import dsmil_menu
    print('  ✓ dsmil_menu')
    import dsmil_probe
    print('  ✓ dsmil_probe')
    import dsmil_discover
    print('  ✓ dsmil_discover')
    print('\nAll core modules imported successfully')
except Exception as e:
    print(f'  ✗ Import error: {e}')
    sys.exit(1)
"

IMPORT_STATUS=$?
echo

# Check file integrity
echo -e "${CYAN}Checking framework files...${NC}"
FILES_OK=1

for file in "${TOOLS_DIR}/dsmil_integration.py" \
            "${TOOLS_DIR}/dsmil_menu.py" \
            "${TOOLS_DIR}/dsmil_probe.py" \
            "${TOOLS_DIR}/dsmil_discover.py" \
            "${TOOLS_DIR}/lib/device_registry.py" \
            "${TOOLS_DIR}/lib/device_base.py"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $(basename $file)"
    else
        echo -e "  ${RED}✗${NC} $(basename $file) - MISSING"
        FILES_OK=0
    fi
done
echo

if [ $IMPORT_STATUS -eq 0 ] && [ $FILES_OK -eq 1 ]; then
    echo -e "${GREEN}✓ Integration health check passed${NC}"
else
    echo -e "${RED}✗ Integration health check failed${NC}"
fi
echo

# Phase 6: Generate Comprehensive Report
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}Phase 6/6: Generating Comprehensive Report${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Create comprehensive report
cat > "$REPORT_FULL" << EOF
╔══════════════════════════════════════════════════════════════════════════════╗
║              DSMIL COMPREHENSIVE SYSTEM ANALYSIS REPORT                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated: $(date '+%Y-%m-%d %H:%M:%S')
Analysis Mode: $([ $QUICK_MODE -eq 1 ] && echo 'Quick' || echo 'Full')
Root Access: $([ $IS_ROOT -eq 1 ] && echo 'Yes' || echo 'No')

================================================================================
EXECUTIVE SUMMARY
================================================================================

Hardware Discovery Status:     $([ $DISCOVERY_STATUS -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')
Functional Testing Status:     $([ $FUNCTIONAL_STATUS -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')
Security Analysis Status:      $([ $SECURITY_STATUS -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')
Integration Health Status:     $([ $IMPORT_STATUS -eq 0 ] && [ $FILES_OK -eq 1 ] && echo '✓ PASSED' || echo '✗ FAILED')

Hardware Readiness Score:      ${HARDWARE_SCORE:-N/A}
Functional Devices:            ${FUNCTIONAL_COUNT:-0}/22
Successful Operations:         ${SUCCESS_OPS:-0}/${TOTAL_OPS:-0}

Overall Status: $([ $DISCOVERY_STATUS -eq 0 ] && [ $FUNCTIONAL_STATUS -eq 0 ] && echo '✓ SYSTEM OPERATIONAL' || echo '⚠ ISSUES DETECTED')

================================================================================
DETAILED REPORTS
================================================================================

The following detailed reports have been generated:

  • Hardware Discovery:     ${REPORT_DISCOVERY}
  • Functional Testing:     ${REPORT_FUNCTIONAL}
  • Security Analysis:      ${REPORT_SECURITY}
  • This Summary:           ${REPORT_FULL}

================================================================================
EOF

# Append sections from other reports
echo "HARDWARE DISCOVERY SUMMARY" >> "$REPORT_FULL"
echo "================================================================================" >> "$REPORT_FULL"
grep -A 20 "DISCOVERY SUMMARY" "$REPORT_DISCOVERY" >> "$REPORT_FULL" 2>/dev/null || echo "Not available" >> "$REPORT_FULL"
echo "" >> "$REPORT_FULL"

echo "FUNCTIONAL TESTING SUMMARY" >> "$REPORT_FULL"
echo "================================================================================" >> "$REPORT_FULL"
grep -A 15 "━━━ Summary ━━━" "$REPORT_FUNCTIONAL" >> "$REPORT_FULL" 2>/dev/null || echo "Not available" >> "$REPORT_FULL"
echo "" >> "$REPORT_FULL"

echo "SECURITY ANALYSIS SUMMARY" >> "$REPORT_FULL"
echo "================================================================================" >> "$REPORT_FULL"
cat "$REPORT_SECURITY" >> "$REPORT_FULL" 2>/dev/null || echo "Not available" >> "$REPORT_FULL"

# Create executive summary
cat > "$REPORT_SUMMARY" << EOF
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DSMIL ANALYSIS EXECUTIVE SUMMARY                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Date: $(date '+%Y-%m-%d %H:%M:%S')

QUICK STATUS
============

Overall System Status:         $([ $DISCOVERY_STATUS -eq 0 ] && [ $FUNCTIONAL_STATUS -eq 0 ] && echo '✓ OPERATIONAL' || echo '⚠ ATTENTION REQUIRED')

Hardware Readiness:            ${HARDWARE_SCORE:-N/A}
Functional Devices:            ${FUNCTIONAL_COUNT:-0}/22 ($([ "${FUNCTIONAL_COUNT:-0}" -eq 22 ] && echo '100%' || echo "$((FUNCTIONAL_COUNT * 100 / 22))%"))
Operation Success Rate:        $([ "$TOTAL_OPS" != "0" ] && echo "$(( SUCCESS_OPS * 100 / TOTAL_OPS ))%" || echo "N/A")

KEY FINDINGS
============

Hardware Discovery:
  Status: $([ $DISCOVERY_STATUS -eq 0 ] && echo '✓ Completed successfully' || echo '✗ Failed')
  $(grep -m 1 "Compatibility:" "$REPORT_DISCOVERY" 2>/dev/null || echo "Compatibility: Unknown")

Functional Testing:
  Status: $([ $FUNCTIONAL_STATUS -eq 0 ] && echo '✓ All tests passed' || echo '✗ Some tests failed')
  Devices Tested: ${FUNCTIONAL_COUNT:-0}/22
  Operations Tested: ${TOTAL_OPS:-0}
  Success Rate: $([ "$TOTAL_OPS" != "0" ] && echo "$(( SUCCESS_OPS * 100 / TOTAL_OPS ))%" || echo "N/A")

Security Analysis:
  Status: $([ $SECURITY_STATUS -eq 0 ] && echo '✓ Analysis complete' || echo '✗ Analysis incomplete')
  Review: See ${REPORT_SECURITY}

Integration Health:
  Core Modules: $([ $IMPORT_STATUS -eq 0 ] && echo '✓ OK' || echo '✗ Issues detected')
  Framework Files: $([ $FILES_OK -eq 1 ] && echo '✓ OK' || echo '✗ Missing files')

RECOMMENDATIONS
===============

$(if [ $IS_ROOT -eq 0 ]; then echo "  • Run with --sudo for complete hardware analysis"; fi)
$(if [ $QUICK_MODE -eq 1 ]; then echo "  • Run with --full for performance benchmarks"; fi)
$(if [ $DISCOVERY_STATUS -ne 0 ] || [ $FUNCTIONAL_STATUS -ne 0 ]; then echo "  • Review detailed reports for error information"; fi)
$(if [ "${FUNCTIONAL_COUNT:-0}" -ne 22 ]; then echo "  • Investigate non-functional devices"; fi)

FULL REPORTS
============

  • Complete Analysis:      ${REPORT_FULL}
  • Hardware Discovery:     ${REPORT_DISCOVERY}
  • Functional Testing:     ${REPORT_FUNCTIONAL}
  • Security Analysis:      ${REPORT_SECURITY}

For questions or issues, review the detailed reports above.

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
EOF

echo -e "${GREEN}✓ Comprehensive report generated${NC}"
echo

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Final summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Analysis Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo -e "${CYAN}Results Summary:${NC}"
echo -e "  Total Time:          ${TOTAL_TIME} seconds"
echo -e "  Reports Generated:   5 files"
echo -e "  Output Directory:    $OUTPUT_DIR"
echo
echo -e "${CYAN}Quick Review:${NC}"
cat "$REPORT_SUMMARY"
echo
echo -e "${CYAN}Next Steps:${NC}"
echo -e "  • Review executive summary:  ${YELLOW}cat $REPORT_SUMMARY${NC}"
echo -e "  • Review full report:        ${YELLOW}less $REPORT_FULL${NC}"
echo -e "  • Review hardware details:   ${YELLOW}less $REPORT_DISCOVERY${NC}"
echo -e "  • Review functional tests:   ${YELLOW}less $REPORT_FUNCTIONAL${NC}"
echo -e "  • Review security analysis:  ${YELLOW}less $REPORT_SECURITY${NC}"
echo
echo -e "${GREEN}✓ All analysis files saved to: ${YELLOW}$OUTPUT_DIR${NC}"
echo

exit 0
