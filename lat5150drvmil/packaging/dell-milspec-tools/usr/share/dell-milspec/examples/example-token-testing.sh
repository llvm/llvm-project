#!/bin/bash
#
# Example: Dell MIL-SPEC Token Testing
# Demonstrates safe SMBIOS token testing
#

set -e

echo "======================================================================"
echo " Dell MIL-SPEC Tools - Token Testing Example"
echo "======================================================================"
echo ""
echo "IMPORTANT: This example uses DRY RUN mode by default."
echo "           No actual SMBIOS tokens will be modified."
echo ""

# Check prerequisites
echo "Checking prerequisites..."
if ! lsmod | grep -q dsmil_72dev; then
    echo "ERROR: DSMIL module not loaded"
    echo "Load the module first: sudo modprobe dsmil_72dev"
    exit 1
fi

if [ ! -e /dev/dsmil0 ]; then
    echo "ERROR: DSMIL device not found"
    exit 1
fi

echo "Prerequisites OK"
echo ""

# Show available token ranges
echo "Available DSMIL Token Ranges:"
echo "  Range_0400: 0x0400-0x0447 (72 tokens) - Low priority"
echo "  Range_0480: 0x0480-0x04C7 (72 tokens) - HIGH PRIORITY [RECOMMENDED]"
echo "  Range_0500: 0x0500-0x0547 (72 tokens) - Medium priority"
echo "  Range_1000: 0x1000-0x1047 (72 tokens) - Low priority"
echo "  ... and 7 more ranges"
echo ""

# Menu
while true; do
    echo "Token Testing Options:"
    echo "  1) Test Range_0480 (Dry Run) - Recommended first test"
    echo "  2) Test Range_0400 (Dry Run)"
    echo "  3) Test Range_0500 (Dry Run)"
    echo "  4) Custom Range (Dry Run)"
    echo "  5) View Test Logs"
    echo "  0) Exit"
    echo ""
    read -p "Select option [0-5]: " choice

    case $choice in
        1)
            echo ""
            echo "Testing Range_0480 in DRY RUN mode..."
            echo ""
            dsmil-test --dry-run --range Range_0480
            echo ""
            read -p "Press Enter to continue..."
            ;;
        2)
            echo ""
            echo "Testing Range_0400 in DRY RUN mode..."
            echo ""
            dsmil-test --dry-run --range Range_0400
            echo ""
            read -p "Press Enter to continue..."
            ;;
        3)
            echo ""
            echo "Testing Range_0500 in DRY RUN mode..."
            echo ""
            dsmil-test --dry-run --range Range_0500
            echo ""
            read -p "Press Enter to continue..."
            ;;
        4)
            echo ""
            echo "Available ranges: Range_0400, Range_0480, Range_0500, Range_1000,"
            echo "                  Range_1100, Range_1200, Range_1300, Range_1400,"
            echo "                  Range_1500, Range_1600, Range_1700"
            echo ""
            read -p "Enter range name: " range_name

            if [[ "$range_name" =~ ^Range_[0-9]{4}$ ]]; then
                echo ""
                echo "Testing $range_name in DRY RUN mode..."
                echo ""
                dsmil-test --dry-run --range "$range_name"
                echo ""
            else
                echo "Invalid range name format"
            fi
            read -p "Press Enter to continue..."
            ;;
        5)
            echo ""
            echo "Recent Test Logs:"
            echo ""
            if [ -d /var/log/dell-milspec ]; then
                find /var/log/dell-milspec -name "token_test_*.log" -mtime -1 2>/dev/null | head -5 | while read log; do
                    echo "  $(basename "$log")"
                done
                echo ""
                read -p "Enter log filename to view (or Enter to skip): " logfile
                if [ -n "$logfile" ] && [ -f "/var/log/dell-milspec/$logfile" ]; then
                    echo ""
                    tail -50 "/var/log/dell-milspec/$logfile"
                    echo ""
                fi
            else
                echo "No log directory found"
            fi
            read -p "Press Enter to continue..."
            ;;
        0)
            echo "Exiting token testing example"
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac

    echo ""
done
