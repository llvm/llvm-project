#!/bin/bash
#
# Example: Dell MIL-SPEC Monitoring
# Demonstrates different monitoring modes
#

set -e

echo "======================================================================"
echo " Dell MIL-SPEC Tools - Monitoring Example"
echo "======================================================================"
echo ""
echo "This example demonstrates the different monitoring modes available."
echo ""

while true; do
    echo "Available Monitoring Modes:"
    echo "  1) Dashboard Mode (comprehensive overview)"
    echo "  2) Resource Mode (detailed system resources)"
    echo "  3) Token Mode (DSMIL token monitoring)"
    echo "  4) Alert Mode (alert history)"
    echo "  5) JSON Output (machine-readable)"
    echo "  0) Exit"
    echo ""
    read -p "Select mode [0-5]: " choice

    case $choice in
        1)
            echo ""
            echo "Starting Dashboard Mode..."
            echo "Press Ctrl+C to stop"
            echo ""
            milspec-monitor --mode dashboard
            ;;
        2)
            echo ""
            echo "Starting Resource Mode..."
            echo "Press Ctrl+C to stop"
            echo ""
            milspec-monitor --mode resources
            ;;
        3)
            echo ""
            echo "Starting Token Mode..."
            echo "Press Ctrl+C to stop"
            echo ""
            milspec-monitor --mode tokens
            ;;
        4)
            echo ""
            echo "Starting Alert Mode..."
            echo "Press Ctrl+C to stop"
            echo ""
            milspec-monitor --mode alerts
            ;;
        5)
            echo ""
            echo "JSON Output (single snapshot):"
            echo ""
            milspec-monitor --json-output
            echo ""
            read -p "Press Enter to continue..."
            ;;
        0)
            echo "Exiting monitoring example"
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac

    echo ""
done
