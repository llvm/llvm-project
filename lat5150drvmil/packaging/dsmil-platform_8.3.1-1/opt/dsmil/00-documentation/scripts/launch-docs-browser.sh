#!/bin/bash

# DSMIL Documentation Browser Launcher with Virtual Environment
# Handles Python package management for Ubuntu 24.04+

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-docs"

echo "═══════════════════════════════════════════════════════════════"
echo "  DSMIL 72-Device Documentation Browser"
echo "  Advanced Document Viewer with AI Analysis"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Function to setup virtual environment
setup_venv() {
    echo "Setting up Python virtual environment..."
    
    # Check if python3-venv is installed
    if ! dpkg -l | grep -q python3-venv; then
        echo "Installing python3-venv..."
        sudo apt-get update
        sudo apt-get install -y python3-venv python3-tk
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Install required packages
    echo "Installing required packages..."
    pip install --quiet --upgrade pip
    pip install --quiet pdfplumber 2>/dev/null || true
    pip install --quiet scikit-learn 2>/dev/null || true
    pip install --quiet markdown 2>/dev/null || true
    
    echo "Virtual environment ready!"
    echo ""
}

# Main execution
main() {
    # Setup virtual environment
    setup_venv
    
    # Default to project root
    DOC_DIR="${1:-$SCRIPT_DIR}"
    
    echo "Documentation Directory: $DOC_DIR"
    echo ""
    echo "Key DSMIL Documents:"
    echo "  📄 DSMIL-72-DEVICE-DISCOVERY-COMPLETE.md"
    echo "  📋 DSMIL-AGENT-COORDINATION-PLAN.md"
    echo "  🏗️ docs/DSMIL_ARCHITECTURE_ANALYSIS.md"
    echo "  🛡️ docs/DSMIL_SAFE_PROBING_METHODOLOGY.md"
    echo "  🔧 docs/DSMIL_MODULAR_ACCESS_FRAMEWORK.md"
    echo "  📚 docs/DSMIL-DOCUMENTATION-INDEX.md"
    echo ""
    echo "Launching browser..."
    echo ""
    
    # Run the documentation browser
    cd "$SCRIPT_DIR"
    python3 docs/universal_docs_browser_enhanced.py "$DOC_DIR"
    
    # Deactivate virtual environment
    deactivate 2>/dev/null || true
    
    echo ""
    echo "Documentation browser closed."
}

# Handle Ctrl+C gracefully
trap 'echo ""; echo "Interrupted. Cleaning up..."; deactivate 2>/dev/null || true; exit 1' INT

# Run main function
main "$@"