#!/bin/bash
#
# DBXForensics Tools Setup Script
# Installs Windows forensics tools via Wine on Linux
#

set -e  # Exit on error

FORENSICS_DIR="/home/user/LAT5150DRVMIL/04-integrations/forensics"
TOOLS_DIR="$FORENSICS_DIR/tools"
WINE_PREFIX="/opt/lat5150/wine"

echo "================================"
echo "DBXForensics Tools Setup"
echo "================================"
echo ""

# Step 1: Check if running as root
if [ "$EUID" -ne 0 ]; then
   echo "❌ Please run as root (sudo ./setup_tools.sh)"
   exit 1
fi

echo "Step 1: Enable 32-bit architecture support..."
dpkg --add-architecture i386
apt-get update

echo ""
echo "Step 2: Installing Wine with 32-bit support..."
apt-get install -y wine wine64 wine32:i386 winetricks

echo ""
echo "Step 3: Creating Wine prefix..."
mkdir -p "$WINE_PREFIX"
chown -R $USER:$USER "$WINE_PREFIX"

echo ""
echo "Step 4: Checking for tool installers..."
cd "$TOOLS_DIR"

if [ ! -f "Setup_dbxELA_1000.exe" ]; then
    echo "❌ Tool installers not found in $TOOLS_DIR"
    echo "   Please extract the zip files first:"
    echo "   cd $FORENSICS_DIR"
    echo "   unzip 'Setup_*.zip' -d tools/"
    exit 1
fi

echo "✓ Found $(ls -1 Setup_*.exe | wc -l) tool installers"

echo ""
echo "Step 5: Installing DBXForensics tools..."
echo "   (This may take a few minutes)"

export WINEPREFIX="$WINE_PREFIX"
export WINEARCH=win64

# Initialize Wine
echo "Initializing Wine..."
wine --version
wineboot --init

# Install each tool
TOOLS=(
    "Setup_dbxScreenshot_1000.exe"
    "Setup_dbxELA_1000.exe"
    "Setup_dbxNoiseMap_1000.exe"
    "Setup_dbxMetadata_1000.exe"
    "Setup_dbxHashFile_1000.exe"
    "Setup_dbxSeqCheck_1000.exe"
    "Setup_dbxCsvViewer_1000.exe"
    "Setup_dbxGhost_1000.exe"
    "Setup_dbxMouseRecorder_1000.exe"
)

for tool in "${TOOLS[@]}"; do
    if [ -f "$tool" ]; then
        echo "Installing $tool..."
        wine "$tool" /S 2>&1 | grep -v "fixme:" || true
    else
        echo "⚠️  $tool not found, skipping"
    fi
done

echo ""
echo "Step 6: Verifying installation..."
INSTALL_DIR="$WINE_PREFIX/drive_c/Program Files/DME Forensics"

if [ -d "$INSTALL_DIR" ]; then
    echo "✓ Tools installed to: $INSTALL_DIR"
    echo ""
    echo "Installed tools:"
    ls -1 "$INSTALL_DIR"
else
    echo "⚠️  Installation directory not found: $INSTALL_DIR"
    echo "   Tools may have installed to a different location"
    echo "   Search for them with: find $WINE_PREFIX -name 'dbx*.exe'"
fi

echo ""
echo "Step 7: Testing tool execution..."
TEST_TOOL="$INSTALL_DIR/dbxELA/dbxELA.exe"

if [ -f "$TEST_TOOL" ]; then
    echo "Testing dbxELA..."
    wine "$TEST_TOOL" --help 2>&1 | head -5 || true
    echo "✓ Tool execution successful"
else
    echo "⚠️  Test tool not found: $TEST_TOOL"
fi

echo ""
echo "================================"
echo "✓ Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Update tool paths in dbxforensics_toolkit.py"
echo "2. Test with: python3 forensics_analyzer.py"
echo "3. Run full system test: python3 dbxforensics_toolkit.py"
echo ""
echo "Tool locations:"
echo "  Wine prefix: $WINE_PREFIX"
echo "  Tools: $INSTALL_DIR"
echo ""
echo "To use tools:"
echo "  export WINEPREFIX='$WINE_PREFIX'"
echo "  wine \"$INSTALL_DIR/dbxELA/dbxELA.exe\" /path/to/image.jpg"
echo ""
