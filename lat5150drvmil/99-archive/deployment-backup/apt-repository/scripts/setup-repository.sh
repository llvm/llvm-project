#!/bin/bash
# APT Repository Setup Script
# Creates self-hosted Dell MIL-SPEC APT repository

set -euo pipefail

REPO_BASE="/home/john/LAT5150DRVMIL/deployment/apt-repository"
REPO_NAME="dell-milspec"
DISTRIBUTIONS="stable testing unstable"
ARCHITECTURES="amd64"
COMPONENTS="main"

echo "═══════════════════════════════════════════════════════"
echo "  Dell MIL-SPEC APT Repository Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

# Create directory structure
echo "Creating repository structure..."
mkdir -p ${REPO_BASE}/{pool/main,gpg,incoming}

for dist in $DISTRIBUTIONS; do
    mkdir -p ${REPO_BASE}/dists/${dist}/${COMPONENTS}/binary-${ARCHITECTURES}
    mkdir -p ${REPO_BASE}/dists/${dist}/${COMPONENTS}/source
done

echo "✓ Directory structure created"

# Create reprepro configuration
echo "Creating reprepro configuration..."
mkdir -p ${REPO_BASE}/conf

cat > ${REPO_BASE}/conf/distributions << EOF
Origin: Dell MIL-SPEC
Label: dell-milspec
Codename: stable
Architectures: amd64 source
Components: main
Description: Dell MIL-SPEC Platform Packages (Stable)
SignWith: yes

Origin: Dell MIL-SPEC
Label: dell-milspec-testing
Codename: testing
Architectures: amd64 source
Components: main
Description: Dell MIL-SPEC Platform Packages (Testing)
SignWith: yes

Origin: Dell MIL-SPEC
Label: dell-milspec-unstable
Codename: unstable
Architectures: amd64 source
Components: main
Description: Dell MIL-SPEC Platform Packages (Unstable)
SignWith: no
EOF

cat > ${REPO_BASE}/conf/options << EOF
verbose
basedir ${REPO_BASE}
ask-passphrase
EOF

echo "✓ Repository configuration created"

# Generate GPG key for signing (if needed)
echo ""
echo "GPG Key Setup:"
echo "  If you don't have a GPG key, generate one with:"
echo "    gpg --full-generate-key"
echo "  Then export it:"
echo "    gpg --armor --export YOUR_EMAIL > ${REPO_BASE}/gpg/public-key.asc"
echo ""

# Create helper scripts
cat > ${REPO_BASE}/scripts/add-package.sh << 'EOFADDPKG'
#!/bin/bash
# Add package to repository

PACKAGE_FILE="$1"
DISTRIBUTION="${2:-stable}"
REPO_BASE="/home/john/LAT5150DRVMIL/deployment/apt-repository"

if [ -z "$PACKAGE_FILE" ]; then
    echo "Usage: $0 <package.deb> [distribution]"
    echo "  distribution: stable|testing|unstable (default: stable)"
    exit 1
fi

if [ ! -f "$PACKAGE_FILE" ]; then
    echo "Error: Package file not found: $PACKAGE_FILE"
    exit 1
fi

cd ${REPO_BASE}
reprepro -b . includedeb ${DISTRIBUTION} ${PACKAGE_FILE}
echo "✓ Added $(basename $PACKAGE_FILE) to ${DISTRIBUTION}"
EOFADDPKG
chmod +x ${REPO_BASE}/scripts/add-package.sh

cat > ${REPO_BASE}/scripts/update-repository.sh << 'EOFUPDATE'
#!/bin/bash
# Update repository metadata

REPO_BASE="/home/john/LAT5150DRVMIL/deployment/apt-repository"

cd ${REPO_BASE}

echo "Updating repository metadata..."
for dist in stable testing unstable; do
    reprepro -b . export ${dist} || true
done

echo "✓ Repository updated"
echo ""
echo "Repository accessible at:"
echo "  file://${REPO_BASE}"
echo ""
echo "Add to sources.list:"
echo "  deb [trusted=yes] file://${REPO_BASE} stable main"
EOFUPDATE
chmod +x ${REPO_BASE}/scripts/update-repository.sh

cat > ${REPO_BASE}/scripts/list-packages.sh << 'EOFLIST'
#!/bin/bash
# List all packages in repository

REPO_BASE="/home/john/LAT5150DRVMIL/deployment/apt-repository"

cd ${REPO_BASE}

for dist in stable testing unstable; do
    echo "═══ ${dist} ==="
    reprepro -b . list ${dist} 2>/dev/null || echo "  (empty)"
    echo ""
done
EOFLIST
chmod +x ${REPO_BASE}/scripts/list-packages.sh

echo "✓ Helper scripts created"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Repository Setup Complete"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Location: ${REPO_BASE}"
echo ""
echo "Next steps:"
echo "  1. Generate or import GPG key"
echo "  2. Add packages: ./scripts/add-package.sh package.deb"
echo "  3. Update metadata: ./scripts/update-repository.sh"
echo "  4. Configure web server (nginx) for HTTPS access"
echo ""
