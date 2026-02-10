#!/usr/bin/env bash
# Usage: bash .ci/build.sh [configure|build|all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

configure() {
    cmake --preset ci-release "${REPO_ROOT}/llvm"
}

build() {
    ninja -C "${BUILD_DIR}" -j"$(nproc)"
}

case "${1:-all}" in
    configure) configure ;;
    build)     build ;;
    all)       configure && build ;;
    *)
        echo "Usage: $0 [configure|build|all]"
        exit 1
        ;;
esac
