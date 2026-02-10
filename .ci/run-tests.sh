#!/usr/bin/env bash
# Usage: bash .ci/run-tests.sh [--retry]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/build"

run_tests() {
    ninja -C "${BUILD_DIR}" check-clang check-llvm
}

if run_tests; then
    exit 0
fi

EXIT_CODE=$?

if [[ "${1:-}" == "--retry" ]]; then
    echo "Tests failed, retrying once..."
    if run_tests; then
        echo "Passed on retry — likely flaky. Check .ci/flaky-tests.txt"
        exit 0
    fi
fi

exit ${EXIT_CODE}
