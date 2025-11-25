#!/bin/bash
#
# Ensure LAT5150 virtual environment exists and dependencies are up to date.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
REQ_FILE="${PROJECT_ROOT}/requirements.txt"
HASH_FILE="${VENV_DIR}/.req_hash"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
    echo "[lat5150-venv] $1"
}

safe_pip_install() {
    if ! "$@" >/dev/null 2>&1; then
        log "Warning: '$*' failed (network or PyPI unavailable); skipping."
        return 1
    fi
    return 0
}

have_pypi_access() {
    "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import urllib.request
try:
    urllib.request.urlopen("https://pypi.org/simple", timeout=3)
except Exception:
    raise SystemExit(1)
PY
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    log "Error: Python interpreter '${PYTHON_BIN}' not found."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtualenv at ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

OPTIONAL_PACKAGES=("langchain" "google-generativeai" "duckduckgo-search")

if have_pypi_access; then
    if [ -f "$REQ_FILE" ]; then
        CURRENT_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
        OLD_HASH=""

        if [ -f "$HASH_FILE" ]; then
            OLD_HASH="$(cat "$HASH_FILE" 2>/dev/null || true)"
        fi

    if [ "$CURRENT_HASH" != "$OLD_HASH" ]; then
        log "requirements.txt changed; installing dependencies into .venv"

        log "Upgrading pip/setuptools/wheel in .venv..."
        "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel

        log "Installing packages from requirements.txt (this may take a while)..."
        "${VENV_DIR}/bin/pip" install -r "$REQ_FILE"

        echo "$CURRENT_HASH" > "$HASH_FILE"
        log "Dependencies installed and hash recorded."
    else
        log "requirements.txt unchanged; skipping dependency install."
    fi
    else
        log "No requirements.txt found; skipping dependency install."
    fi

    for pkg in "${OPTIONAL_PACKAGES[@]}"; do
        log "Ensuring optional package '${pkg}' is installed..."
        safe_pip_install "${VENV_DIR}/bin/pip" install "$pkg" || log "Optional package ${pkg} could not be installed"
    done
else
    log "PyPI unreachable – skipping dependency and optional package installation."
fi
