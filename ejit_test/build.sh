#!/usr/bin/env bash
# Build all EJIT integration tests.
# Usage: ./build.sh [--run] [--arch=x86|aarch64] [--analyze-deps] [<test_name> ...]
#   --run          Build and run all tests
#   --arch=<arch>  Target architecture (default: auto-detect from build dirs)
#   --analyze-deps Build & show which LLVM .a files were actually linked
#   test_name      Build only the named test (without .c extension)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Architecture setup ---
ARCH=""
SELECTED=()
DO_RUN=false
ANALYZE_DEPS=false

for arg in "$@"; do
  case "$arg" in
    --run)         DO_RUN=true ;;
    --analyze-deps) ANALYZE_DEPS=true ;;
    --arch=*)      ARCH="${arg#--arch=}" ;;
    --arch)        shift; ARCH="$1" ;;
    *)             SELECTED+=("$arg") ;;
  esac
done

# Auto-detect: prefer X86, fallback to aarch64, else error
if [[ -z "${ARCH}" ]]; then
  if [[ -d "${ROOT_DIR}/build_x86" ]]; then
    ARCH="x86"
  elif [[ -d "${ROOT_DIR}/build_aarch64" ]]; then
    ARCH="aarch64"
  else
    echo "ERROR: could not auto-detect architecture. Set --arch=x86|aarch64"
    exit 1
  fi
fi

case "${ARCH}" in
  x86|X86|x86_64)
    BUILD_DIR="${ROOT_DIR}/build_x86"
    ;;
  aarch64|AArch64|arm64)
    BUILD_DIR="${ROOT_DIR}/build_aarch64"
    ;;
  *)
    echo "ERROR: unknown architecture '${ARCH}'. Use x86 or aarch64."
    exit 1
    ;;
esac

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "ERROR: build directory '${BUILD_DIR}' not found."
  echo "  Create it first: cmake -S llvm -B ${BUILD_DIR} ... -DLLVM_TARGETS_TO_BUILD=${ARCH^}"
  exit 1
fi

CLANG="${ROOT_DIR}/build/bin/clang"
CXX="${CLANG}++"

BUILD_INCLUDE="${BUILD_DIR}/include"
LLVM_INCLUDE="${ROOT_DIR}/llvm/include"
LLVM_BUILD_INCLUDE="${ROOT_DIR}/build/include"

INCLUDES="-I${LLVM_INCLUDE} -I${BUILD_INCLUDE} -I${LLVM_BUILD_INCLUDE}"

# --- Linker (lld) from the target build ---
LD_LLD="${BUILD_DIR}/bin/ld.lld"
if [[ ! -x "${LD_LLD}" ]]; then
  echo "ERROR: lld not found at ${LD_LLD}. Build it first: ninja -C ${BUILD_DIR} lld"
  exit 1
fi

# --- Library resolution ---
# Only --whole-archive the EJIT runtime (libLLVMEJIT.a) — it contains
# SyncCompiler/AsyncCompiler whose symbols are reached indirectly
# (factory / virtual dispatch), so the linker can't see them via normal
# undefined-symbol scanning.
#
# libLLVMEmbeddedJIT.a (AOT passes) is NOT linked — it runs inside clang
# at compile time, never at runtime.
#
# All other LLVM/Clang .a files are resolved by the linker automatically.
EJIT_RUNTIME="${BUILD_DIR}/lib/libLLVMEJIT.a"
OTHER_LIBS=$(ls "${BUILD_DIR}/lib/"*.a 2>/dev/null | grep -v gtest | grep -v libLLVMEJIT | grep -v libLLVMEmbeddedJIT || true)
LINK_LIBS="-lz -lpthread -ldl"

if [[ ! -f "${EJIT_RUNTIME}" ]]; then
  echo "ERROR: EJIT runtime not found at ${EJIT_RUNTIME}"
  exit 1
fi

OUTDIR="${SCRIPT_DIR}/out"
mkdir -p "${OUTDIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ALL_TESTS=(
  ejit_attr_test
  ejit_complex_test
  ejit_config_api_test
  ejit_external_idx_test
  ejit_jit_verify_test
  ejit_lifecycle_test
  ejit_multidim_test
  ejit_multiversion_test
  ejit_nested_struct_test
  ejit_opt_level_test
  ejit_perf_bench
  ejit_ptr_period_test
  ejit_trace_test
)

declare -A TEST_ARGS
TEST_ARGS[ejit_complex_test]="0 1 2 3"
TEST_ARGS[ejit_opt_level_test]="L2"
TEST_ARGS[ejit_multiversion_test]="0 3 7"
TEST_ARGS[ejit_jit_verify_test]="0 1 5"
TEST_ARGS[ejit_external_idx_test]="3 1"
TEST_ARGS[ejit_lifecycle_test]="3 7 2"
TEST_ARGS[ejit_multidim_test]="0"
TEST_ARGS[ejit_nested_struct_test]="0"
TEST_ARGS[ejit_trace_test]="0"
TEST_ARGS[ejit_attr_test]="0"
TEST_ARGS[ejit_config_api_test]="0"
TEST_ARGS[ejit_perf_bench]="0 1"
TEST_ARGS[ejit_ptr_period_test]="0 1 3"

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${ALL_TESTS[@]}")
fi

# --- Build ---
build_one() {
  local name="$1"
  local src="${SCRIPT_DIR}/${name}.c"
  local obj="/tmp/ejit_${name}.o"
  local bin="${OUTDIR}/${name}"
  local map_file="/tmp/ejit_${name}.map"

  if [[ ! -f "${src}" ]]; then
    echo -e "${RED}  SKIP: ${src} not found${NC}"
    return 1
  fi

  echo "  Compiling ${name}.c ..."
  "${CLANG}" -O2 ${INCLUDES} -c "${src}" -o "${obj}"

  echo "  Linking ${name} (${ARCH}) ..."
  "${CXX}" -fuse-ld="${LD_LLD}" \
    -Os -Wl,--gc-sections -Wl,--strip-all \
    -Wl,--whole-archive "${EJIT_RUNTIME}" -Wl,--no-whole-archive \
    ${OTHER_LIBS} \
    ${LINK_LIBS} \
    -Wl,-M \
    "${obj}" -o "${bin}" > "${map_file}" 2>&1

  echo -e "  ${GREEN}OK${NC}: ${bin}"

  # Show dependency summary on request
  if ${ANALYZE_DEPS}; then
    local dep_count
    dep_count=$(grep -oP "${BUILD_DIR}/lib/\K[^(]+\.a" "${map_file}" | sort -u | wc -l)
    echo "         ${dep_count} LLVM .a files linked (map: ${map_file})"
  fi
}

# --- Run ---
run_one() {
  local name="$1"
  local bin="${OUTDIR}/${name}"
  local args="${TEST_ARGS[$name]:-}"

  if [[ ! -x "${bin}" ]]; then
    echo -e "${RED}  SKIP: ${bin} not found${NC}"
    return 1
  fi

  echo "  Running ${name} ${args} ..."
  local log="/tmp/ejit_${name}.log"
  if ${bin} ${args} > "${log}" 2>&1; then
    local nfails
    nfails=$(grep -c "FAIL" "${log}" 2>/dev/null || true)
    nfails=$(echo "${nfails}" | tr -d '[:space:]')
    if [[ -z "${nfails}" || "${nfails}" == "0" ]]; then
      echo -e "  ${GREEN}PASS${NC}"
    else
      echo -e "  ${RED}FAIL${NC} (${nfails} check failures)"
      grep "FAIL" "${log}" | head -5
    fi
  else
    echo -e "  ${RED}FAIL${NC} (exit code $?)"
    tail -10 "${log}"
  fi
}

echo "=== Building EJIT Integration Tests ==="
echo "Arch:    ${ARCH}"
echo "Build:   ${BUILD_DIR}"
echo "Output:  ${OUTDIR}"
echo "Tests:   ${SELECTED[*]}"
echo ""

for t in "${SELECTED[@]}"; do
  build_one "$t"
done

if ${DO_RUN}; then
  echo ""
  echo "=== Running Tests ==="
  for t in "${SELECTED[@]}"; do
    run_one "$t"
  done
fi

echo ""
echo "=== Done ==="
