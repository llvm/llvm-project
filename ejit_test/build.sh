#!/usr/bin/env bash
# Build all EJIT integration tests.
# Usage: ./build.sh [--run] [<test_name> ...]
#   --run     Build and run all tests
#   test_name Build only the named test (without .c extension)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CLANG="${ROOT_DIR}/build/bin/clang"
CXX="${CLANG}++"  # use clang++ for linking C++ runtime
BUILD_DIR="${ROOT_DIR}/build_x86"
BUILD_INCLUDE="${BUILD_DIR}/include"
LLVM_INCLUDE="${ROOT_DIR}/llvm/include"
LLVM_BUILD_INCLUDE="${ROOT_DIR}/build/include"

INCLUDES="-I${LLVM_INCLUDE} -I${BUILD_INCLUDE} -I${LLVM_BUILD_INCLUDE}"
LIBS=$(ls "${BUILD_DIR}/lib/"*.a 2>/dev/null | grep -v gtest || true)
LINK_FLAGS="-Os -Wl,--gc-sections -Wl,--strip-all"
LINK_LIBS="-lz -lpthread -ldl"

OUTDIR="${SCRIPT_DIR}/out"
mkdir -p "${OUTDIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# All test files (C source files in this directory, excluding helpers)
ALL_TESTS=(
  ejit_attr_test
  ejit_complex_test
  ejit_config_api_test
  ejit_external_idx_test
  ejit_jit_verify_test
  ejit_lifecycle_test
  ejit_multiversion_test
  ejit_opt_level_test
  ejit_perf_bench
  ejit_trace_test
)

# Extra args to pass to each test when running
declare -A TEST_ARGS
TEST_ARGS[ejit_complex_test]="0 1 2 3"
TEST_ARGS[ejit_opt_level_test]="L2"
TEST_ARGS[ejit_multiversion_test]="0 3 7"
TEST_ARGS[ejit_jit_verify_test]="0 1 5"
TEST_ARGS[ejit_external_idx_test]="3 1"
TEST_ARGS[ejit_lifecycle_test]="3 7 2"
TEST_ARGS[ejit_trace_test]="0"
TEST_ARGS[ejit_attr_test]="0"
TEST_ARGS[ejit_config_api_test]="0"
TEST_ARGS[ejit_perf_bench]="0 1"  # perf bench: cellIdx=0, 1 warmup round

build_one() {
  local name="$1"
  local src="${SCRIPT_DIR}/${name}.c"
  local obj="/tmp/ejit_${name}.o"
  local bin="${OUTDIR}/${name}"

  if [[ ! -f "${src}" ]]; then
    echo -e "${RED}  SKIP: ${src} not found${NC}"
    return 1
  fi

  echo "  Compiling ${name}.c ..."
  "${CLANG}" -O2 ${INCLUDES} -c "${src}" -o "${obj}"
  echo "  Linking ${name} ..."
  "${CXX}" ${LINK_FLAGS} "${obj}" \
    -Wl,--whole-archive ${LIBS} -Wl,--no-whole-archive \
    ${LINK_LIBS} -o "${bin}"
  echo -e "  ${GREEN}OK${NC}: ${bin}"
}

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

# Parse args
DO_RUN=false
SELECTED=()

for arg in "$@"; do
  case "$arg" in
    --run) DO_RUN=true ;;
    *)     SELECTED+=("$arg") ;;
  esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${ALL_TESTS[@]}")
fi

echo "=== Building EJIT Integration Tests ==="
echo "Output: ${OUTDIR}"
echo "Tests:  ${SELECTED[*]}"
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
