#!/usr/bin/env bash
# Build all EJIT integration tests.
# Usage: ./build.sh [--run] [--arch=x86|aarch64] [--analyze-deps] [--lipo=FILE] [<test>...]
#   --run          Build and run all tests
#   --arch=<arch>  Target architecture (default: auto-detect from build dirs)
#   --analyze-deps Build & show which LLVM .a files were actually linked
#   --lipo=<FILE>  Use a single lipo .a (from lipo.py) instead of 44 individual .a
#   test_name      Build only the named test (without .c extension)
#
# Requires a release build (static libs): ./build.sh release x86
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ARCH=""
SELECTED=()
DO_RUN=false
ANALYZE_DEPS=false
LIPO_FILE=""
NO_STRIP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)         DO_RUN=true ;;
    --analyze-deps) ANALYZE_DEPS=true ;;
    --arch=*)      ARCH="${1#--arch=}" ;;
    --arch)        ARCH="$2"; shift ;;
    --lipo=*)      LIPO_FILE="${1#--lipo=}" ;;
    --lipo)        LIPO_FILE="auto" ;;
    --no-strip)    NO_STRIP=true ;;
    *)             SELECTED+=("$1") ;;
  esac
  shift
done

# Auto-detect arch from available build dirs (prefer release, fallback to debug-static)
find_build_dir() {
  local arch="$1"
  for dir in \
    "${ROOT_DIR}/build_release_${arch}" \
    "${ROOT_DIR}/build_debug_${arch}_static"; do
    if [[ -d "${dir}" && -f "${dir}/lib/libLLVMEJIT.a" ]]; then
      echo "${dir}"; return 0
    fi
  done
  return 1
}

if [[ -z "${ARCH}" ]]; then
  for a in x86 aarch64; do
    BUILD_DIR=$(find_build_dir "$a" || true)
    if [[ -n "${BUILD_DIR}" ]]; then ARCH="$a"; break; fi
  done
  if [[ -z "${ARCH}" ]]; then
    echo "ERROR: no build with static libs found."
    echo "  Run: ../build.sh release x86"
    echo "    or ../build.sh debug x86 --static"
    exit 1
  fi
else
  BUILD_DIR=$(find_build_dir "${ARCH}" || true)
  if [[ -z "${BUILD_DIR}" ]]; then
    echo "ERROR: no static-lib build for ${ARCH}."
    echo "  Run: ../build.sh release ${ARCH}"
    echo "    or ../build.sh debug ${ARCH} --static"
    exit 1
  fi
fi

# Everything from a single release build directory (static libs only)
CLANG="${BUILD_DIR}/bin/clang"
CXX="${CLANG}++"
BUILD_INCLUDE="${BUILD_DIR}/include"
LLVM_INCLUDE="${ROOT_DIR}/llvm/include"
INCLUDES="-I${LLVM_INCLUDE} -I${BUILD_INCLUDE}"

LD_LLD="${BUILD_DIR}/bin/ld.lld"
[[ -x "${LD_LLD}" ]] || { echo "ERROR: lld not found at ${LD_LLD}"; exit 1; }

EJIT_RUNTIME="${BUILD_DIR}/lib/libLLVMEJIT.a"
[[ -f "${EJIT_RUNTIME}" ]] || { echo "ERROR: libLLVMEJIT.a not found in ${BUILD_DIR}/lib/"; exit 1; }

# Minimal .a set verified by link-then-test on 2026-05-25.
# Core = EJIT's direct LINK_COMPONENTS + essential transitive deps
# (OrcJIT needs OrcTargetProcess/RuntimeDyld/BitWriter,
#  X86CodeGen needs GlobalISel/CFGuard/IRPrinter/Instrumentation,
#  CodeGen needs ObjCARCOpts/CGData,
#  AsmPrinter needs DebugInfoDWARF/CodeView/DWARFLowLevel/MCParser,
#  X86Desc needs MCDisassembler, JITLink needs Option, Core needs Remarks).
_set_min_libs() {
  local _l="${BUILD_DIR}/lib"
  # shared across all targets
  MIN_LIBS_COMMON="
    ${_l}/libLLVMCore.a ${_l}/libLLVMSupport.a ${_l}/libLLVMDemangle.a
    ${_l}/libLLVMBinaryFormat.a ${_l}/libLLVMBitReader.a ${_l}/libLLVMBitstreamReader.a
    ${_l}/libLLVMAnalysis.a ${_l}/libLLVMScalarOpts.a ${_l}/libLLVMInstCombine.a
    ${_l}/libLLVMipo.a ${_l}/libLLVMTransformUtils.a ${_l}/libLLVMCodeGen.a
    ${_l}/libLLVMCodeGenTypes.a ${_l}/libLLVMTarget.a ${_l}/libLLVMTargetParser.a
    ${_l}/libLLVMSelectionDAG.a ${_l}/libLLVMAsmPrinter.a ${_l}/libLLVMMC.a
    ${_l}/libLLVMObject.a ${_l}/libLLVMProfileData.a ${_l}/libLLVMExecutionEngine.a
    ${_l}/libLLVMOrcJIT.a ${_l}/libLLVMOrcShared.a ${_l}/libLLVMJITLink.a
    ${_l}/libLLVMRemarks.a ${_l}/libLLVMOption.a ${_l}/libLLVMMCDisassembler.a
    ${_l}/libLLVMIRPrinter.a ${_l}/libLLVMCFGuard.a
    ${_l}/libLLVMInstrumentation.a
    ${_l}/libLLVMDebugInfoCodeView.a ${_l}/libLLVMDebugInfoDWARF.a
    ${_l}/libLLVMDebugInfoDWARFLowLevel.a ${_l}/libLLVMMCParser.a
    ${_l}/libLLVMCGData.a ${_l}/libLLVMObjCARCOpts.a
    ${_l}/libLLVMGlobalISel.a
    ${_l}/libLLVMOrcTargetProcess.a ${_l}/libLLVMRuntimeDyld.a
    ${_l}/libLLVMBitWriter.a
  "
  case "$1" in
    x86)
      MIN_LIBS="${MIN_LIBS_COMMON}
        ${_l}/libLLVMX86CodeGen.a ${_l}/libLLVMX86Desc.a ${_l}/libLLVMX86Info.a"
      ;;
    aarch64)
      MIN_LIBS="${MIN_LIBS_COMMON}
        ${_l}/libLLVMAArch64CodeGen.a ${_l}/libLLVMAArch64Desc.a
        ${_l}/libLLVMAArch64Info.a ${_l}/libLLVMAArch64Utils.a"
      ;;
  esac
}

if [[ -n "${LIPO_FILE}" ]]; then
  # Lipo mode: use single pre-built .a instead of 44 individual .a files.
  # The lipo .a already bundles EJIT + all LLVM dependencies (see lipo.py).
  # If --lipo is passed without =FILE, auto-detect from build dir + arch.
  if [[ "${LIPO_FILE}" == "auto" ]]; then
    LIPO_FILE="${SCRIPT_DIR}/lipo/ejit.o"
  fi
  [[ -f "${LIPO_FILE}" ]] || { echo "ERROR: lipo file not found: ${LIPO_FILE}"; exit 1; }
  USE_LIPO=true
  LIPO_ABS=$(cd "$(dirname "${LIPO_FILE}")" && pwd)/$(basename "${LIPO_FILE}")
  echo "Lipo mode: ${LIPO_ABS}"
else
  USE_LIPO=false
  _set_min_libs "${ARCH}"
  OTHER_LIBS=$(echo "${MIN_LIBS}" | sed '/^$/d' | tr '\n' ' ')
fi
LINK_LIBS="-lz -lpthread -ldl"
STRIP_FLAG="-Wl,--strip-all"
if ${NO_STRIP}; then STRIP_FLAG=""; fi

OUTDIR="${SCRIPT_DIR}/out"
mkdir -p "${OUTDIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ALL_TESTS=(
  ejit_attr_test
  ejit_complex_test
  ejit_config_api_test
  ejit_dump_test
  ejit_external_idx_test
  ejit_jit_verify_test
  ejit_lifecycle_test
  ejit_multidim_test
  ejit_multiversion_test
  ejit_nested_struct_test
  ejit_opt_level_test
  ejit_manual_register_test
  ejit_perf_bench
  ejit_ptr_period_test
  ejit_trace_test
)

# Per-test compile flags (e.g. for disabling global constructors)
declare -A COMPILE_FLAGS
COMPILE_FLAGS[ejit_manual_register_test]="-mllvm -enable-ejit-global-ctors=false"

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
TEST_ARGS[ejit_dump_test]="0 3"
TEST_ARGS[ejit_ptr_period_test]="0 1 3"

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${ALL_TESTS[@]}")
fi

# --- Build ---
build_one() {
  local name="$1"
  local src="${SCRIPT_DIR}/${name}.c"
  local obj="${TMPDIR:-/tmp}/ejit_${name}.o"
  local bin="${OUTDIR}/${name}"
  local map_file="${TMPDIR:-/tmp}/ejit_${name}.map"

  if [[ ! -f "${src}" ]]; then
    echo -e "${RED}  SKIP: ${src} not found${NC}"
    return 1
  fi

  echo "  Compiling ${name}.c ..."
  local extra_flags="${COMPILE_FLAGS[${name}]:-}"
  "${CLANG}" -O2 ${INCLUDES} ${extra_flags} -c "${src}" -o "${obj}"

  echo "  Linking ${name} (${ARCH}) ..."
  if ${USE_LIPO}; then
    "${CXX}" -fuse-ld="${LD_LLD}" \
      -Os -Wl,--gc-sections ${STRIP_FLAG} \
      "${LIPO_ABS}" \
      ${LINK_LIBS} \
      "${obj}" -o "${bin}"
    echo "${CXX}" -fuse-ld="${LD_LLD}" \
      -Os -Wl,--gc-sections ${STRIP_FLAG} \
      "${LIPO_ABS}" \
      ${LINK_LIBS} \
      "${obj}" -o "${bin}"
  else
    "${CXX}" -fuse-ld="${LD_LLD}" \
      -Os -Wl,--gc-sections ${STRIP_FLAG} \
      -Wl,--whole-archive "${EJIT_RUNTIME}" -Wl,--no-whole-archive \
      ${OTHER_LIBS} \
      ${LINK_LIBS} \
      -Wl,-M \
      "${obj}" -o "${bin}" > "${map_file}" 2>&1
  fi

  echo -e "  ${GREEN}OK${NC}: ${bin}"

  # Show dependency summary on request
  if ${ANALYZE_DEPS}; then
    if ${USE_LIPO}; then
      local lipo_sz=$(du -h "${LIPO_ABS}" | cut -f1)
      echo "         Lipo mode: 1 .a (${lipo_sz}) — ${LIPO_ABS}"
    else
      local deps
      deps=$(grep -oP "${BUILD_DIR}/lib/\K[^(]+\.a" "${map_file}" | sort -u)
      local dep_count
      dep_count=$(echo "${deps}" | grep -c '.' || true)
      echo "         ${dep_count} LLVM .a files linked:"
      echo "${deps}" | while IFS= read -r lib; do
        [[ -n "${lib}" ]] && echo "           ${lib}"
      done
    fi
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
  local log="${TMPDIR:-/tmp}/ejit_${name}.log"
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

BUILD_FAILED=0
for t in "${SELECTED[@]}"; do
  build_one "$t" || BUILD_FAILED=1
done
if [[ $BUILD_FAILED -ne 0 ]]; then
  echo -e "${RED}Build failed, stopping.${NC}"
  exit 1
fi

if ${DO_RUN}; then
  echo ""
  echo "=== Running Tests ==="
  for t in "${SELECTED[@]}"; do
    run_one "$t"
  done
fi

echo ""
echo "=== Done ==="
