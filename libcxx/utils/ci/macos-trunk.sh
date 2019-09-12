#!/usr/bin/env bash

set -ue

function usage() {
  cat <<EOM
$(basename ${0}) [-h|--help] --monorepo-root <MONOREPO-ROOT> --std <STD> --arch <ARCHITECTURE> [--lit-args <ARGS...>]

This script is used to continually test libc++ and libc++abi trunk on MacOS.

  --monorepo-root     Full path to the root of the LLVM monorepo. Both libc++ and libc++abi from the monorepo are used.
  --std               Version of the C++ Standard to run the tests under (c++03, c++11, etc..).
  --arch              Architecture to build the tests for (32, 64).
  --libcxx-exceptions Whether to enable exceptions when building libc++ and running the libc++ tests. libc++abi is always built with support for exceptions because other libraries in the runtime depend on it (like libobjc). This must be ON or OFF.
  [--cmake-args]      Additional arguments to pass to CMake (both the libc++ and the libc++abi configuration). If there are multiple arguments, quote them to paass them as a single argument to this script.
  [--lit-args]        Additional arguments to pass to lit. If there are multiple arguments, quote them to pass them as a single argument to this script.
  [--no-cleanup]      Do not cleanup the temporary directory that was used for testing at the end. This can be useful to debug failures. Make sure to clean up manually after.
  [-h, --help]        Print this help.
EOM
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --monorepo-root)
    MONOREPO_ROOT="${2}"
    if [[ ! -e "${MONOREPO_ROOT}" ]]; then
      echo "--monorepo-root '${MONOREPO_ROOT}' is not a valid directory"
      usage
      exit 1
    fi
    shift; shift
    ;;
    --std)
    STD="${2}"
    shift; shift
    ;;
    --arch)
    ARCH="${2}"
    shift; shift
    ;;
    --libcxx-exceptions)
    LIBCXX_EXCEPTIONS="${2}"
    shift; shift
    ;;
    --cmake-args)
    ADDITIONAL_CMAKE_ARGS="${2}"
    shift; shift
    ;;
    --lit-args)
    ADDITIONAL_LIT_ARGS="${2}"
    shift; shift
    ;;
    --no-cleanup)
    NO_CLEANUP=""
    shift
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *)
    echo "${1} is not a supported argument"
    usage
    exit 1
    ;;
  esac
done

if [[ -z ${MONOREPO_ROOT+x} ]]; then echo "--monorepo-root is a required parameter"; usage; exit 1; fi
if [[ -z ${STD+x} ]]; then echo "--std is a required parameter"; usage; exit 1; fi
if [[ -z ${ARCH+x} ]]; then echo "--arch is a required parameter"; usage; exit 1; fi
if [[ "${LIBCXX_EXCEPTIONS}" != "ON" && "${LIBCXX_EXCEPTIONS}" != "OFF" ]]; then echo "--libcxx-exceptions is a required parameter and must be either ON or OFF"; usage; exit 1; fi
if [[ -z ${ADDITIONAL_CMAKE_ARGS+x} ]]; then ADDITIONAL_CMAKE_ARGS=""; fi
if [[ -z ${ADDITIONAL_LIT_ARGS+x} ]]; then ADDITIONAL_LIT_ARGS=""; fi


TEMP_DIR="$(mktemp -d)"
echo "Created temporary directory ${TEMP_DIR}"
function cleanup {
  if [[ -z ${NO_CLEANUP+x} ]]; then
    echo "Removing temporary directory ${TEMP_DIR}"
    rm -rf "${TEMP_DIR}"
  else
    echo "Temporary directory is at '${TEMP_DIR}', make sure to clean it up yourself"
  fi
}
trap cleanup EXIT


LLVM_BUILD_DIR="${TEMP_DIR}/llvm-build"
LLVM_INSTALL_DIR="${TEMP_DIR}/llvm-install"

echo "@@@ Setting up LIT flags @@@"
LIT_FLAGS="-sv --param=std=${STD} ${ADDITIONAL_LIT_ARGS}"
if [[ "${ARCH}" == "32" ]]; then
  LIT_FLAGS+=" --param=enable_32bit=true"
fi
echo "@@@@@@"


echo "@@@ Configuring CMake @@@"
mkdir -p "${LLVM_BUILD_DIR}"
(cd "${LLVM_BUILD_DIR}" &&
  xcrun cmake \
    -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Apple.cmake" \
    -GNinja \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}" \
    -DLIBCXX_ENABLE_EXCEPTIONS="${LIBCXX_EXCEPTIONS}" \
    -DLIBCXXABI_ENABLE_EXCEPTIONS=ON \
    ${ADDITIONAL_CMAKE_ARGS} \
    -DLLVM_LIT_ARGS="${LIT_FLAGS}" \
    -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" \
    -DCMAKE_OSX_ARCHITECTURES="i386;x86_64" \
    "${MONOREPO_ROOT}/llvm"
)
echo "@@@@@@"


echo "@@@ Building libc++.dylib and libc++abi.dylib from sources (just to make sure it works) @@@"
ninja -C "${LLVM_BUILD_DIR}" install-cxx install-cxxabi -v
echo "@@@@@@"


echo "@@@ Running tests for libc++ @@@"
# TODO: We should run check-cxx-abilist too
ninja -C "${LLVM_BUILD_DIR}" check-cxx
echo "@@@@@@"


echo "@@@ Running tests for libc++abi @@@"
ninja -C "${LLVM_BUILD_DIR}" check-cxxabi
echo "@@@@@@"
