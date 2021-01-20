#!/usr/bin/env bash

set -ue

function usage() {
  cat <<EOM
$(basename ${0}) [-h|--help] --monorepo-root <MONOREPO-ROOT> --std <STD> --deployment-target <TARGET> [--libcxx-roots <DIR>] [--lit-args <ARGS...>] [--no-cleanup]

This script is used to continually test the back-deployment use case of libc++ and libc++abi on MacOS.

Specifically, this script runs the libc++ test suite against the just-built headers and linking against the just-built dylib, but it runs the tests against the dylibs for the given deployment target.

  --monorepo-root     Full path to the root of the LLVM monorepo. Both libc++ and libc++abi headers from the monorepo are used.
  --std               Version of the C++ Standard to run the tests under (c++03, c++11, etc..).
  --deployment-target The deployment target to run the tests for. This should be a version number of MacOS (e.g. 10.12). All MacOS versions until and including 10.9 are supported.
  [--libcxx-roots]    The path to previous libc++/libc++abi dylibs to use for back-deployment testing. Those are normally downloaded automatically, but if specified, this option will override the directory used. The directory should have the same layout as the roots downloaded automatically.
  [--lit-args]        Additional arguments to pass to lit (optional). If there are multiple arguments, quote them to pass them as a single argument to this script.
  [--no-cleanup]      Do not cleanup the temporary directory that was used for testing at the end. This can be useful to debug failures. Make sure to clean up manually after.
  [-h, --help]        Print this help.
EOM
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --monorepo-root)
    MONOREPO_ROOT="${2}"
    if [[ ! -d "${MONOREPO_ROOT}" ]]; then
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
    --deployment-target)
    DEPLOYMENT_TARGET="${2}"
    shift; shift
    ;;
    --lit-args)
    ADDITIONAL_LIT_ARGS="${2}"
    shift; shift
    ;;
    --libcxx-roots)
    PREVIOUS_DYLIBS_DIR="${2}"
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
if [[ -z ${DEPLOYMENT_TARGET+x} ]]; then echo "--deployment-target is a required parameter"; usage; exit 1; fi
if [[ -z ${ADDITIONAL_LIT_ARGS+x} ]]; then ADDITIONAL_LIT_ARGS=""; fi
if [[ -z ${PREVIOUS_DYLIBS_DIR+x} ]]; then PREVIOUS_DYLIBS_DIR=""; fi

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

PREVIOUS_DYLIBS_URL="http://lab.llvm.org:8080/roots/libcxx-roots.tar.gz"
LLVM_TARBALL_URL="https://github.com/llvm-mirror/llvm/archive/master.tar.gz"


echo "@@@ Configuring CMake @@@"
mkdir -p "${LLVM_BUILD_DIR}"
(cd "${LLVM_BUILD_DIR}" &&
  xcrun cmake \
    -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Apple.cmake" \
    -GNinja \
    -DCMAKE_MAKE_PROGRAM="$(xcrun --find ninja)" \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}" \
    -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" \
    -DCMAKE_OSX_ARCHITECTURES="x86_64" \
    "${MONOREPO_ROOT}/llvm"
)
echo "@@@@@@"


echo "@@@ Building and installing libc++ and libc++abi @@@"
xcrun ninja -C "${LLVM_BUILD_DIR}" install-cxx install-cxxabi
echo "@@@@@@"


if [[ ${PREVIOUS_DYLIBS_DIR} == "" ]]; then
  echo "@@@ Downloading dylibs for older deployment targets @@@"
  PREVIOUS_DYLIBS_DIR="${TEMP_DIR}/libcxx-dylibs"
  mkdir "${PREVIOUS_DYLIBS_DIR}"
  curl "${PREVIOUS_DYLIBS_URL}" | tar -xz --strip-components=1 -C "${PREVIOUS_DYLIBS_DIR}"
  echo "@@@@@@"
fi

LIBCXX_ROOT_ON_DEPLOYMENT_TARGET="${PREVIOUS_DYLIBS_DIR}/macOS/libc++/${DEPLOYMENT_TARGET}"
LIBCXXABI_ROOT_ON_DEPLOYMENT_TARGET="${PREVIOUS_DYLIBS_DIR}/macOS/libc++abi/${DEPLOYMENT_TARGET}"

# TODO: We need to also run the tests for libc++abi.
echo "@@@ Running tests for libc++ @@@"
"${LLVM_BUILD_DIR}/bin/llvm-lit" -sv "${MONOREPO_ROOT}/libcxx/test" \
                                 --param=enable_experimental=false \
                                 --param=enable_debug_tests=false \
                                 ${ENABLE_FILESYSTEM} \
                                 --param=cxx_headers="${LLVM_INSTALL_DIR}/include/c++/v1" \
                                 --param=std="${STD}" \
                                 --param=target_triple="x86_64-apple-macosx${DEPLOYMENT_TARGET}" \
                                 --param=cxx_library_root="${LLVM_INSTALL_DIR}/lib" \
                                 --param=cxx_runtime_root="${LIBCXX_ROOT_ON_DEPLOYMENT_TARGET}" \
                                 --param=abi_library_path="${LIBCXXABI_ROOT_ON_DEPLOYMENT_TARGET}" \
                                 --param=use_system_cxx_lib="True" \
                                 ${ADDITIONAL_LIT_ARGS}
echo "@@@@@@"
