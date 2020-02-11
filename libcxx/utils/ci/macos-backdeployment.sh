#!/usr/bin/env bash

set -ue

function usage() {
  cat <<EOM
$(basename ${0}) [-h|--help] --monorepo-root <MONOREPO-ROOT> --std <STD> --arch <ARCHITECTURE> --deployment-target <TARGET> --sdk-version <SDK-VERSION> [--lit-args <ARGS...>]

This script is used to continually test the back-deployment use case of libc++ and libc++abi on MacOS.

  --monorepo-root     Full path to the root of the LLVM monorepo. Both libc++ and libc++abi headers from the monorepo are used.
  --std               Version of the C++ Standard to run the tests under (c++03, c++11, etc..).
  --arch              Architecture to build the tests for (32, 64).
  --deployment-target The deployment target to run the tests for. This should be a version number of MacOS (e.g. 10.12). All MacOS versions until and including 10.9 are supported.
  --sdk-version       The version of the SDK to test with. This should be a version number of MacOS (e.g. 10.12). We'll link against the libc++ dylib in that SDK, but we'll run against the one on the given deployment target. The SDK version must be no older than the deployment target.
  [--lit-args]        Additional arguments to pass to lit (optional). If there are multiple arguments, quote them to pass them as a single argument to this script.
  [--no-cleanup]      Do not cleanup the temporary directory that was used for testing at the end. This can be useful to debug failures. Make sure to clean up manually after.
  [-h, --help]        Print this help.
EOM
}

function version-less-equal() {
    [ "$1" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

function version-less() {
    [ "$1" = "$2" ] && return 1 || version-less-equal $1 $2
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
    --arch)
    ARCH="${2}"
    shift; shift
    ;;
    --deployment-target)
    DEPLOYMENT_TARGET="${2}"
    shift; shift
    ;;
    --sdk-version)
    MACOS_SDK_VERSION="${2}"
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
if [[ -z ${DEPLOYMENT_TARGET+x} ]]; then echo "--deployment-target is a required parameter"; usage; exit 1; fi
if [[ -z ${MACOS_SDK_VERSION+x} ]]; then echo "--sdk-version is a required parameter"; usage; exit 1; fi
if [[ -z ${ADDITIONAL_LIT_ARGS+x} ]]; then ADDITIONAL_LIT_ARGS=""; fi

if version-less "${MACOS_SDK_VERSION}" "${DEPLOYMENT_TARGET}"; then
  echo "SDK version ${MACOS_SDK_VERSION} shouldn't be older than the deployment target (${DEPLOYMENT_TARGET})"
  usage
  exit 1
fi

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


echo "@@@ Configuring architecture-related stuff @@@"
if [[ "${ARCH}" == "64" ]]; then CMAKE_ARCH_STRING="x86_64"; else CMAKE_ARCH_STRING="i386"; fi
if [[ "${ARCH}" == "64" ]]; then LIT_ARCH_STRING="";         else LIT_ARCH_STRING="--param=enable_32bit=true"; fi
echo "@@@@@@"


echo "@@@ Configuring CMake @@@"
mkdir -p "${LLVM_BUILD_DIR}"
(cd "${LLVM_BUILD_DIR}" &&
  xcrun cmake \
    -C "${MONOREPO_ROOT}/libcxx/cmake/caches/Apple.cmake" \
    -GNinja \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}" \
    -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" \
    -DCMAKE_OSX_ARCHITECTURES="${CMAKE_ARCH_STRING}" \
    "${MONOREPO_ROOT}/llvm"
)
echo "@@@@@@"


echo "@@@ Installing the latest libc++ headers @@@"
ninja -C "${LLVM_BUILD_DIR}" install-cxx-headers
echo "@@@@@@"


echo "@@@ Downloading dylibs for older deployment targets @@@"
# TODO: We should also link against the libc++abi.dylib that was shipped in the SDK
PREVIOUS_DYLIBS_DIR="${TEMP_DIR}/libcxx-dylibs"
mkdir "${PREVIOUS_DYLIBS_DIR}"
curl "${PREVIOUS_DYLIBS_URL}" | tar -xz --strip-components=1 -C "${PREVIOUS_DYLIBS_DIR}"
LIBCXX_ON_DEPLOYMENT_TARGET="${PREVIOUS_DYLIBS_DIR}/macOS/${DEPLOYMENT_TARGET}/libc++.dylib"
LIBCXXABI_ON_DEPLOYMENT_TARGET="${PREVIOUS_DYLIBS_DIR}/macOS/${DEPLOYMENT_TARGET}/libc++abi.dylib"
LIBCXX_IN_SDK="${PREVIOUS_DYLIBS_DIR}/macOS/${MACOS_SDK_VERSION}/libc++.dylib"
echo "@@@@@@"


# TODO: We need to also run the tests for libc++abi.
echo "@@@ Running tests for libc++ @@@"
"${LLVM_BUILD_DIR}/bin/llvm-lit" -sv "${MONOREPO_ROOT}/libcxx/test" \
                                 --param=enable_experimental=false \
                                 ${LIT_ARCH_STRING} \
                                 --param=cxx_headers="${LLVM_INSTALL_DIR}/include/c++/v1" \
                                 --param=std="${STD}" \
                                 --param=platform="macosx${DEPLOYMENT_TARGET}" \
                                 --param=cxx_runtime_root="$(dirname "${LIBCXX_ON_DEPLOYMENT_TARGET}")" \
                                 --param=abi_library_path="$(dirname "${LIBCXXABI_ON_DEPLOYMENT_TARGET}")" \
                                 --param=use_system_cxx_lib="$(dirname "${LIBCXX_IN_SDK}")" \
                                 ${ADDITIONAL_LIT_ARGS}
echo "@@@@@@"
