#!/usr/bin/env bash

# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

# Define color variables
CYAN="\033[1;36m"
GREEN="\033[1;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
RESET="\033[0m"

# Function to echo messages in red
function echo_error() {
  echo -e "${RED}$1${RESET}"
}

function check_requirements() {
  if ! command -v cmake &> /dev/null; then
    echo_error "Error: cmake is not installed. Please install it before continuing."
    exit 1
  fi
  if ! command -v ninja &> /dev/null; then
    echo_error "Error: ninja is not installed. Please install it before continuing."
    exit 1
  fi
}

# Define paths
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0")
LLVM_PROJECT_ROOT="$GIT_ROOT"

# Function to get the last build type
function get_last_build_type() {
  local cache_file="${LLVM_PROJECT_ROOT}/build/CMakeCache.txt"
  if [[ -f "${cache_file}" ]]; then
    local build_type
    build_type=$(grep "CMAKE_BUILD_TYPE:STRING=" "${cache_file}" | cut -d'=' -f2)
    if [[ -n "${build_type}" ]]; then
      echo "${build_type}"
    else
      echo ""
    fi
  else
    echo ""
  fi
}

# Function to configure LLVM
function configure_llvm() {

  local build_type="${1:-Debug}" # Default to Debug if no argument is provided
  cd "${LLVM_PROJECT_ROOT}" || exit

  check_requirements
  cmake -G "Ninja" \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      -DLLVM_TARGETS_TO_BUILD="" \
      -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="Parasol" \
      -DCMAKE_BUILD_TYPE="${build_type}" \
      -DLLVM_ENABLE_ASSERTIONS=On \
      -DLLVM_ENABLE_ZLIB=On \
      -DLLVM_ENABLE_ZSTD=Off \
      -B build \
      -S llvm
}

# Function to build LLVM
function build_llvm() {
  check_requirements

  cd "${LLVM_PROJECT_ROOT}" || exit
  cmake --build build
}

# Update the show_help function to include the new build type options
function show_help() {
  echo -e "${GREEN}Usage:${RESET} $0 COMMAND [BUILD_TYPE]"
  echo -e "${CYAN}Commands:${RESET}"
  echo -e "  ${YELLOW}configure [Debug|Release|RelWithDebInfo|Last]${RESET}"
  echo -e "              Configure LLVM for compilation with an [optional]"
  echo -e "              specified build type. Defaults to Debug. Last selects"
  echo -e "              the last build type used, if it can be found."
  echo -e "  ${YELLOW}build${RESET}"
  echo -e "              Compile LLVM using the previously set configuration."
  echo -e "  ${YELLOW}all [Debug|Release|RelWithDebInfo|Last]${RESET}"
  echo -e "              Perform both configuration and compilation of LLVM"
  echo -e "              with an [optional] specified build type. Defaults to"
  echo -e "              Debug. Last selects the last build type used, if it"
  echo -e "              can be found."
  echo -e "  ${YELLOW}last-build-type${RESET}"
  echo -e "              Display the last build type used."
  echo -e "  ${YELLOW}clean${RESET}"
  echo -e "              Clean up the build directory."
  echo -e "  ${YELLOW}help${RESET}"
  echo -e "              Display this help message."
}

# Parse command-line options
case "$1" in
  clean)
    rm -rf "${LLVM_PROJECT_ROOT}/build"
    ;;
  configure)
    BUILD_TYPE="${2:-Debug}" # Default to Debug if no second argument is provided
    if [[ "${BUILD_TYPE}" == "Last" ]]; then
      BUILD_TYPE=$(get_last_build_type)
    fi
    if [[ "${BUILD_TYPE}" =~ ^(Debug|Release|RelWithDebInfo)$ ]]; then
      configure_llvm "${BUILD_TYPE}"
    else
      echo_error "Error: Invalid build type '${BUILD_TYPE}'. Valid options are Debug, Release, RelWithDebInfo."
      exit 1
    fi
    ;;
  build)
    build_llvm
    ;;
  all)
    BUILD_TYPE="${2:-Debug}" # Default to Debug if no second argument is provided
    if [[ "${BUILD_TYPE}" == "Last" ]]; then
      BUILD_TYPE=$(get_last_build_type)
    fi
    if [[ "${BUILD_TYPE}" =~ ^(Debug|Release|RelWithDebInfo)$ ]]; then
      configure_llvm "${BUILD_TYPE}"
      build_llvm
    else
      echo_error "Error: Invalid build type '${BUILD_TYPE}'. Valid options are Debug, Release, RelWithDebInfo."
      exit 1
    fi
    ;;
  last-build-type)
    get_last_build_type
    ;;
  help)
    show_help
    exit 0
    ;;
  *)
    echo "Error: Invalid option '$1'"
    show_help
    exit 1
    ;;
esac
