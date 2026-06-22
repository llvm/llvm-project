#!/usr/bin/env bash

if test -d llvm-project/aocc-essentials
then
  ess_dir=llvm-project/aocc-essentials
else
  ess_dir=aocc-essentials
fi

LLVM_GIT_DIR=llvm-project

mod_cmd=". ${ess_dir}/build_essentials/linux/aocc_env.sh"
echo "${mod_cmd}"
${mod_cmd}

# Parse the LLVM version from the in-tree cmake module.
get_llvm_version() {
  local _ver_file=llvm-project/cmake/Modules/LLVMVersion.cmake
  LLVM_VERSION_MAJOR=$(grep -w "set(LLVM_VERSION_MAJOR" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')
  LLVM_VERSION_MINOR=$(grep -w "set(LLVM_VERSION_MINOR" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')
  LLVM_VERSION_PATCH=$(grep -w "set(LLVM_VERSION_PATCH" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')
}

# Build the argv array passed to aocc_build.py. Single source of truth.
construct_build_args() {
  local fc=$1
  build_args=(
    --build_mode=All
    --fc="${fc}"
    --verbose
  )
  if test -n "${BUILD_TYPE:-}"
  then
    build_args+=(--aocc_build_type="${BUILD_TYPE}")
    if test "${BUILD_TYPE}" = "RELEASE" || test "${BUILD_TYPE}" = "STAGING"
    then
      build_args+=(--branch="${BRANCH_NAME}")
    fi
  fi
  if test "${LLVM_VERSION_MAJOR}" -le 21
  then
    build_args+=(--enable_how=PROJECTS)
  fi
  if test -n "${REL_MODE:-}"
  then
    build_args+=(--release_mode="${REL_MODE}")
  fi
  if test -n "${STAGE_BUILD_NUM:-}"
  then
    build_args+=(--stage_build_num="${STAGE_BUILD_NUM}")
  fi
  if test -n "${VERSION_VAL:-}"
  then
    build_args+=(--version_val="${VERSION_VAL}")
  fi
}

# Run a complete LLVM build.
execute_build() {
  get_llvm_version

  local clang_version=$LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR.$LLVM_VERSION_PATCH
  local main_commit_hash
  main_commit_hash=$(git -C "${LLVM_GIT_DIR}" rev-parse HEAD)

  {
    echo "CLANG_VERSION value is - ${clang_version}"
    echo "${main_commit_hash}"
  } > VERSION_AND_COMMIT_HASH_OF_BUILD_"${BUILD_NUMBER}".txt

  local dual_flang_build="${AOCC_DUAL_FLANG_BUILD:-false}"
  local fc=LLVM
  if test "${dual_flang_build}" = "true"
  then
    fc=LLVM_then_Classic
  fi

  construct_build_args "${fc}"

  echo python3 -u "${ess_dir}/build_essentials/aocc_build.py" "${build_args[@]}"
  python3 -u "${ess_dir}/build_essentials/aocc_build.py" "${build_args[@]}"
}

# Apply test suppressions via upstream lit's --unsupported mechanism.
_suppress_file="${LLVM_GIT_DIR}/.amd-ci/upstream/lit_suppressions.txt"
if test -f "${_suppress_file}"
then
  LIT_UNSUPPORTED="$(sed 's/^[[:space:]]*//' "${_suppress_file}" | grep -v '^#' | grep -v '^$' | paste -sd ';')"
  if test -n "${LIT_UNSUPPORTED}"
  then
    export LIT_UNSUPPORTED
    echo "LIT_UNSUPPORTED=${LIT_UNSUPPORTED}"
  fi
fi

execute_build
