#!/usr/bin/env bash

if test -d llvm-project/aocc-essentials
then
  ess_dir=llvm-project/aocc-essentials
else
  ess_dir=aocc-essentials
fi

_ver_file=llvm-project/cmake/Modules/LLVMVersion.cmake
LLVM_VERSION_MAJOR=$(grep -w "set(LLVM_VERSION_MAJOR" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')
LLVM_VERSION_MINOR=$(grep -w "set(LLVM_VERSION_MINOR" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')
LLVM_VERSION_PATCH=$(grep -w "set(LLVM_VERSION_PATCH" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')

if test "${LLVM_VERSION_MAJOR}" -gt 21
then
  export AOCC_GCC_VER="11.4.0"
fi

mod_cmd=". ${ess_dir}/build_essentials/linux/aocc_env.sh"
echo "${mod_cmd}"
${mod_cmd}

CLANG_VERSION=$LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR.$LLVM_VERSION_PATCH
MAIN_COMMIT_HASH=$(git -C llvm-project rev-parse HEAD)
{
  echo "CLANG_VERSION value is - $CLANG_VERSION"
  echo "$MAIN_COMMIT_HASH"
} > VERSION_AND_COMMIT_HASH_OF_BUILD_"${BUILD_NUMBER}".txt

dual_flang_build="${AOCC_DUAL_FLANG_BUILD:-false}"

if test "${dual_flang_build}" = "true"
then
  fc=LLVM_then_Classic
else
  fc=LLVM
fi

if test -n "${CHANGE_TARGET}"
then
  actual_target="${CHANGE_TARGET}"
else
  actual_target="${BRANCH_NAME}"
fi

build_args=(
  --build_mode=All
  --fc="${fc}"
  --verbose
)
if test -n "${BUILD_TYPE:-}"
then
  build_args+=(--aocc_build_type="${BUILD_TYPE}")
  if test "${BUILD_TYPE}" = "RELEASE" -o "${BUILD_TYPE}" = "STAGING"
  then
    build_args+=(--branch="${BRANCH_NAME}")
  fi
fi
if test "${actual_target}" != "amd-staging" -a "${actual_target}" != "upstream-main"
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

echo python3 -u "${ess_dir}/build_essentials/aocc_build.py" "${build_args[@]}"
python3 -u "${ess_dir}/build_essentials/aocc_build.py" "${build_args[@]}"
