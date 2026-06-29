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

# Directory (relative to the workspace) holding the SPEC dependency library
# tars (jemalloc, libm, amdalloc, ssmalloc) pre-downloaded on the Jenkins agent
# by the Jenkinsfile 'download-spec-libs' stage using the orequest pw_file
# method. Bind-mounted into the container at ${bind_point}/spec_libs.
SPEC_LIBS_DIR_DEFAULT='spec_libs'

# Artifactory password file (orequest). /proj is bind-mounted read-only into the
# container, so this is readable here too and used only as a fallback when the
# pre-downloaded tars are not present.
ARTIFACTORY_PW_FILE_DEFAULT='/proj/csse_jenkins2/swtools/.safe-secrets/orequest-AMD-Artifactory'

# Download a single artifactory tar using the orequest pw_file method (mirrors
# gcc/.amd-ci/upstream/Jenkinsfile). Used only as a fallback; the primary path
# is the pre-downloaded tars under ${SPEC_LIBS_DIR}.
fetch_artifactory_tar() {
  local url="$1" out="$2"
  if test -z "${url}"
  then
    return 1
  fi
  if test -f "${url}"
  then
    cp -v "${url}" "${out}"
    return 0
  fi
  local pw_file="${ARTIFACTORY_PW_FILE:-${ARTIFACTORY_PW_FILE_DEFAULT}}"
  if test -f "${pw_file}"
  then
    local pw
    pw=$(head -1 "${pw_file}" | tr -d '\n\r')
    wget --user=orequest --password="${pw}" --auth-no-challenge -q "${url}" -O "${out}"
  else
    echo "WARNING: pw_file ${pw_file} not found; attempting anonymous download of ${url}"
    wget -q "${url}" -O "${out}" || curl -fSL --connect-timeout 30 --retry 3 -o "${out}" "${url}"
  fi
}

# Write clang/clang++/flang config files into the compiler's bin/ so that every
# driver invocation automatically adds <compiler>/lib to the linker search path
# and to the runtime rpath. This is required because on Linux the SPEC harness
# (aocc-devops spec_par.sh do_runcpu) does NOT pass --define <id>_dir=... to
# runcpu, so the cfg's %{..._dir} macros are undefined and it invokes the
# compiler as a bare "clang" on PATH with empty -L paths. Without this, the
# bundled libs in lib/ are invisible to ld.lld and the link fails with
# "ld.lld: error: unable to find library -lamdalloc". <CFGDIR> keeps the path
# relocatable (expands to the directory containing the config file, i.e. bin/).
write_compiler_link_config() {
  local comp_root="$1"
  local cfg_body=$'-L<CFGDIR>/../lib\n-Wl,-rpath,<CFGDIR>/../lib\n'
  local drv
  for drv in clang clang++ flang
  do
    printf '%s' "${cfg_body}" > "${comp_root}/bin/${drv}.cfg"
  done
  echo "Wrote driver config files (-L/-rpath <compiler>/lib) to ${comp_root}/bin/:"
  ls -l "${comp_root}/bin"/{clang,clang++,flang}.cfg 2>/dev/null || true
}

# Bundle the SPEC dependency libraries (jemalloc, libm, amdalloc, ssmalloc) into
# the freshly built compiler package. SPEC cfgs such as
# cpu2017_amd_linux_znver5_llvm_*_bestmined.cfg link -lamdalloc / -lamdalloc-ext,
# so the libs must live in the compiler's lib/ AND that dir must be on the
# linker search path. We place the libs in lib/ here and, via
# write_compiler_link_config, add a driver config that puts lib/ on the -L and
# rpath search paths (the cfg itself cannot, since <id>_dir is undefined on
# Linux -- see write_compiler_link_config).
bundle_spec_libs_into_cut() {
  local libs_dir="${SPEC_LIBS_DIR:-${SPEC_LIBS_DIR_DEFAULT}}"

  local work
  work=$(mktemp -d)
  # shellcheck disable=2064
  trap "rm -rf '${work}'" RETURN

  # Gather the dependency tars. Prefer the ones pre-downloaded on the agent
  # (pw_file method, in ${libs_dir}); fall back to fetching here if absent.
  local -a tars=()
  if test -d "${libs_dir}"
  then
    local t
    for t in "${libs_dir}"/*.tar "${libs_dir}"/*.tar.xz "${libs_dir}"/*.tar.gz
    do
      test -f "${t}" && tars+=("$(readlink -f "${t}")")
    done
  fi
  if test "${#tars[@]}" -eq 0
  then
    echo "No pre-downloaded SPEC library tars in ${libs_dir}; trying pw_file fallback for amdalloc."
    local fb="${work}/amdalloc.tar"
    if fetch_artifactory_tar "${AMDALLOC_URL:-https://atlartifactory.amd.com:8443/artifactory/SW-AOCC-REL-LOCAL/amdalloc/1.0.4/amdalloc-1.0.4.tar}" "${fb}"
    then
      tars+=("${fb}")
    fi
  fi
  if test "${#tars[@]}" -eq 0
  then
    echo "ERROR: no SPEC dependency libraries available to bundle into the compiler."
    return 1
  fi

  # Locate the compiler tar produced by aocc_build.py (inside the *_build dir).
  local cut_tar
  cut_tar=$(find . -path '*_build/*' -name "*-${BUILD_NUMBER:-}*.tar.xz" 2>/dev/null | sort -V | tail -1)
  if test -z "${cut_tar}"
  then
    cut_tar=$(find . -maxdepth 1 -name "*-${BUILD_NUMBER:-}*.tar.xz" 2>/dev/null | sort -V | tail -1)
  fi
  if test -z "${cut_tar}"
  then
    echo "ERROR: no compiler tar matching *-${BUILD_NUMBER:-}*.tar.xz found; cannot bundle libraries."
    return 1
  fi
  local cut_tar_abs
  cut_tar_abs=$(readlink -f "${cut_tar}")
  echo "Bundling SPEC libraries into compiler package: ${cut_tar_abs}"

  # Unpack the compiler tar so we can inject libs into its lib/.
  local stage="${work}/cut"
  mkdir -p "${stage}"
  ( cd "${stage}" && tar -xf "${cut_tar_abs}" )
  local comp_root
  comp_root=$(find "${stage}" -mindepth 1 -maxdepth 1 -type d | head -1)
  if test -z "${comp_root}" || test ! -d "${comp_root}/lib"
  then
    echo "ERROR: unpacked compiler tree has no lib/ dir under ${stage}."
    return 1
  fi

  # Extract each dependency tar and copy its 64-bit libs into the compiler lib/.
  # cp -a preserves the lib*.so -> lib*.so.N symlinks. 32-bit (lib32/) libs are
  # intentionally skipped for this x86-64 compiler.
  local t ed
  for t in "${tars[@]}"
  do
    ed="${work}/$(basename "${t}").x"
    mkdir -p "${ed}"
    if ! tar -xf "${t}" -C "${ed}" --strip-components=1
    then
      echo "WARNING: failed to extract ${t}; skipping."
      continue
    fi
    if test -d "${ed}/lib"
    then
      cp -a "${ed}/lib/." "${comp_root}/lib/" 2>/dev/null || true
    fi
    # Some packages keep the libraries at the top level rather than under lib/.
    find "${ed}" -maxdepth 1 -type f \( -name '*.so' -o -name '*.so.*' -o -name '*.a' \) \
      -exec cp -a {} "${comp_root}/lib/" \; 2>/dev/null || true
    echo "Bundled libraries from $(basename "${t}")."
  done

  if ls "${comp_root}/lib"/libamdalloc.so* >/dev/null 2>&1
  then
    echo "amdalloc libs now present in compiler lib/:"
    ls -l "${comp_root}/lib"/libamdalloc* || true
  else
    echo "WARNING: libamdalloc.so* not found in compiler lib/ after bundling; SPEC -lamdalloc may still fail."
  fi

  # Make <compiler>/lib discoverable by the linker/loader for every driver
  # invocation (the SPEC cfg cannot add -L on Linux; see the function comment).
  write_compiler_link_config "${comp_root}"

  # Repack to the same path so downstream consumers see no structural change.
  local comp_dir
  comp_dir=$(basename "${comp_root}")
  local repacked="${work}/repacked.tar.xz"
  ( cd "${stage}" && env XZ_OPT="-T0" tar -cJf "${repacked}" "${comp_dir}" )
  mv -f "${repacked}" "${cut_tar_abs}"
  echo "Repacked compiler package with SPEC libraries bundled: ${cut_tar_abs}"
}

execute_build
bundle_spec_libs_into_cut || { echo "SPEC library bundling failed."; exit 1; }
