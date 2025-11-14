#!/usr/bin/env bash

#{Function to print command execution status by checking it's return value
execute_and_check()
{
        cmd="$1"
        echo "********************************************************"
        echo "Executing command : $cmd - please wait ..."
        echo "********************************************************"
        $cmd
        retVal=$?

        if [ "$retVal" -eq 0 ]; then
                echo "********************************************************"
                echo "Execution of command : $cmd - was successful" >> "${WORKSPACE}/command_executed.txt"
                echo "Execution of command : $cmd - was successful"
                echo "********************************************************"
        else
                echo "########################################################"
                echo "Execution of command : $cmd - was failed"
                echo "Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
                rm -v "${WORKSPACE}/build_success.txt"
                exit 1
                echo "########################################################"
        fi
}
#}

WORKSPACE=$PWD
touch "${WORKSPACE}"/build_success.txt

_ver_file=llvm-project/cmake/Modules/LLVMVersion.cmake
LLVM_VERSION_MAJOR=$(grep -w "set(LLVM_VERSION_MAJOR" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')
LLVM_VERSION_MINOR=$(grep -w "set(LLVM_VERSION_MINOR" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')
LLVM_VERSION_PATCH=$(grep -w "set(LLVM_VERSION_PATCH" "${_ver_file}" | awk -F' ' '{print $NF}' | tr -d ')')
CLANG_VERSION=$LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR.$LLVM_VERSION_PATCH
echo "CLANG_VERSION value is - $CLANG_VERSION" > "${WORKSPACE}"/VERSION_AND_COMMIT_HASH_OF_BUILD_"${BUILD_NUMBER}".txt

if test -d aocc-essentials
then
  ess_dir=aocc-essentials
else
  ess_dir=llvm-project/aocc-essentials
fi

mod_cmd1=". ${ess_dir}/build_essentials/linux/aocc_env.sh"
echo "${mod_cmd1}"
${mod_cmd1}
mod_cmd2=". /proj/csse_jenkins2/swtools/c/rhel7-gcc8.3.1/environment-modules-5.4.0-j6k56q3i/init/bash"
echo "${mod_cmd2}"
${mod_cmd2}
mod_cmd3="module load gcc/11.4.0"
echo "${mod_cmd3}"
${mod_cmd3}

MEMORY_KILOS=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEMORY_GIGS=$(( MEMORY_KILOS / 1000000 ))
MEMORY_COMPILE_LIMIT=$(( MEMORY_GIGS / 4 ))
MEMORY_LINK_LIMIT=$(( MEMORY_GIGS / 12 ))
CURRENT_DATE=$(date +"%Y_%m_%d")

dual_flang_build="${AOCC_DUAL_FLANG_BUILD:-false}"

next_build_dir="${WORKSPACE}/BUILD"
inst_dir="${JOB_NAME}-${BUILD_NUMBER}"
cmake_base_args=(
  -G Ninja
  -DCMAKE_BUILD_TYPE:STRING=Release
  -DCMAKE_INSTALL_PREFIX="${WORKSPACE}/${inst_dir}"
  -DLLVM_VERSION_SUFFIX="pre"
  -DAOCC_REVISION="AOCC_6.0.0-Build#${BUILD_NUMBER} ${CURRENT_DATE}"
  -DCMAKE_CXX_STANDARD=17
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$LD_LIBRARY_PATH"
  -DLLVM_ENABLE_ASSERTIONS=ON
  -DLLVM_LIT_ARGS=-v
  -DCLANG_DEFAULT_LINKER=lld
  -S "${WORKSPACE}/llvm-project/llvm"
  -DPython3_EXECUTABLE:STRING=/proj/csse_jenkins2/swtools/apps/python/versions/3.8.12/bin/python
  -DCMAKE_INSTALL_MESSAGE=LAZY
  -DCMAKE_C_COMPILER=gcc
  -DCMAKE_CXX_COMPILER=g++
  -DLLVM_PARALLEL_COMPILE_JOBS="${MEMORY_COMPILE_LIMIT}"
  -DLLVM_PARALLEL_LINK_JOBS="${MEMORY_LINK_LIMIT}"
  -DBUILD_SHARED_LIBS:STRING=ON
  -DLIBOMP_OMP_VERSION=50
  -DLIBOMP_OMPT_SUPPORT=ON
  -DLIBOMP_USE_DEBUGGER=ON
  -DLIBOMP_CFLAGS="-O2"
  -DLIBOMP_CPPFLAGS="-O2"
  -DLIBOMPTARGET_OMPD_SUPPORT=ON
  -DLIBOMP_OMPD_ENABLED=ON
  -DLIBOMP_OMPD_SUPPORT=ON
  -DFLANG_ENABLE_WERROR=ON
  -DLLVM_TARGETS_TO_BUILD:STRING="X86;AMDGPU"
  -DFLANG_RUNTIME_F128_MATH_LIB=libquadmath
)

if test -d "${WORKSPACE}/llvm-project/aocc-essentials"
then
  echo "This looks like an AOCC branch."
  aocc_branch=1
else
  aocc_branch=0
fi

if test ${aocc_branch} -ne 0
then
  echo "AOCC branch; adding extra required CMake switches."
  cmake_base_args+=(
    -DCMAKE_CXX_FLAGS="-Wno-error=pedantic -pthread"
    -DLLVM_LIT_ARGS="--xunit-xml-output=testresults.xunit.xml -v --timeout=600 --param blacklist=${WORKSPACE}/llvm-project/prj-essentials/devo/lit.blacklist.cfg"
  )
fi

if test "${dual_flang_build}" = "true"
then
  amd_prefix="aocc-"
fi

amd_tool_links() {
  ln -sf clang++ "${amd_prefix}clang++"
  ln -sf clang "${amd_prefix}clang"
  ln -sf clang-cpp "${amd_prefix}clang-cpp"
  ln -sf clang-cl "${amd_prefix}clang-cl"
  ln -sf lld "${amd_prefix}lld"
  ln -sf "${amd_prefix}flang" flang
}

execute_and_check "mkdir -p ${next_build_dir}"
execute_and_check "cd ${next_build_dir}"

cmake_args=("${cmake_base_args[@]}")

if test ${aocc_branch} -ne 0
then
  cmake_args+=(
    -DLLVM_ENABLE_CLASSIC_FLANG=OFF
  )
fi

next_vendor_string="AMD AOCC"
cmake_args+=(
  -B "${next_build_dir}"
  -DCLANG_VENDOR="${next_vendor_string}"
  -DFLANG_VENDOR="${next_vendor_string}"
  -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra;flang"
  -DLLVM_ENABLE_RUNTIMES="openmp;flang-rt;compiler-rt"
)

set -x
cmake "${cmake_args[@]}"
ninja -C "${next_build_dir}" install
stat=$?
set +x
if [ "${stat}" -eq 0 ]; then
  echo "********************************************************"
  echo "Execution of command : ninja -C ${next_build_dir} install - was successful" >> "${WORKSPACE}/command_executed.txt"
  echo "Execution of command : ninja -C ${next_build_dir} install - was successful"
  echo "********************************************************"
else
  echo "########################################################"
  echo "Execution of command : ninja -C ${next_build_dir} install - was failed"
  echo "Please check BUILD_INSTALL.log file for details"
  echo "Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
  rm -v "${WORKSPACE}"/build_success.txt
  exit 1
  echo "########################################################"
fi

execute_and_check "cd ${WORKSPACE}"

if test "${dual_flang_build}" = "true"
then
  (
    cd "${next_build_dir}/bin" || exit 1
    amd_tool_links
  ) || exit 1
  (
    cd "${inst_dir}/bin" || exit 1
    amd_tool_links
  ) || exit 1
fi

if test "${dual_flang_build}" != "true"
then
  execute_and_check "tar -cJf ${inst_dir}.tar.xz ${inst_dir}"
fi

set -x
ninja -C "${next_build_dir}" check-flang check-mlir
stat=$?
set +x
if [ "${stat}" -eq 0 ]; then
  echo "********************************************************"
  echo "Execution of command : ninja -C ${next_build_dir} check-flang check-mlir - was successful" >> "${WORKSPACE}/command_executed.txt"
  echo "Execution of command : ninja -C ${next_build_dir} check-flang check-mlir - was successful"
  echo "********************************************************"
else
  echo "########################################################"
  echo "Execution of command : ninja -C ${next_build_dir} check-flang check-mlir - was failed"
  echo "Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
  rm -v "${WORKSPACE}"/build_success.txt
  exit 1
  echo "########################################################"
fi

if test "${dual_flang_build}" = "true"
then
  next_build_dir="${WORKSPACE}/BUILD_c"
  execute_and_check "mkdir -p ${next_build_dir}"
  execute_and_check "cd ${next_build_dir}"

  cmake_args=("${cmake_base_args[@]}")
  next_vendor_string="AMD"
  cmake_args+=(
    -DLLVM_ENABLE_CLASSIC_FLANG=ON
    -B "${next_build_dir}"
    -DCLANG_VENDOR="${next_vendor_string}"
    -DFLANG_VENDOR="${next_vendor_string}"
    -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra"
    -DLLVM_ENABLE_RUNTIMES="openmp;compiler-rt"
  )

  set -x
  cmake "${cmake_args[@]}"
  ninja -C "${next_build_dir}" install
  stat=$?
  set +x

  if test "${stat}" -ne 0
  then
    rm -v "${WORKSPACE}"/build_success.txt
    exit 1
  fi

  set -x
  ninja -C "${next_build_dir}" check-clang # Code smell: check-all fails in check of Python bindings.
  stat=$?
  set +x

  if test "${stat}" -ne 0
  then
    rm -v "${WORKSPACE}"/build_success.txt
    exit 1
  fi

  pgmath64_build_dir="${next_build_dir}/libpgmath64"
  next_build_dir="${pgmath64_build_dir}"
  mkdir -p "${next_build_dir}/include"
  cp /usr/lib/gcc/x86_64-redhat-linux/8/include/quadmath.h "${next_build_dir}/include"

  cmake_args=(
    -G Ninja
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_INSTALL_PREFIX="${WORKSPACE}/${inst_dir}"
    -S "${WORKSPACE}/llvm-project/classic-flang/runtime/libpgmath"
    -B "${next_build_dir}"
    -DPython3_EXECUTABLE:STRING=/proj/csse_jenkins2/swtools/apps/python/versions/3.8.12/bin/python
    -DCMAKE_INSTALL_MESSAGE=LAZY
    -DCMAKE_C_COMPILER="${WORKSPACE}/${inst_dir}/bin/clang"
    -DCMAKE_CXX_COMPILER="${WORKSPACE}/${inst_dir}/bin/clang++"
    -DLLVM_PARALLEL_COMPILE_JOBS="${MEMORY_COMPILE_LIMIT}"
    -DLLVM_PARALLEL_LINK_JOBS="${MEMORY_LINK_LIMIT}"
    -DCMAKE_C_FLAGS="-I${next_build_dir}/include"
    -DCMAKE_CXX_FLAGS="-I${next_build_dir}/include"
  )

  set -x
  cmake "${cmake_args[@]}"
  ninja -C "${next_build_dir}" install
  stat=$?
  set +x

  if test "${stat}" -ne 0
  then
    rm -v "${WORKSPACE}"/build_success.txt
    exit 1
  fi

  # ToDo libpgmath32?

  next_build_dir="${next_build_dir}/classic-flang"

  cmake_args=(
    -G Ninja
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_INSTALL_PREFIX="${WORKSPACE}/${inst_dir}"
    -S "${WORKSPACE}/llvm-project/classic-flang"
    -B "${next_build_dir}"
    -DPYTHON_EXECUTABLE:STRING=/proj/csse_jenkins2/swtools/apps/python/versions/3.8.12/bin/python
    -DCMAKE_INSTALL_MESSAGE=LAZY
    -DCMAKE_C_COMPILER="${WORKSPACE}/${inst_dir}/bin/clang"
    -DCMAKE_CXX_COMPILER="${WORKSPACE}/${inst_dir}/bin/clang++"
    -DCMAKE_Fortran_COMPILER="${WORKSPACE}/${inst_dir}/bin/flang"
    -DCMAKE_Fortran_COMPILER_ID=Flang
    -DLLVM_PARALLEL_COMPILE_JOBS="${MEMORY_COMPILE_LIMIT}"
    -DLLVM_PARALLEL_LINK_JOBS="${MEMORY_LINK_LIMIT}"
    -DCMAKE_C_FLAGS="-I${pgmath64_build_dir}/include"
    -DCMAKE_CXX_FLAGS="-I${pgmath64_build_dir}/include"
    -DFLANG_INCLUDE_DOCS:BOOL=OFF
    -DFLANG_INCLUDE_TESTS:BOOL=OFF
    -DFLANG_OPENMP_GPU_NVIDIA=on
    -DFLANG_OPENMP_GPU_AMD=on
    -DLLVM_CONFIG="${WORKSPACE}/${inst_dir}/bin/llvm-config"
  )

  set -x
  cmake "${cmake_args[@]}"
  ninja -C "${next_build_dir}" install
  stat=$?
  set +x

  if test "${stat}" -ne 0
  then
    rm -v "${WORKSPACE}"/build_success.txt
    exit 1
  fi

  execute_and_check "cd ${WORKSPACE}"

  execute_and_check "tar -cJf ${inst_dir}.tar.xz ${inst_dir}"
fi

execute_and_check "cd ${WORKSPACE}/llvm-project"
MAIN_COMMIT_HASH=$(git rev-parse HEAD)
echo "$MAIN_COMMIT_HASH" >> "${WORKSPACE}"/VERSION_AND_COMMIT_HASH_OF_BUILD_"${BUILD_NUMBER}".txt

if [ ! -f "${WORKSPACE}/build_success.txt" ]
then
    echo "Job failed - Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
    exit 1
fi
