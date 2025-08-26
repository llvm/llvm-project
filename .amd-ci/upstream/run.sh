#!/usr/bin/env bash

#{Function to print command execution status by checking it's return value
execute_and_check()
{
        cmd="$1"
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

execute_and_check "mkdir -p ${WORKSPACE}/BUILD"
execute_and_check "cd  ${WORKSPACE}/BUILD"

MEMORY_KILOS=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEMORY_GIGS=$(( MEMORY_KILOS / 1000000 ))
MEMORY_COMPILE_LIMIT=$(( MEMORY_GIGS / 4 ))
MEMORY_LINK_LIMIT=$(( MEMORY_GIGS / 12 ))

set -x
cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_INSTALL_PREFIX="${WORKSPACE}/${JOB_NAME}-${BUILD_NUMBER}" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$LD_LIBRARY_PATH" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_LIT_ARGS=-v \
    -DCLANG_DEFAULT_LINKER=lld \
    -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra;flang" \
    -DLLVM_ENABLE_RUNTIMES="openmp;flang-rt;compiler-rt" \
    -S "${WORKSPACE}/llvm-project/llvm" \
    -B "${WORKSPACE}/BUILD" \
    -DPython3_EXECUTABLE:STRING=/proj/csse_jenkins2/swtools/apps/python/versions/3.8.12/bin/python \
    -DCMAKE_INSTALL_MESSAGE=LAZY \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DLLVM_PARALLEL_COMPILE_JOBS=${MEMORY_COMPILE_LIMIT} \
    -DLLVM_PARALLEL_LINK_JOBS=${MEMORY_LINK_LIMIT} \
    -DBUILD_SHARED_LIBS:STRING=ON \
    -DLIBOMP_OMP_VERSION=50 \
    -DLIBOMP_OMPT_SUPPORT=ON \
    -DLIBOMP_USE_DEBUGGER=ON \
    -DLIBOMP_CFLAGS="-O2" \
    -DLIBOMP_CPPFLAGS="-O2" \
    -DLIBOMPTARGET_OMPD_SUPPORT=ON \
    -DLIBOMP_OMPD_ENABLED=ON \
    -DLIBOMP_OMPD_SUPPORT=ON \
    -DFLANG_ENABLE_WERROR=ON \
    -DLLVM_TARGETS_TO_BUILD:STRING="X86;AMDGPU" \
    -DFLANG_RUNTIME_F128_MATH_LIB=libquadmath

ninja -C "${WORKSPACE}"/BUILD install
stat=$?
set +x
    if [ "${stat}" -eq 0 ]; then
        echo "********************************************************"
        echo "Execution of command : ninja -C ${WORKSPACE}/BUILD install - was successful" >> "${WORKSPACE}/command_executed.txt"
        echo "Execution of command : ninja -C ${WORKSPACE}/BUILD install - was successful"
        echo "********************************************************"
    else
        echo "########################################################"
        echo "Execution of command : ninja -C ${WORKSPACE}/BUILD install - was failed"
        echo "Please check BUILD_INSTALL.log file for details"
        echo "Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
        rm -v "${WORKSPACE}"/build_success.txt
        exit 1
        echo "########################################################"
    fi

execute_and_check "cd ${WORKSPACE}"
execute_and_check "tar -cJf ${JOB_NAME}-${BUILD_NUMBER}.tar.xz ${JOB_NAME}-${BUILD_NUMBER}"

set -x
ninja -C "${WORKSPACE}"/BUILD check-flang check-mlir
stat=$?
set +x
    if [ "${stat}" -eq 0 ]; then
        echo "********************************************************"
        echo "Execution of command : ninja -C ${WORKSPACE}/BUILD check-flang check-mlir - was successful" >> "${WORKSPACE}/command_executed.txt"
        echo "Execution of command : ninja -C ${WORKSPACE}/BUILD check-flang check-mlir - was successful"
        echo "********************************************************"
    else
        echo "########################################################"
        echo "Execution of command : ninja -C ${WORKSPACE}/BUILD check-flang check-mlir - was failed"
        echo "Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
        rm -v "${WORKSPACE}"/build_success.txt
        exit 1
        echo "########################################################"
    fi

execute_and_check "cd ${WORKSPACE}/llvm-project"
MAIN_COMMIT_HASH=$(git rev-parse HEAD)
echo "$MAIN_COMMIT_HASH" >> "${WORKSPACE}"/VERSION_AND_COMMIT_HASH_OF_BUILD_"${BUILD_NUMBER}".txt

if [ ! -f "${WORKSPACE}/build_success.txt" ]
then
    echo "Job failed - Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
    exit 1
fi
