#!/bin/bash -x

BRANCH=
BUILD_NUMBER=
PR_NUMBER=
SRC_DIR="../llvm.src"
DST_DIR="."

# $1 exit code
# $2 error message
exit_if_err()
{
    if [ $1 -ne 0  ]; then
        echo "Error: $2"
        exit $1
    fi
}

unset OPTIND
while getopts ":b:r:n:s:d" option; do
    case $option in
        b) BRANCH=$OPTARG ;;
        n) BUILD_NUMBER=$OPTARG ;;
        s) SRC_DIR=$OPTARG ;;
        d) DST_DIR=$OPTARG ;;
        r) PR_NUMBER=$OPTARG ;;
    esac
done && shift $(($OPTIND - 1))

# we're in llvm.obj dir
BUILD_DIR=${PWD}

# Get changed build script files if it is PR
if [ -n "${PR_NUMBER}" ];then
    cd ${SRC_DIR}
    git fetch origin sycl
    exit_if_err $? "fail to get the latest changes in sycl branch"
    git fetch -t origin refs/pull/${PR_NUMBER}/merge
    exit_if_err $? "fail to get tags"
    git checkout -B refs/pull/${PR_NUMBER}/merge
    exit_if_err $? "fail to create branch for specific tag"
    base_commit=`git merge-base origin/sycl refs/pull/${PR_NUMBER}/merge`
    exit_if_err $? "fail to get base commit"

    BUILD_SCRIPT=`git --no-pager diff ${base_commit} refs/pull/${PR_NUMBER}/merge --name-only buildbot`
    cd -
fi

## Clean up build directory if build scripts has changed
cd ${DST_DIR}
if [  -n "$BUILD_SCRIPT" ]; then
   rm -rf *
fi
cd -

## GET dependencies
if [ ! -d "OpenCL-Headers" ]; then
    git clone https://github.com/KhronosGroup/OpenCL-Headers OpenCL-Headers
    exit_if_err $? "failed to clone OpenCL-Headers"
else
    cd OpenCL-Headers
    git pull --ff --ff-only origin
    exit_if_err $? "failed to update OpenCL-Headers"
fi

OPENCL_HEADERS=${BUILD_DIR}/OpenCL-Headers

cd ${BUILD_DIR}
if [ ! -d "OpenCL-ICD-Loader" ]; then
    git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader OpenCL-ICD-Loader
    exit_if_err $? "failed to clone OpenCL-ICD-Loader"
else
    cd OpenCL-ICD-Loader
    git pull --ff --ff-only origin
    exit_if_err $? "failed to update OpenCL-ICD-Loader"
fi

cd ${BUILD_DIR}/OpenCL-ICD-Loader
mkdir build
cd build
cmake ..
make C_INCLUDE_PATH=${OPENCL_HEADERS}
exit_if_err $? "failed to build OpenCL-ICD-Loader"
