#!/bin/bash

BRANCH=
BUILD_NUMBER=
PR_NUMBER=

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
while getopts ":b:r:n:" option; do
    case $option in
        b) BRANCH=$OPTARG ;;
        n) BUILD_NUMBER=$OPTARG ;;
        r) PR_NUMBER=$OPTARG ;;
    esac
done && shift $(($OPTIND - 1))

# we're in llvm.obj dir
BUILD_DIR=${PWD}

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
make C_INCLUDE_PATH=${OPENCL_HEADERS}
exit_if_err $? "failed to build OpenCL-ICD-Loader"
