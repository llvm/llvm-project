#!/bin/bash

BRANCH=
BUILD_NUMBER=
PR_NUMBER=
TESTCASE=

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
while getopts ":b:r:n:t:" option; do
    case $option in
        b) BRANCH=$OPTARG ;;
        n) BUILD_NUMBER=$OPTARG ;;
        r) PR_NUMBER=$OPTARG ;;
        t) TESTCASE=$OPTARG ;;
    esac
done && shift $(($OPTIND - 1))

# we're in llvm.obj dir
BUILD_DIR=${PWD}

if [ -z "${TESTCASE}" ]; then
    echo "No target provided"
    exit 1
fi

make ${TESTCASE} VERBOSE=1 LIT_ARGS="-v -j `nproc`"
