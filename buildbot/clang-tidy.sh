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

if [ -z "${PR_NUMBER}" ]; then
    echo "No PR number provided"
    exit 1
fi

# we're in llvm.src dir
SRC_DIR=${PWD}
BUILDER_DIR=$(cd ..; pwd)

# Get changed files
base_commit=`git merge-base origin/sycl refs/pull/${PR_NUMBER}/merge`
exit_if_err $? "fail to get base commit"

path_list_file=${BUILDER_DIR}/changed_files.txt
git --no-pager diff ${base_commit} refs/pull/${PR_NUMBER}/merge --name-only > ${path_list_file}
cat ${path_list_file}

# Run clang-tidy
while IFS='' read -r line ; do
    file_name=$(basename ${line})
    file_ext=${file_name##*.}
    if [[ "${file_ext}" == "h" || "${file_ext}" == "hpp" || "${file_ext}" == "c" || "${file_ext}" == "cc" || "${file_ext}" == "cpp" ]]; then
        ${BUILDER_DIR}/llvm.obj/bin/clang-tidy ${line}
    fi
done < "${path_list_file}"
