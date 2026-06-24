#!/bin/bash

# Copies the 'demangle' library, excluding 'DemangleConfig.h', to llvm. If no
# llvm directory is specified, then assume a monorepo layout.

set -e

cd $(dirname $0)
HDRS="ItaniumDemangle.h ItaniumNodes.def StringViewExtras.h Utility.h"
TEST_HDRS="DemangleTestCases.inc"
LLVM_DEMANGLE_DIR=$1
LLVM_TESTING_DIR=

if [[ -z "$LLVM_DEMANGLE_DIR" ]]; then
    LLVM_DEMANGLE_DIR="../../../llvm/include/llvm/Demangle"
    LLVM_TESTING_DIR=$LLVM_DEMANGLE_DIR/../Testing/Demangle
fi

if [[ ! -d "$LLVM_DEMANGLE_DIR" ]]; then
    echo "No such directory: $LLVM_DEMANGLE_DIR" >&2
    exit 1
fi

if [[ ! -d "$LLVM_TESTING_DIR" ]]; then
    LLVM_TESTING_DIR="../../../llvm/include/llvm/Testing/Demangle"
fi

if [[ ! -d "$LLVM_TESTING_DIR" ]]; then
    echo "No such directory: $LLVM_TESTING_DIR" >&2
    exit 1
fi

read -p "This will overwrite the copies of $HDRS in $LLVM_DEMANGLE_DIR and $TEST_HDRS in $LLVM_TESTING_DIR; are you sure? [y/N]" -n 1 -r ANSWER
echo

copy_files() {
    local src=$1
    local dst=$2
    local hdrs=$3

    cp -f README.txt $dst
    chmod -w $dst/README.txt

    for I in $hdrs ; do
	    echo "Copying ${src}/$I to ${dst}/$I"
	    rm -f $dst/$I
	    dash=$(echo "$I---------------------------" | cut -c -27 |\
		       sed 's|[^-]*||')
	    sed -e '1s|^//=*-* .*\..* -*.*=*// *$|//===--- '"$I $dash"'-*- mode:c++;eval:(read-only-mode) -*-===//|' \
	        -e '2s|^// *$|//       Do not edit! See README.txt.|' \
	        $src/$I >$dst/$I
	    chmod -w $dst/$I
    done
}

if [[ $ANSWER =~ ^[Yy]$ ]]; then
    copy_files . $LLVM_DEMANGLE_DIR "${HDRS}"
    copy_files ../../test $LLVM_TESTING_DIR "${TEST_HDRS}"
fi
