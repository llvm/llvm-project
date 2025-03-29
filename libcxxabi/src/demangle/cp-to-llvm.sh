#!/bin/bash

# Copies the 'demangle' library, excluding 'DemangleConfig.h', to llvm. If no
# llvm directory is specified, then assume a monorepo layout.

set -e

cd $(dirname $0)
HDRS="ItaniumDemangle.h ItaniumNodes.def StringViewExtras.h Utility.h"
SRCS="Utility.cpp"
LLVM_DEMANGLE_INCLUDE_DIR=$1
LLVM_DEMANGLE_SOURCE_DIR=$2

if [[ -z "$LLVM_DEMANGLE_INCLUDE_DIR" ]]; then
    LLVM_DEMANGLE_INCLUDE_DIR="../../../llvm/include/llvm/Demangle"
fi

if [[ -z "$LLVM_DEMANGLE_SOURCE_DIR" ]]; then
    LLVM_DEMANGLE_SOURCE_DIR="../../../llvm/lib/Demangle"
fi

if [[ ! -d "$LLVM_DEMANGLE_INCLUDE_DIR" ]]; then
    echo "No such directory: $LLVM_DEMANGLE_INCLUDE_DIR" >&2
    exit 1
fi

if [[ ! -d "$LLVM_DEMANGLE_SOURCE_DIR" ]]; then
    echo "No such directory: $LLVM_DEMANGLE_SOURCE_DIR" >&2
    exit 1
fi

read -p "This will overwrite the copies of $HDRS in $LLVM_DEMANGLE_INCLUDE_DIR and $SRCS in $LLVM_DEMANGLE_SOURCE_DIR; are you sure? [y/N]" -n 1 -r ANSWER
echo

function copy_files() {
    local dest_dir=$1
    local files=$2
    local adjust_include_paths=$3

    cp -f README.txt $dest_dir
    chmod -w $dest_dir/README.txt
    for I in $files ; do
    rm -f $dest_dir/$I
    dash=$(echo "$I---------------------------" | cut -c -27 |\
    	   sed 's|[^-]*||')
    sed -e '1s|^//=*-* .*\..* -*.*=*// *$|//===--- '"$I $dash"'-*- mode:c++;eval:(read-only-mode) -*-===//|' \
        -e '2s|^// *$|//       Do not edit! See README.txt.|' \
        $I >$dest_dir/$I

    if [[ "$adjust_include_paths" = true ]]; then
        sed -i '' \
            -e 's|#include "DemangleConfig.h"|#include "llvm/Demangle/DemangleConfig.h"|' \
            -e 's|#include "Utility.h"|#include "llvm/Demangle/Utility.h"|' \
            $dest_dir/$I
    fi

    chmod -w $dest_dir/$I
    done
}

if [[ $ANSWER =~ ^[Yy]$ ]]; then
  copy_files $LLVM_DEMANGLE_INCLUDE_DIR "$HDRS" false
  copy_files $LLVM_DEMANGLE_SOURCE_DIR "$SRCS" true
fi
