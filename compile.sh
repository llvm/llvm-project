#!/usr/bin/env bash

# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

function echo_error() {
  echo -e "\033[0;31m$1\033[0m"
}

# Find the root of the git repository
GIT_ROOT=$(git rev-parse --show-toplevel)
# If we move this script file to other repos, we will need to adjust this
LLVM_PROJECT_ROOT="$GIT_ROOT"

# Check that the LLVM project directory exists.
if [[ ! -d "${LLVM_PROJECT_ROOT}" ]]; then
  echo_error "Error: This script expects the LLVM project to be in the tfhe-llvm directory next to the repository root."
  exit 1
fi

LLVM_BIN="${LLVM_PROJECT_ROOT}/build/bin"

CLANG="${LLVM_BIN}/clang"
LLC="${LLVM_BIN}/llc"
LLVM_DIS="${LLVM_BIN}/llvm-dis"
LLVM_STRIP="${LLVM_BIN}/llvm-strip"

# Compile the C code; if the -O2 is given to this script then it should be
# passed onto the following commands.
CFILE=$1
OPT=$2
SELECTION_BACKEND=$3
DEBUG_FLAGS=${4:-}

# Determine the LLC option based on the SELECTION_BACKEND argument
if [ "$SELECTION_BACKEND" == "globalisel" ]; then
  LLC_OPTION="-global-isel"
elif [ "$SELECTION_BACKEND" == "selectiondag" ]; then
  LLC_OPTION=""
else
  echo_error "Error: Invalid SELECTION_BACKEND value. Use 'globalisel' or 'selectiondag'."
  exit 1
fi

"${CLANG}" -fdeclspec "$OPT" -emit-llvm -target parasol -c "$CFILE".c -o "$CFILE".bc

# Immediately try to disassemble into a *.ll file
if ! "${LLVM_DIS}" "$CFILE".bc; then
  echo_error "Error: Disassembly failed"
  exit 1
fi

# Store the command and its arguments in a string
LLC_CMD="${LLC} ${LLC_OPTION} -march=parasol ${OPT} -filetype=asm ${CFILE}.bc -o ${CFILE}.S"

# Add the DEBUG_FLAGS to the command if it is not empty
if [ -n "$DEBUG_FLAGS" ]; then
  LLC_CMD="${LLC_CMD} ${DEBUG_FLAGS}"
fi

# Run the command
if ! eval "$LLC_CMD"; then
  echo_error "Error: llc command for assembly output failed"
  exit 2
fi

# Run the LLC command but output an object file
if ! "${LLC}" ${LLC_OPTION} -march=parasol "$OPT" -filetype=obj "$CFILE".bc -o "$CFILE".o; then
  echo_error "Error: llc command for object file output failed"
  exit 3
fi

# Strip the object file of .comment to make the ELF files not include the
# compiler information. This should allow compiler changes that generate the
# same byte code to produce the same ELF. There may be some other places of
# non-determinism though.
if ! "${LLVM_STRIP}" --no-strip-all --remove-section .comment "$CFILE".o; then
  echo_error "Error: llvm-strip command failed"
  exit 4
fi
