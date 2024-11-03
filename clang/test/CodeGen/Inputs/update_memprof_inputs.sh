#!/bin/bash

if [ -z $1 ]; then
  echo "Path to clang required!"
  echo "Usage: update_memprof_inputs.sh /path/to/updated/clang"
  exit 1
else
  CLANG=$1
fi

# Allows the script to be invoked from other directories.
OUTDIR=$(dirname $(realpath -s $0))

DEFAULT_MEMPROF_FLAGS="-fuse-ld=lld -Wl,--no-rosegment -gmlt -fdebug-info-for-profiling -fmemory-profile -mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id -no-pie"

${CLANG} ${DEFAULT_MEMPROF_FLAGS} ${OUTDIR}/../memprof.cpp -o ${OUTDIR}/memprof.exe
env MEMPROF_OPTIONS=log_path=stdout ${OUTDIR}/memprof.exe > ${OUTDIR}/memprof.memprofraw
