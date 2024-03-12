#!/usr/bin/env bash

set -x
set -e
MODULE_DIR='/usr/local/include/c++/v1/__modules'

FLAGS='-std=c++26 -pthread'
LIBCXX_FLAGS='-nostdinc++ -isystem /usr/local/include/c++/v1 -isystem /usr/local/include/x86_64-unknown-linux-gnu/c++/v1'
function compile_module {
  MODULE_NAME=$1
  shift
  clang++ $LIBCXX_FLAGS $FLAGS -fmodule-output=./$MODULE_NAME.pcm -c $MODULE_DIR/$MODULE_NAME.cppm -o $MODULE_NAME.o  "$@"

}
compile_module std $@
compile_module std.compat -fmodule-file=std=./std.pcm $@

ar rcs libc++modules.a std.o std.compat.o
