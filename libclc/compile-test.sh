#!/bin/sh

clang -target nvptx--nvidiacl -Iptx-nvidiacl/include -Igeneric/include -Xclang -mlink-bitcode-file -Xclang nvptx--nvidiacl/lib/builtins.bc -include clc/clc.h -Dcl_clang_storage_class_specifiers -Dcl_khr_fp64 "$@"
