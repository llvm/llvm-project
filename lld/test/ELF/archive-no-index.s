# REQUIRES: x86
# Tests error on archive file without a symbol table
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux -o /dev/null %S/Inputs/archive.s
# RUN: rm -f %t.a
# RUN: llvm-ar crS %t.a %t.archive.o

# RUN: not ld.lld -o out.wasm %t.o %t.a 2>&1 | FileCheck %s

.globl _start
_start:

# CHECK: error: {{.*}}.a: archive has no index; run ranlib to add one
