# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s -o %t/hello.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/main.o
# RUN: %lld -L %t %t/main.o -lhello.o -o %t/a.out
# RUN: llvm-nm %t/a.out | FileCheck %s

# CHECK: _main
# CHECK: _print_hello

.globl _main
_main:
    call _print_hello
    ret
