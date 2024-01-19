# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s -o %t/hello.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/main.o
# RUN: %lld -L %t %t/main.o %t/hello.o -o %t/a.out

.globl _main
_main:
    call _print_hello
    ret
