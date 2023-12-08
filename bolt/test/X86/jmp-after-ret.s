## This test checks that the unreachable unconditional branch is removed
## if it is located after return instruction.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt | FileCheck %s

# CHECK: UCE removed 1 blocks

    .text
    .globl main
    .type main, %function
    .size main, .Lend-main
main:
    je 1f
    retq
    jmp main
1:
    movl $0x2, %ebx
    retq
.Lend:
