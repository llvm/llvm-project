# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --data %t.fdata --print-finalized \
# RUN: -o %t.out --three-way-branch | FileCheck %s
# RUN: %t.exe
# RUN: %t.out

# FDATA: 1 main 8 1 main a 0 22
# FDATA: 1 main 8 1 main #.BB1# 0 50
# FDATA: 1 main 12 1 main 14 0 30
# FDATA: 1 main 12 1 main #.BB2# 0 40
# CHECK: Successors: .Ltmp1 (mispreds: 0, count: 40), .Ltmp0 (mispreds: 0, count: 52)
# CHECK: Successors: .LFT0 (mispreds: 0, count: 22), .LFT1 (mispreds: 0, count: 30)

    .text
    .globl main
    .type main, %function
    .size main, .Lend-main
main:
    mov $0x0, %eax
    cmp $0x1, %eax
    jge .BB1
    mov $0xf, %eax
    xor %eax, %eax
    retq
.BB1:
    jg .BB2
    retq
.BB2:
    mov $0x7, %eax
    retq
.Lend:
