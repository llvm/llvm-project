# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: link_fdata %s %t.o %t.fdata2 "FDATA2"
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -data %t.fdata -reorder-blocks=cache+ -print-finalized \
# RUN:    -loop-inversion-opt -o %t.out | FileCheck %s
# RUN: llvm-bolt %t.exe -data %t.fdata2 -reorder-blocks=cache+ -print-finalized \
# RUN:    -loop-inversion-opt -o %t.out2 | FileCheck --check-prefix="CHECK2" %s

# The case where loop is used:
# FDATA: 1 main 2 1 main #.J1# 0 420
# FDATA: 1 main b 1 main #.Jloop# 0 420
# FDATA: 1 main b 1 main d 0 1
# CHECK: BB Layout   : .LBB00, .Ltmp0, .Ltmp1, .LFT0

# The case where loop is unused:
# FDATA2: 1 main 2 1 main #.J1# 0 420
# FDATA2: 1 main b 1 main #.Jloop# 0 1
# FDATA2: 1 main b 1 main d 0 420
# CHECK2: BB Layout   : .LBB00, .Ltmp1, .LFT0, .Ltmp0

    .text
    .globl main
    .type main, %function
    .size main, .Lend-main
main:
    xor %eax, %eax
    jmp .J1
.Jloop:
    inc %rax
.J1:
    cmp $16, %rax
    jl .Jloop
    retq
.Lend:
