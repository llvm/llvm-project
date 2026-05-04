## Verifies that llvm-bolt ignores function calls to 0.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.null -v=2 2>&1 | FileCheck %s
# CHECK: Function main has a call to address zero.

        .text
  .globl main
  .type main, %function
main:
# FDATA: 0 [unknown] 0 1 main 0 0 0
        .cfi_startproc
.LBB00:
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        movl    $0x0, %eax
        testq   %rax, %rax
.LBB00_br:      je      .Ltmp0
# FDATA: 1 main #.LBB00_br# 1 main #.Ltmp0# 0 0
# FDATA: 1 main #.LBB00_br# 1 main #.LFT0# 0 0

.LFT0:
        movl    $0x0, %eax
.LFT0_br:       callq   0
# FDATA: 1 main #.LFT0_br# 1 main #.Ltmp0# 0 0

.Ltmp0:
        movl    $0x0, %eax
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        retq

        .cfi_endproc
.size main, .-main
