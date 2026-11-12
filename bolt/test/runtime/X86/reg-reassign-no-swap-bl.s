# This test case is used to reproduce an issue found in the mongod database.
# In function rankRegisters, if there is a BH Reg in the basic block, then the BL Reg
# also cannot be swap.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out -data=%t.fdata --reg-reassign | FileCheck %s
# RUN: %t.out

# CHECK: Reg Reassignment Pass: no changes were made
  .text
  .globl  main
  .globl  main.cold
  .p2align  4, 0x90
  .type  main,@function
  .type  main.cold,@function
main.cold:
bb1:
  mov    $0x2,  %bh
bb2:
  jmp    bb5
main:                                   # @main
  .cfi_startproc
# %bb.0:                                # %entry
  pushq  %rax
  pushq   %r12
  pushq   %rbx
  .cfi_def_cfa_offset 16
  mov     $0x1,   %r12
  shr     $0x14,  %r12
  add     $0x14,  %r12
  mov     $0x11,  %rbx
  mov    $0x1,  %bh
  mov    $0x1,  %bl
bb3:
  add    $0x1,   %r12
bb4:
  jmp    bb1
bb5:
  cmp     $0x201,  %rbx
  jne     0x0
bb6:
  xorl    %eax, %eax
  popq    %rcx
  popq    %rbx
  popq    %r12
  .cfi_def_cfa_offset 8
  retq
# FDATA: 1 main.cold #bb2# 1 main 0 0 100
# FDATA: 1 main #bb3# 1 main #bb4# 0 100
# FDATA: 1 main #bb4# 1 main.cold 0 0 100

.Lfunc_end0:
  .size  main, .Lfunc_end0-main
  .cfi_endproc
