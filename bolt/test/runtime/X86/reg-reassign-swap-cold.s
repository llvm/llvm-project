# This test case reproduces a bug where, during register swapping,
# the code fragments associated with the function need to be swapped
# together (which may be generated during PGO optimization). If not
# handled properly, optimized binary execution can result in a segmentation fault.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out -data=%t.fdata --reg-reassign | FileCheck %s
# RUN: %t.out

# CHECK: BOLT-INFO: Reg Reassignment Pass Stats
# CHECK-NEXT: 2 functions affected.
  .text
  .globl  main
  .globl  main.cold
  .p2align  4, 0x90
  .type   main,@function
  .type   main.cold,@function
main.cold:
bb1:
  cmp     $0x3, %r12
  jne     bb8
bb2:
  jmp     bb4
main:                                   # @main
  .cfi_startproc
# %bb.0:                                # %entry
  pushq   %rax
  pushq   %r12
  pushq   %rbx
  .cfi_def_cfa_offset 16
  mov     $0x1,  %r12
  mov     $0x2,  %rbx
  add     $0x1,  %r12
  shr     $0x14, %r12
  mov     $0x3,  %r12
bb3:
  jmp     bb1
bb4:
  cmp     $0x3,  %r12
bb5:
  jne     bb8
bb6:
  xorl    %eax, %eax
bb7:
  popq    %rcx
  popq    %rbx
  popq    %r12
  .cfi_def_cfa_offset 8
  retq
bb8:
  mov  $0x1, %rax
  jmp  bb7
# FDATA: 1 main.cold #bb2# 1 main #bb4# 0 100
# FDATA: 1 main #bb5# 1 main #bb6# 0 100
# FDATA: 1 main #bb3# 1 main.cold 0 0 100

.Lfunc_end0:
  .size  main, .Lfunc_end0-main
  .cfi_endproc
