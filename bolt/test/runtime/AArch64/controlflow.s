# This test checks relocations for control-flow instructions

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt -use-old-text=0 -lite=0 -trap-old-code
# RUN: %t.bolt

  .macro panic
.panic\@:
  mov x0, #0
  br x0
  .endm

  .text
  .align 4
  .global test_call_foo
  .type test_call_foo, %function
test_call_foo:
  mov x0, x30
  add x0, x0, #8
  br x0
  panic
  .size test_call_foo, .-test_call_foo

  .global main
  .type main, %function
main:
  stp x30, x29, [sp, #-16]!
  b test_branch
  panic
test_branch:
  b test_cond_branch
  panic
  .size main, .-main

  .global test_cond_branch
  .type test_cond_branch, %function
test_cond_branch:
  mov x0, #0
  cmp x0, #0
  b.eq test_branch_reg
  panic
  .size test_cond_branch, .-test_cond_branch

  .global test_branch_reg
  .type test_branch_reg, %function
test_branch_reg:
  adr x0, test_branch_zero
  br x0
  panic
  .size test_branch_reg, .-test_branch_reg

  .global test_branch_zero
  .type test_branch_zero, %function
test_branch_zero:
  mov x0, #0
  cbz x0, test_branch_non_zero
  panic
  .size test_branch_zero, .-test_branch_zero

  .global test_branch_non_zero
  .type test_branch_non_zero, %function
test_branch_non_zero:
  mov x0, #1
  cbnz x0, test_bit_branch_zero
  panic
  .size test_branch_non_zero, .-test_branch_non_zero

  .global test_bit_branch_zero
  .type test_bit_branch_zero, %function
test_bit_branch_zero:
  mov x0, #0
  tbz x0, 0, test_bit_branch_non_zero
  panic
  .size test_bit_branch_zero, .-test_bit_branch_zero

  .global test_bit_branch_non_zero
  .type test_bit_branch_non_zero, %function
test_bit_branch_non_zero:
  mov x0, #1
  tbnz x0, 0, test_call
  panic
  .size test_bit_branch_non_zero, .-test_bit_branch_non_zero

  .global test_call
  .type test_call, %function
test_call:
  bl test_call_foo
  panic
  b test_call_reg
  panic
  .size test_call, .-test_call

  .global test_call_reg
  .type test_call_reg, %function
test_call_reg:
  adr x0, test_call_foo
  blr x0
  panic
  b finalize
  panic
  .size test_call_reg, .-test_call_reg

  .global finalize
  .type finalize, %function
finalize:
  ldp x30, x29, [sp], #16
  mov x0, #0
  ret
  panic
  .size finalize, .-finalize
