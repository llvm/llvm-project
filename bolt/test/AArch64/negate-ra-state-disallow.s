# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags  %t.o -o %t.exe -Wl,-q
# RUN: not llvm-bolt %t.exe -o %t.exe.bolt --update-branch-protection=false 2>&1 | FileCheck %s

# CHECK: BOLT-ERROR: --update-branch-protection is set to false, but foo contains .cfi-negate-ra-state

  .text
  .globl  foo
  .p2align        2
  .type   foo,@function
foo:
  .cfi_startproc
  hint    #25
  .cfi_negate_ra_state
  mov x1, #0
  hint    #29
  .cfi_negate_ra_state
  ret
  .cfi_endproc
  .size   foo, .-foo

  .global _start
  .type _start, %function
_start:
  b foo
