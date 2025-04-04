# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags  %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.exe.bolt | FileCheck %s

# check that the output is listing foo as incorrect
# CHECK: BOLT-INFO: inconsistent RAStates in function foo

# check that foo got Ignored, so it's not in the new .text section
# RUN: llvm-objdump %t.exe.bolt -d -j .text > %t.exe.dump
# RUN: not grep "<foo>:" %t.exe.dump


# How is this test incorrect?
# There is an extra .cfi_negate_ra_state in foo.
# Because of this, we will get to the autiasp (hint #29)
# in a (seemingly) unsigned state. That is incorrect.
  .text
  .globl  foo
  .p2align        2
  .type   foo,@function
foo:
  .cfi_startproc
  hint    #25
  .cfi_negate_ra_state
  sub     sp, sp, #16
  stp     x29, x30, [sp, #16]             // 16-byte Folded Spill
  .cfi_def_cfa_offset 16
  str     w0, [sp, #12]
  ldr     w8, [sp, #12]
  .cfi_negate_ra_state
  add     w0, w8, #1
  ldp     x29, x30, [sp, #16]             // 16-byte Folded Reload
  add     sp, sp, #16
  hint    #29
  .cfi_negate_ra_state
  ret
.Lfunc_end1:
  .size   foo, .Lfunc_end1-foo
  .cfi_endproc

  .global _start
  .type _start, %function
_start:
  b foo
