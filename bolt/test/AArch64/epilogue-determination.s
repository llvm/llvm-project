# Test that we will not incorrectly take the first basic block in function
# `_foo` as epilogue due to the first load from stack instruction.

# RUN: %clang %cflags %s -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt --print-cfg | FileCheck %s

  .text
  .global _foo
  .type _foo, %function
_foo:
  ldr w8, [sp]
  adr x10, _jmptbl
  ldrsw x9, [x10, x9, lsl #2]
  add x10, x10, x9
  br x10
# CHECK-NOT: x10 # TAILCALL
# CHECK: x10 # UNKNOWN CONTROL FLOW
  mov x0, 0
  ret
  mov x0, 1
  ret

  .balign 4
_jmptbl:
  .long -16
  .long -8

  .global _bar
  .type _bar, %function
_bar:
  stp x29, x30, [sp, #-0x10]!
  mov x29, sp
  sub sp, sp, #0x10
  ldr x8, [x29, #0x30]
  blr x8
  add sp, sp, #0x10
  ldp x29, x30, [sp], #0x10
  br x2
# CHECK-NOT: x2 # UNKNOWN CONTROL FLOW
# CHECK: x2 # TAILCALL

  .global _start
  .type _start, %function
_start:
  ret

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE
