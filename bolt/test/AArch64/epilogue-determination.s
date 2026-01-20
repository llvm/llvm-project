# Test that we will not incorrectly take the first basic block in function
# `_foo` or the second basic block in function `_goo` as epilogue, and will
# recognize epilogues in the other cases.

# RUN: %clang %cflags %s -o %t.so -Wl,-q,-z,undefs
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
# CHECK-LABEL: _foo
# CHECK: br x10
# CHECK-SAME: # UNKNOWN CONTROL FLOW
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
  tbnz x0, #0x3, _L2
  ldr x8, [x29, #0x30]
  blr x8
_L2:
  add sp, sp, #0x10
  ldp x29, x30, [sp], #0x10
  br x2
# CHECK-LABEL: _bar
# CHECK: br x2
# CHECK-SAME: # TAILCALL

  .global _goo
  .type _goo, %function
_goo:
  ldr w8, [sp]
  adr x10, _jmptbl2
  ldrsw x9, [x10, x9, lsl #2]
  add x10, x10, x9
  str x30, [sp, #-0x10]!
  bl _bar
  ldr x30, [sp], #0x10
  mov x1, x0
  mov x0, xzr
  br x10
# CHECK-LABEL: _goo
# CHECK: br x10
# CHECK-SAME: # UNKNOWN CONTROL FLOW
  mov x0, 0
  ret
  mov x0, 1
  ret

  .balign 4
_jmptbl2:
  .long -16
  .long -8

  .global _faz
  .type _faz, %function
_faz:
  str x30, [sp, #-0x10]!
  tbnz x0, #0x1, _L3
  bl _bar
_L3:
  ldr x30, [sp], #0x10
  mov x0, x1
  br x3
# CHECK-LABEL: _faz
# CHECK: br x3
# CHECK-SAME: # TAILCALL

  .global _start
  .type _start, %function
_start:
  ret

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE
