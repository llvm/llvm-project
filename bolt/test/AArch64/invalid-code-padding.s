# RUN: %clang %cflags %s -o %t.so -Wl,-q
# RUN: not llvm-bolt %t.so -o %t.bolt --fail-on-invalid-padding \
# RUN:   2>&1 | FileCheck %s
# CHECK: BOLT-ERROR: found 1 instance(s) of invalid code padding

  .text
  .align 2
  .global foo
  .type foo, %function
foo:
  cmp x0, x1
  b.eq .Ltmp1
  adrp x1, jmptbl
  add x1, x1, :lo12:jmptbl
  ldrsw x2, [x1, x2, lsl #2]
  br x2
  b .Ltmp1
.Ltmp2:
  add x0, x0, x1
  ret
  .size foo, .-foo

.Ltmp1:
  add x0, x0, x1
  b .Ltmp2

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE

  .section .rodata, "a"
  .align 2
  .global jmptbl
jmptbl:
  .word .text+0x28 - .
