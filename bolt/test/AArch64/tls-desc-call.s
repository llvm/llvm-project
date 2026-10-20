# RUN: %clang %cflags %s -o %t.so -fPIC -shared -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt --debug-only=bolt 2>&1 | FileCheck %s

# REQUIRES: asserts

## Verify that R_AARCH64_TLSDESC_CALL relocations are ignored

# CHECK-NOT: Relocation {{.*}} R_AARCH64_TLSDESC_CALL

  .text
  .globl  get_tls_var
  .p2align  2
  .type get_tls_var,@function
get_tls_var:
  .cfi_startproc
  str     x30, [sp, #-16]!
  adrp  x0, :tlsdesc:tls_var
  ldr x1, [x0, :tlsdesc_lo12:tls_var]
  add x0, x0, :tlsdesc_lo12:tls_var
  .tlsdesccall tls_var
  blr x1
  mrs x8, TPIDR_EL0
  ldr w0, [x8, x0]
  ldr x30, [sp], #16
  ret
  .size get_tls_var, .-get_tls_var
  .cfi_endproc

  .type tls_var,@object
  .section  .tdata,"awT",@progbits
  .globl  tls_var
  .p2align  2, 0x0
tls_var:
  .word 42
  .size tls_var, 4
