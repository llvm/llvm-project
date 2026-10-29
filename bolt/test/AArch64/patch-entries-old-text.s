## Check that original function entries are not patched when the original
## text section will be reused for new code: the patches would be lost.
## --force-patch makes PatchEntries patch every function whenever it runs.

# REQUIRES: system-linux

# RUN: %clang %cflags -pie %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 --force-patch -v=1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-PATCH
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 --force-patch -v=1 \
# RUN:   --use-old-text --align-text=4 \
# RUN:   | FileCheck %s --check-prefix=CHECK-NOPATCH

# CHECK-PATCH: patching entries in original code
# CHECK-NOPATCH-NOT: patching entries in original code

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  stp x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  mov x29, sp
  bl foo
  ldp x29, x30, [sp], #16
  .cfi_def_cfa_offset 0
  .cfi_restore w30
  .cfi_restore w29
  ret
  .cfi_endproc
  .size _start, .-_start

## BOLT drops the nops, so the new code fits in the old space.
  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  .rept 16
  nop
  .endr
  ret
  .cfi_endproc
  .size foo, .-foo
