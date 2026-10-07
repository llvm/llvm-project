# This test checks that LongJmp rejects non-call branches beyond 128MiB
# even when BTI is enabled.

# REQUIRES: system-linux

# RUN: %clang %s %cflags -Wl,-q -o %t -mbranch-protection=bti -Wl,-z,force-bti
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: not llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions \
# RUN: --print-split --print-only foo --print-longjmp 2>&1 | FileCheck %s

# CHECK: Binary Function "foo" after split-functions

# CHECK:      cmp     x0, #0x0
# CHECK: Successors: .Ltmp0

# CHECK: -------   HOT-COLD SPLIT POINT   -------

# CHECK:      mov     x0, #0x2
# CHECK-NEXT: ret

# CHECK: BOLT-INFO: Starting stub-insertion pass
# CHECK: BOLT-ERROR: Unable to relax non-call branch beyond 128MiB

  .text
  .globl  foo
  .type foo, %function
foo:
.cfi_startproc
.entry_bb:
# FDATA: 1 foo #.entry_bb# 10
    cmp x0, #0
    b .Lcold_bb1
.Lcold_bb1:
    mov x0, #2
    ret
.cfi_endproc
  .size foo, .-foo

# empty space, so the splitting needs short stubs
.data
.space 0x8000000

## Force relocation mode.
.reloc 0, R_AARCH64_NONE
