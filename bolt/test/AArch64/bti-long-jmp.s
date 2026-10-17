# This test checks that BOLT can generate BTI landing pads for targets of stubs inserted in LongJmp.

# REQUIRES: system-linux

# RUN: %clang %s %cflags -Wl,-q -o %t -mbranch-protection=bti -Wl,-z,force-bti
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions \
# RUN: --print-split --print-only foo --print-longjmp 2>&1 | FileCheck %s

# CHECK: BOLT-INFO: Starting stub-insertion pass
# CHECK: Binary Function "foo" after long-jmp

# CHECK:      cmp     x0, #0x0
# CHECK-NEXT: Successors: .LStub0

# CHECK:      adrp    x16, .Ltmp0
# CHECK-NEXT: add     x16, x16, :lo12:.Ltmp0
# CHECK-NEXT: br      x16 # UNKNOWN CONTROL FLOW

# CHECK: -------   HOT-COLD SPLIT POINT   -------

# CHECK:      bti     c
# CHECK-NEXT: mov     x0, #0x2
# CHECK-NEXT: ret

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
