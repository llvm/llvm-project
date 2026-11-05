# This test checks that BOLT can generate BTI landing pads for targets of stubs inserted in LongJmp.

# REQUIRES: system-linux

# RUN: %clang %s %cflags -Wl,-q -o %t -mbranch-protection=bti -Wl,-z,force-bti
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata \
# RUN: --hot-functions-at-end --align-text=0x10000000 --lite=0 \
# RUN: --print-only=foo,bar --print-longjmp 2>&1 | FileCheck %s

# CHECK: BOLT-INFO: Starting stub-insertion pass
# CHECK: Binary Function "foo" after long-jmp

# CHECK: cmp x0, #0x0
# CHECK-NEXT: bl .LStub0
# CHECK: adrp x16, bar
# CHECK-NEXT: add x16, x16, :lo12:bar
# CHECK-NEXT: br x16 # UNKNOWN CONTROL FLOW
# CHECK: Binary Function "bar" after long-jmp
# CHECK: bti c
# CHECK-NEXT: mov x0, #0x2
# CHECK-NEXT: ret

  .text
  .globl  foo
  .type foo, %function
foo:
.cfi_startproc
.entry_bb:
# FDATA: 1 foo #.entry_bb# 10
    cmp x0, #0
    bl bar
    ret
.cfi_endproc
  .size foo, .-foo

# Align hot text to 256MiB, so the call to the cold function needs a short stub.
  .globl bar
  .type bar, %function
bar:
    mov x0, #2
    ret
  .size bar, .-bar

## Force relocation mode.
.reloc 0, R_AARCH64_NONE
