# Checking that we generate an OpNegateRAState CFI after the split point,
# when splitting a region with signed RA state.
# We split at the fallthrough label.

# REQUIRES: system-linux

# RUN: %clang %s %cflags -march=armv8.3-a -Wl,-q -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions \
# RUN: --print-only foo --print-split --print-all 2>&1 | FileCheck %s

# Checking that we don't see any OpNegateRAState CFIs before the insertion pass.
# CHECK-NOT: OpNegateRAState
# CHECK: Binary Function "foo" after insert-negate-ra-state-pass

# CHECK:       paciasp
# CHECK-NEXT:  OpNegateRAState

# CHECK: -------   HOT-COLD SPLIT POINT   -------

# CHECK:         OpNegateRAState
# CHECK-NEXT:    mov x0, #0x1
# CHECK-NEXT:    autiasp
# CHECK-NEXT:    OpNegateRAState
# CHECK-NEXT:    ret

# End of the insert-negate-ra-state-pass logs
# CHECK: Binary Function "foo" after finalize-functions

  .text
  .globl  foo
  .type foo, %function
foo:
.cfi_startproc
.entry_bb:
# FDATA: 1 foo #.entry_bb# 10
     paciasp
    .cfi_negate_ra_state     // indicating that paciasp changed the RA state to signed
    cmp x0, #0
    b.eq .Lcold_bb1
.Lfallthrough:               // split point
    mov x0, #1
    autiasp
    .cfi_negate_ra_state     // indicating that autiasp changed the RA state to unsigned
    ret
.Lcold_bb1:                  // Instructions below are not important, they are just here so the cold block is not empty.
    .cfi_negate_ra_state     // ret has unsigned RA state, but the next inst (autiasp) has signed RA state
    mov x0, #2
    retaa
.cfi_endproc
  .size foo, .-foo

## Force relocation mode.
.reloc 0, R_AARCH64_NONE
