## Check that ICF does not fold functions with identical code but different
## return address signing state. BOLT keeps that state as instruction
## annotations instead of CFI pseudo instructions.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown -mattr=+pauth \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --icf=all 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-ICF
# RUN: llvm-nm -n %t.bolt > %t.txt
# RUN: llvm-dwarfdump --eh-frame %t.bolt >> %t.txt
# RUN: FileCheck %s --input-file %t.txt

## Only signed_b is folded, into signed_a.
# CHECK-ICF: BOLT-INFO: ICF folded 1 out of {{[0-9]+}} functions

# CHECK-DAG: [[#%x,SIGNED_C:]] T signed_c
# CHECK-DAG: [[#%x,ENTRY_SIGNED:]] T entry_signed
# CHECK-DAG: [[#%x,RA_LATE:]] T ra_late
# CHECK-DAG: [[#%x,SAME:]] T signed_a
# CHECK-DAG: [[#SAME]] T signed_b

## signed_c keeps its own CFI, which says that the return address is signed.
# CHECK:      FDE cie={{[0-9a-f]+}} pc=[[#%.8x,SIGNED_C]]...
# CHECK-NEXT:   Format: DWARF32
# CHECK-NEXT:   DW_CFA_advance_loc: 4 to 0x[[#%x,SIGNED_C+4]]
# CHECK-NEXT:   DW_CFA_AARCH64_negate_ra_state:

## entry_signed is entered with a signed return address.
# CHECK:      FDE cie={{[0-9a-f]+}} pc=[[#%.8x,ENTRY_SIGNED]]...
# CHECK-NEXT:   Format: DWARF32
# CHECK-NEXT:   DW_CFA_AARCH64_negate_ra_state:

## ra_late signs the return address after its first instruction.
# CHECK:      FDE cie={{[0-9a-f]+}} pc=[[#%.8x,RA_LATE]]...
# CHECK-NEXT:   Format: DWARF32
# CHECK-NEXT:   DW_CFA_advance_loc: 4 to 0x[[#%x,RA_LATE+4]]
# CHECK-NEXT:   DW_CFA_AARCH64_negate_ra_state:

  .text
  .globl _start
  .type _start, %function
_start:
  bl signed_a
  bl signed_b
  bl unsigned_c
  bl signed_c
  bl entry_unsigned
  bl entry_signed
  bl ra_early
  bl ra_late
  ret
  .size _start, .-_start

  .globl callee
  .type callee, %function
callee:
  ret
  .size callee, .-callee

  .macro SIGNED name, n
  .globl \name
  .type \name, %function
\name:
  .cfi_startproc
  paciasp
  .cfi_negate_ra_state
  stp x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  mov w0, #\n
  bl callee
  ldp x29, x30, [sp], #16
  .cfi_def_cfa_offset 0
  autiasp
  .cfi_negate_ra_state
  .cfi_restore w30
  .cfi_restore w29
  ret
  .cfi_endproc
  .size \name, .-\name
  .endm

## Identical code and signing state: folded.
  SIGNED signed_a, 1
  SIGNED signed_b, 1

## Same code as signed_c, but the hand-written CFI never says that the return
## address is signed.
  .globl unsigned_c
  .type unsigned_c, %function
unsigned_c:
  .cfi_startproc
  paciasp
  stp x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  mov w0, #2
  bl callee
  ldp x29, x30, [sp], #16
  .cfi_def_cfa_offset 0
  autiasp
  .cfi_restore w30
  .cfi_restore w29
  ret
  .cfi_endproc
  .size unsigned_c, .-unsigned_c

  SIGNED signed_c, 2

## Identical code entered with an unsigned and a signed return address, like
## the cold parts of a function without and with pac-ret.
  .globl entry_unsigned
  .type entry_unsigned, %function
entry_unsigned:
  .cfi_startproc
  mov w0, #3
  bl callee
  brk #1
  .cfi_endproc
  .size entry_unsigned, .-entry_unsigned

  .globl entry_signed
  .type entry_signed, %function
entry_signed:
  .cfi_startproc
  .cfi_negate_ra_state
  mov w0, #3
  bl callee
  brk #1
  .cfi_endproc
  .size entry_signed, .-entry_signed

## Both functions sign the return address, but at different instructions.
  .globl ra_early
  .type ra_early, %function
ra_early:
  .cfi_startproc
  .cfi_negate_ra_state
  mov w0, #4
  bl callee
  brk #1
  .cfi_endproc
  .size ra_early, .-ra_early

  .globl ra_late
  .type ra_late, %function
ra_late:
  .cfi_startproc
  mov w0, #4
  .cfi_negate_ra_state
  bl callee
  brk #1
  .cfi_endproc
  .size ra_late, .-ra_late
