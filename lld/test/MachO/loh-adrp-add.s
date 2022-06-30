# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 %t.o -o %t
# RUN: llvm-objdump -d --macho %t | FileCheck %s

# CHECK-LABEL: _main:
## Out of range, before
# CHECK-NEXT: adrp x0
# CHECK-NEXT: add x0, x0
## In range, before
# CHECK-NEXT: adr x1
# CHECK-NEXT: nop
## Registers don't match (invalid input)
# CHECK-NEXT: adrp x2
# CHECK-NEXT: add x0
## Targets don't match (invalid input)
# CHECK-NEXT: adrp x3
# CHECK-NEXT: add x3
## Not an adrp instruction (invalid input)
# CHECK-NEXT: nop
# CHECK-NEXT: add x4
## In range, after
# CHECK-NEXT: adr x5
# CHECK-NEXT: nop
## In range, add's destination register is not the same as its source
# CHECK-NEXT: adr x7
# CHECK-NEXT: nop
## Valid, non-adjacent instructions - start
# CHECK-NEXT: adr x8
## Out of range, after
# CHECK-NEXT: adrp x9
# CHECK-NEXT: add x9, x9
## Valid, non-adjacent instructions - end
# CHECK-NEXT: nop

.text
.align 2
_before_far:
  .space 1048576

_before_near:
  nop

.globl _main
_main:
L1:
  adrp x0, _before_far@PAGE
L2:
  add  x0, x0, _before_far@PAGEOFF
L3:
  adrp x1, _before_near@PAGE
L4:
  add  x1, x1, _before_near@PAGEOFF
L5:
  adrp x2, _before_near@PAGE
L6:
  add  x0, x0, _before_near@PAGEOFF
L7:
  adrp x3, _before_near@PAGE
L8:
  add  x3, x3, _after_near@PAGEOFF
L9:
  nop
L10:
  add  x4, x4, _after_near@PAGEOFF
L11:
  adrp x5, _after_near@PAGE
L12:
  add  x5, x5, _after_near@PAGEOFF
L13:
  adrp x6, _after_near@PAGE
L14:
  add  x7, x6, _after_near@PAGEOFF
L15:
  adrp x8, _after_near@PAGE
L16:
  adrp x9, _after_far@PAGE
L17:
  add  x9, x9, _after_far@PAGEOFF
L18:
  add  x8, x8, _after_near@PAGEOFF

_after_near:
  .space 1048576

_after_far:
  nop

.loh AdrpAdd L1, L2
.loh AdrpAdd L3, L4
.loh AdrpAdd L5, L6
.loh AdrpAdd L7, L8
.loh AdrpAdd L9, L10
.loh AdrpAdd L11, L12
.loh AdrpAdd L13, L14
.loh AdrpAdd L15, L18
.loh AdrpAdd L16, L17
