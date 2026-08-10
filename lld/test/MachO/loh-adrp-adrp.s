# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 %t.o -o %t
# RUN: llvm-objdump -d --macho %t | FileCheck %s

# CHECK-LABEL: _main:
## Valid
# CHECK-NEXT: adrp x0
# CHECK-NEXT: nop
## Mismatched registers
# CHECK-NEXT: adrp x1
# CHECK-NEXT: adrp x2
## Not on the same page
# CHECK-NEXT: adrp x3
# CHECK-NEXT: adrp x3
## Not an adrp instruction (invalid)
# CHECK-NEXT: nop
# CHECK-NEXT: adrp x4
## Other relaxations take precedence over AdrpAdrp
# CHECK-NEXT: adr x6
# CHECK-NEXT: nop
# CHECK-NEXT: adr x6
# CHECK-NEXT: nop

.text
.align 2

.globl _main
_main:
L1:
  adrp x0, _foo@PAGE
L2:
  adrp x0, _bar@PAGE
L3:
  adrp x1, _foo@PAGE
L4:
  adrp x2, _bar@PAGE
L5:
  adrp x3, _foo@PAGE
L6:
  adrp x3, _baz@PAGE
L7:
  nop
L8:
  adrp x4, _baz@PAGE
L9:
  adrp x5, _foo@PAGE
L10:
  add  x6, x5, _foo@PAGEOFF
L11:
  adrp x5, _bar@PAGE
L12:
  add  x6, x5, _bar@PAGEOFF

.data
.align 12
_foo:
  .byte 0
_bar:
  .byte 0
.space 4094
_baz:
  .byte 0

.loh AdrpAdrp L1, L2
.loh AdrpAdrp L3, L4
.loh AdrpAdrp L5, L6
.loh AdrpAdrp L7, L8
.loh AdrpAdrp L9, L11
.loh AdrpAdd  L9, L10
.loh AdrpAdd  L11, L12
