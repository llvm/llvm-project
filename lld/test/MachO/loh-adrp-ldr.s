# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 %t.o -o %t
# RUN: llvm-objdump -d --macho %t | FileCheck %s

.text
.align 2
_before_far:
 .space 1048576

.align 2
_before_near:
  .quad 0

.globl _main
# CHECK-LABEL: _main:
_main:
## Out of range, before
L1:  adrp  x0, _before_far@PAGE
L2:  ldr   x0, [x0, _before_far@PAGEOFF]
# CHECK-NEXT: adrp x0
# CHECK-NEXT: ldr x0

## In range, before
L3:  adrp  x1, _before_near@PAGE
L4:  ldr   x1, [x1, _before_near@PAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x1, #-20

## Registers don't match (invalid input)
L5:  adrp  x2, _before_near@PAGE
L6:  ldr   x3, [x3, _before_near@PAGEOFF]
# CHECK-NEXT: adrp x2
# CHECK-NEXT: ldr x3

## Targets don't match (invalid input)
L7:  adrp  x4, _before_near@PAGE
L8:  ldr   x4, [x4, _after_near@PAGEOFF]
# CHECK-NEXT: adrp x4
# CHECK-NEXT: ldr x4

## Not an adrp instruction
L9:  udf   0
L10: ldr   x5, [x5, _after_near@PAGEOFF]
# CHECK-NEXT: udf
# CHECK-NEXT: ldr x5

## Not an ldr with an immediate offset
L11: adrp  x6, _after_near@PAGE
L12: ldr   x6, 0
# CHECK-NEXT: adrp x6
# CHECK-NEXT: ldr x6, #0

## Byte load, unsupported
L15: adrp  x8, _after_near@PAGE
L16: ldr   b8, [x8, _after_near@PAGEOFF]
# CHECK-NEXT: adrp x8
# CHECK-NEXT: ldr b8

## Halfword load, unsupported
L17: adrp  x9, _after_near@PAGE
L18: ldr   h9, [x9, _after_near@PAGEOFF]
# CHECK-NEXT: adrp x9
# CHECK-NEXT: ldr h9

## Word load
L19: adrp  x10, _after_near@PAGE
L20: ldr   w10, [x10, _after_near@PAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr w10, _after_near

## With addend
L21: adrp  x11, _after_near@PAGE + 8
L22: ldr   x11, [x11, _after_near@PAGEOFF + 8]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x11

## Signed 32-bit read from 16-bit value, unsupported
L23: adrp  x12, _after_near@PAGE
L24: ldrsb w12, [x12, _after_near@PAGEOFF]
# CHECK-NEXT: adrp x12
# CHECK-NEXT: ldrsb w12

## 64-bit load from signed 32-bit value
L25: adrp  x13, _after_near@PAGE
L26: ldrsw x13, [x13, _after_near@PAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldrsw x13, _after_near

## Single precision FP read
L27: adrp  x14, _after_near@PAGE
L28: ldr   s0, [x14, _after_near@PAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr s0, _after_near

## Double precision FP read
L29: adrp  x15, _after_near@PAGE
L30: ldr   d0, [x15, _after_near@PAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr d0, _after_near

## Quad precision FP read
L31: adrp  x16, _after_near@PAGE
L32: ldr   q0, [x16, _after_near@PAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr q0, _after_near

## Out of range, after
L33: adrp  x17, _after_far@PAGE
L34: ldr   x17, [x17, _after_far@PAGEOFF]
# CHECK-NEXT: adrp x17
# CHECK-NEXT: ldr x17

.data
.align 4
_after_near:
  .quad 0
  .quad 0
.space 1048576

_after_far:
  .quad 0

.loh AdrpLdr L1, L2
.loh AdrpLdr L3, L4
.loh AdrpLdr L5, L6
.loh AdrpLdr L7, L8
.loh AdrpLdr L9, L10
.loh AdrpLdr L11, L12
.loh AdrpLdr L15, L16
.loh AdrpLdr L17, L18
.loh AdrpLdr L19, L20
.loh AdrpLdr L21, L22
.loh AdrpLdr L23, L24
.loh AdrpLdr L25, L26
.loh AdrpLdr L27, L28
.loh AdrpLdr L29, L30
.loh AdrpLdr L31, L32
.loh AdrpLdr L33, L34
