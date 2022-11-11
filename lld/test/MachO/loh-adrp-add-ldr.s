# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 %t.o -o %t
# RUN: llvm-objdump --no-print-imm-hex -d --macho %t | FileCheck %s

## This is mostly a copy of loh-adrp-ldr-got-ldr.s's `local.s` test, except that Adrp+Ldr+Ldr
## triples have been changed to Adrp+Add+Ldr. The performed optimization is the same.
.text
.align 2
.globl _main
_main:

### Transformation to a literal LDR
## Basic case
L1: adrp x0, _close@PAGE
L2: add  x1, x0, _close@PAGEOFF
L3: ldr  x2, [x1]
# CHECK-LABEL: _main:
# CHECK-NEXT: nop
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x2

## Load with offset
L4: adrp x0, _close@PAGE
L5: add x1, x0, _close@PAGEOFF
L6: ldr  x2, [x1, #8]
# CHECK-NEXT: nop
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x2

## 32 bit load
L7: adrp x0, _close@PAGE
L8: add  x1, x0, _close@PAGEOFF
L9: ldr  w1, [x1]
# CHECK-NEXT: nop
# CHECK-NEXT: nop
# CHECK-NEXT: ldr w1, _close

## Floating point
L10: adrp x0, _close@PAGE
L11: add  x1, x0, _close@PAGEOFF
L12: ldr  s1, [x1]
# CHECK-NEXT: nop
# CHECK-NEXT: nop
# CHECK-NEXT: ldr s1, _close

L13: adrp x0, _close@PAGE
L14: add  x1, x0, _close@PAGEOFF
L15: ldr  d1, [x1, #8]
# CHECK-NEXT: nop
# CHECK-NEXT: nop
# CHECK-NEXT: ldr d1, _close8

L16: adrp x0, _close@PAGE
L17: add  x1, x0, _close@PAGEOFF
L18: ldr  q0, [x1]
# CHECK-NEXT: nop
# CHECK-NEXT: nop
# CHECK-NEXT: ldr q0, _close


### Transformation to ADR+LDR
## 1 byte floating point load
L19: adrp x0, _close@PAGE
L20: add  x1, x0, _close@PAGEOFF
L21: ldr  b2, [x1]
# CHECK-NEXT: adr x1
# CHECK-NEXT: nop
# CHECK-NEXT: ldr b2, [x1]

## 1 byte GPR load, zero extend
L22: adrp x0, _close@PAGE
L23: add  x1, x0, _close@PAGEOFF
L24: ldrb w2, [x1]
# CHECK-NEXT: adr x1
# CHECK-NEXT: nop
# CHECK-NEXT: ldrb w2, [x1]

## 1 byte GPR load, sign extend
L25: adrp  x0, _close@PAGE
L26: add   x1, x0, _close@PAGEOFF
L27: ldrsb x2, [x1]
# CHECK-NEXT: adr x1
# CHECK-NEXT: nop
# CHECK-NEXT: ldrsb x2, [x1]

## Unaligned
L28: adrp x0, _unaligned@PAGE
L29: add  x1, x0, _unaligned@PAGEOFF
L30: ldr  x2, [x1]
# CHECK-NEXT: adr x1
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x2, [x1]


### Transformation to ADRP + immediate LDR
## Basic test: target is far
L31: adrp x0, _far@PAGE
L32: add  x1, x0, _far@PAGEOFF
L33: ldr  x2, [x1]
# CHECK-NEXT: adrp x0
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x2

## With offset
L34: adrp x0, _far@PAGE
L35: add  x1, x0, _far@PAGEOFF
L36: ldr  x2, [x1, #8]
# CHECK-NEXT: adrp x0
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x2

### No changes
## Far and unaligned
L37: adrp x0, _far_unaligned@PAGE
L38: add  x1, x0, _far_unaligned@PAGEOFF
L39: ldr  x2, [x1]
# CHECK-NEXT: adrp x0
# CHECK-NEXT: add x1, x0
# CHECK-NEXT: ldr x2, [x1]

## Far with large offset (_far_offset@PAGE + #255 > 4095)
L40: adrp x0, _far_offset@PAGE
L41: add  x1, x0, _far_offset@PAGEOFF
L42: ldrb w2, [x1, #255]
# CHECK-NEXT: adrp x0
# CHECK-NEXT: add x1, x0
# CHECK-NEXT: ldrb w2, [x1, #255]

### Invalid inputs; the instructions should be left untouched.
## Registers don't match
L43: adrp x0, _far@PAGE
L44: add  x1, x0, _far@PAGEOFF
L45: ldr  x2, [x2]
# CHECK-NEXT: adrp x0
# CHECK-NEXT: add x1, x0
# CHECK-NEXT: ldr x2, [x2]

## Targets don't match
L46: adrp x0, _close@PAGE
L47: add  x1, x0, _close8@PAGEOFF
L48: ldr  x2, [x1]
# CHECK-NEXT: adrp x0
# CHECK-NEXT: add x1, x0
# CHECK-NEXT: ldr x2, [x1]

.data
.align 4
    .quad 0
_close:
    .quad 0
_close8:
    .quad 0
    .byte 0
_unaligned:
    .quad 0

.space 1048576
.align 12
    .quad 0
_far:
     .quad 0
    .byte 0
_far_unaligned:
    .quad 0
.space 4000
_far_offset:
    .byte 0

.loh AdrpAddLdr L1, L2, L3
.loh AdrpAddLdr L4, L5, L6
.loh AdrpAddLdr L7, L8, L9
.loh AdrpAddLdr L10, L11, L12
.loh AdrpAddLdr L13, L14, L15
.loh AdrpAddLdr L16, L17, L18
.loh AdrpAddLdr L19, L20, L21
.loh AdrpAddLdr L22, L23, L24
.loh AdrpAddLdr L25, L26, L27
.loh AdrpAddLdr L28, L29, L30
.loh AdrpAddLdr L31, L32, L33
.loh AdrpAddLdr L34, L35, L36
.loh AdrpAddLdr L37, L38, L39
.loh AdrpAddLdr L40, L41, L42
.loh AdrpAddLdr L43, L44, L45
