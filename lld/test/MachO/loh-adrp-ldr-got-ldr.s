# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/lib.s -o %t/lib.o
# RUN: %lld -arch arm64 -dylib -o %t/lib.dylib %t/lib.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/external.s -o %t/near-got.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/external.s -defsym=PADDING=1 -o %t/far-got.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/local.s -o %t/local.o
# RUN: %lld -arch arm64 %t/near-got.o %t/lib.dylib -o %t/NearGot
# RUN: %lld -arch arm64 %t/far-got.o %t/lib.dylib -o %t/FarGot
# RUN: %lld -arch arm64 %t/local.o -o %t/Local
# RUN: llvm-objdump --no-print-imm-hex -d --macho %t/NearGot | FileCheck %s -check-prefix=NEAR-GOT
# RUN: llvm-objdump --no-print-imm-hex -d --macho %t/FarGot | FileCheck %s -check-prefix=FAR-GOT
# RUN: llvm-objdump --no-print-imm-hex -d --macho %t/Local | FileCheck %s -check-prefix=LOCAL

#--- external.s
.text
.align 2
.globl _main
_main:

## Basic test
L1: adrp x0, _external@GOTPAGE
L2: ldr  x1, [x0, _external@GOTPAGEOFF]
L3: ldr  x2, [x1]
# NEAR-GOT-LABEL: _main:
# NEAR-GOT-NEXT: nop
# NEAR-GOT-NEXT: ldr x1, #{{.*}} ; literal pool symbol address: _external
# NEAR-GOT-NEXT: ldr x2, [x1]
# FAR-GOT-LABEL: _main:
# FAR-GOT-NEXT:  adrp x0
# FAR-GOT-NEXT:  ldr x1
# FAR-GOT-NEXT:  ldr x2, [x1]

## The second load has an offset
L4: adrp x0, _external@GOTPAGE
L5: ldr  x1, [x0, _external@GOTPAGEOFF]
L6: ldr  q2, [x1, #16]
# NEAR-GOT-NEXT: nop
# NEAR-GOT-NEXT: ldr x1, #{{.*}} ; literal pool symbol address: _external
# NEAR-GOT-NEXT: ldr q2, [x1, #16]
# FAR-GOT-NEXT:  adrp x0
# FAR-GOT-NEXT:  ldr x1
# FAR-GOT-NEXT:  ldr q2, [x1, #16]

### Tests for invalid inputs
.ifndef PADDING
## Registers don't match
L7: adrp x0, _external@GOTPAGE
L8: ldr  x1, [x1, _external@GOTPAGEOFF]
L9: ldr  x2, [x1]
# NEAR-GOT-NEXT: adrp x0
# NEAR-GOT-NEXT: ldr x1
# NEAR-GOT-NEXT: ldr x2, [x1]

## Registers don't match
L10: adrp x0, _external@GOTPAGE
L11: ldr  x1, [x0, _external@GOTPAGEOFF]
L12: ldr  x2, [x0]
# NEAR-GOT-NEXT: adrp x0
# NEAR-GOT-NEXT: ldr x1
# NEAR-GOT-NEXT: ldr x2, [x0]

## Not an LDR (immediate)
L13: adrp x0, _external@GOTPAGE
L14: ldr  x1, 0
L15: ldr  x2, [x1]
# NEAR-GOT-NEXT: adrp x0
# NEAR-GOT-NEXT: ldr x1
# NEAR-GOT-NEXT: ldr x2, [x1]

.loh AdrpLdrGotLdr L7, L8, L9
.loh AdrpLdrGotLdr L10, L11, L12
.loh AdrpLdrGotLdr L13, L14, L15
.endif

.loh AdrpLdrGotLdr L1, L2, L3
.loh AdrpLdrGotLdr L4, L5, L6

.ifdef PADDING
.space 1048576
.endif
.data


#--- lib.s
.data
.align 4
.globl _external
_external:
    .zero 32

#--- local.s
.text
.align 2
.globl _main
_main:

### Transformation to a literal LDR
## Basic case
L1: adrp x0, _close@GOTPAGE
L2: ldr  x1, [x0, _close@GOTPAGEOFF]
L3: ldr  x2, [x1]
# LOCAL-LABEL: _main:
# LOCAL-NEXT: nop
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr x2

## Load with offset
L4: adrp x0, _close@GOTPAGE
L5: ldr  x1, [x0, _close@GOTPAGEOFF]
L6: ldr  x2, [x1, #8]
# LOCAL-NEXT: nop
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr x2

## 32 bit load
L7: adrp x0, _close@GOTPAGE
L8: ldr  x1, [x0, _close@GOTPAGEOFF]
L9: ldr  w1, [x1]
# LOCAL-NEXT: nop
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr w1, _close

## Floating point
L10: adrp x0, _close@GOTPAGE
L11: ldr  x1, [x0, _close@GOTPAGEOFF]
L12: ldr  s1, [x1]
# LOCAL-NEXT: nop
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr s1, _close

L13: adrp x0, _close@GOTPAGE
L14: ldr  x1, [x0, _close@GOTPAGEOFF]
L15: ldr  d1, [x1, #8]
# LOCAL-NEXT: nop
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr d1, _close8

L16: adrp x0, _close@GOTPAGE
L17: ldr  x1, [x0, _close@GOTPAGEOFF]
L18: ldr  q0, [x1]
# LOCAL-NEXT: nop
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr q0, _close


### Transformation to ADR+LDR
## 1 byte floating point load
L19: adrp x0, _close@GOTPAGE
L20: ldr  x1, [x0, _close@GOTPAGEOFF]
L21: ldr  b2, [x1]
# LOCAL-NEXT: adr x1
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr b2, [x1]

## 1 byte GPR load, zero extend
L22: adrp x0, _close@GOTPAGE
L23: ldr  x1, [x0, _close@GOTPAGEOFF]
L24: ldrb w2, [x1]
# LOCAL-NEXT: adr x1
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldrb w2, [x1]

## 1 byte GPR load, sign extend
L25: adrp  x0, _close@GOTPAGE
L26: ldr   x1, [x0, _close@GOTPAGEOFF]
L27: ldrsb x2, [x1]
# LOCAL-NEXT: adr x1
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldrsb x2, [x1]

## Unaligned
L28: adrp x0, _unaligned@GOTPAGE
L29: ldr  x1, [x0, _unaligned@GOTPAGEOFF]
L30: ldr  x2, [x1]
# LOCAL-NEXT: adr x1
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr x2, [x1]


### Transformation to ADRP + immediate LDR
## Basic test: target is far
L31: adrp x0, _far@GOTPAGE
L32: ldr  x1, [x0, _far@GOTPAGEOFF]
L33: ldr  x2, [x1]
# LOCAL-NEXT: adrp x0
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr x2

## With offset
L34: adrp x0, _far@GOTPAGE
L35: ldr  x1, [x0, _far@GOTPAGEOFF]
L36: ldr  x2, [x1, #8]
# LOCAL-NEXT: adrp x0
# LOCAL-NEXT: nop
# LOCAL-NEXT: ldr x2

### No changes other than GOT relaxation
## Far and unaligned
L37: adrp x0, _far_unaligned@GOTPAGE
L38: ldr  x1, [x0, _far_unaligned@GOTPAGEOFF]
L39: ldr  x2, [x1]
# LOCAL-NEXT: adrp x0
# LOCAL-NEXT: add x1, x0
# LOCAL-NEXT: ldr x2, [x1]

## Far with large offset (_far_offset@GOTPAGEOFF + #255 > 4095)
L40: adrp x0, _far_offset@GOTPAGE
L41: ldr  x1, [x0, _far_offset@GOTPAGEOFF]
L42: ldrb w2, [x1, #255]
# LOCAL-NEXT: adrp x0
# LOCAL-NEXT: add x1, x0
# LOCAL-NEXT: ldrb w2, [x1, #255]

### Tests for invalid inputs, only GOT relaxation should happen
## Registers don't match
L43: adrp x0, _far@GOTPAGE
L44: ldr  x1, [x0, _far@GOTPAGEOFF]
L45: ldr  x2, [x2]
# LOCAL-NEXT: adrp x0
# LOCAL-NEXT: add x1, x0
# LOCAL-NEXT: ldr x2, [x2]

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


.loh AdrpLdrGotLdr L1, L2, L3
.loh AdrpLdrGotLdr L4, L5, L6
.loh AdrpLdrGotLdr L7, L8, L9
.loh AdrpLdrGotLdr L10, L11, L12
.loh AdrpLdrGotLdr L13, L14, L15
.loh AdrpLdrGotLdr L16, L17, L18
.loh AdrpLdrGotLdr L19, L20, L21
.loh AdrpLdrGotLdr L22, L23, L24
.loh AdrpLdrGotLdr L25, L26, L27
.loh AdrpLdrGotLdr L28, L29, L30
.loh AdrpLdrGotLdr L31, L32, L33
.loh AdrpLdrGotLdr L34, L35, L36
.loh AdrpLdrGotLdr L37, L38, L39
.loh AdrpLdrGotLdr L40, L41, L42
.loh AdrpLdrGotLdr L43, L44, L45
