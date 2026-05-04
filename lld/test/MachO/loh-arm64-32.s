# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %s -o %t.o
# RUN: %lld-watchos -U _external %t.o -o %t
# RUN: llvm-objdump -d --macho %t | FileCheck %s

.text
.align 2
.globl _foo
_foo:
    ret
.globl _bar
_bar:
    ret

.globl _main
_main:
# CHECK-LABEL: _main:

L1: adrp x0, _foo@PAGE
L2: add  x0, x0, _foo@PAGEOFF
# CHECK-NEXT: adr x0
# CHECK-NEXT: nop

L3: adrp x0, _ptr@PAGE
L4: add  x1, x0, _ptr@PAGEOFF
L5: ldr  x2, [x1]
# CHECK-NEXT: nop
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x2

L6: adrp x0, _foo@PAGE
L7: adrp x0, _bar@PAGE
# CHECK-NEXT: adrp x0
# CHECK-NEXT: nop

L8: adrp x0, _ptr@PAGE
L9: ldr  x0, [x0, _ptr@PAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x0

L10: adrp x0, _ptr@PAGE
L11: ldr  w0, [x0, _ptr@PAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr w0, _ptr

L12: adrp x0, _external@PAGE
L13: ldr  w1, [x0, _external@PAGEOFF]
L14: ldr  x2, [x1]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr w1, 0x{{.*}}
# CHECK-NEXT: ldr x2, [x1]

.data
.align 4
_ptr:
    .quad 0

.loh AdrpAdd L1, L2
.loh AdrpAddLdr L3, L4, L5
.loh AdrpAdrp L6, L7
.loh AdrpLdr L8, L9
.loh AdrpLdrGot L10, L11
.loh AdrpLdrGotLdr L12, L13, L14
