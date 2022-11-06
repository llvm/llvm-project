// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t --shared -o %t.so
// The output file is large, most of it zeroes. We dissassemble only the
// parts we need to speed up the test and avoid a large output file
// RUN: llvm-objdump --no-print-imm-hex -d %t.so --start-address=0x1000004 --stop-address=0x100001c | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.so --start-address=0x1100008 --stop-address=0x1100022 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.so --start-address=0x1100020 --stop-address=0x1100064 --triple=armv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK3 %s

/// A branch to a Thunk that we create on pass N, can drift out of range if
/// other Thunks are added in between. In this case we must create a new Thunk
/// for the branch that is in range. We also need to make sure that if the
/// destination of the Thunk is in the PLT the new Thunk also targets the PLT
 .syntax unified
 .thumb

 .macro FUNCTION suff
 .section .text.\suff\(), "ax", %progbits
 .thumb
 .balign 0x80000
 .globl tfunc\suff\()
 .type  tfunc\suff\(), %function
tfunc\suff\():
 bx lr
 .endm

 .globl imported
 .type imported, %function
 .globl imported2
 .type imported2, %function
 .globl imported3
 .type imported3, %function
.globl imported4
 .type imported4, %function
 FUNCTION 00
 FUNCTION 01
 FUNCTION 02
 FUNCTION 03
 FUNCTION 04
 FUNCTION 05
 FUNCTION 06
 FUNCTION 07
 FUNCTION 08
 FUNCTION 09
 FUNCTION 10
 FUNCTION 11
 FUNCTION 12
 FUNCTION 13
 FUNCTION 14
 FUNCTION 15
 FUNCTION 16
 FUNCTION 17
 FUNCTION 18
 FUNCTION 19
 FUNCTION 20
 FUNCTION 21
 FUNCTION 22
 FUNCTION 23
 FUNCTION 24
 FUNCTION 25
 FUNCTION 26
 FUNCTION 27
 FUNCTION 28
 FUNCTION 29
 FUNCTION 30
 FUNCTION 31
/// Precreated Thunk Pool goes here
// CHECK1: <__ThumbV7PILongThunk_imported>:
// CHECK1-NEXT:  1000004:       f240 0c30       movw    r12, #48
// CHECK1-NEXT:  1000008:       f2c0 0c10       movt    r12, #16
// CHECK1-NEXT:  100000c:       44fc    add     r12, pc
// CHECK1-NEXT:  100000e:       4760    bx      r12
// CHECK1: <__ThumbV7PILongThunk_imported2>:
// CHECK1-NEXT:  1000010:       f240 0c34       movw    r12, #52
// CHECK1-NEXT:  1000014:       f2c0 0c10       movt    r12, #16
// CHECK1-NEXT:  1000018:       44fc    add     r12, pc
// CHECK1-NEXT:  100001a:       4760    bx      r12

 .section .text.32, "ax", %progbits
 .space 0x80000
 .section .text.33, "ax", %progbits
 .space 0x80000 - 0x14
 .section .text.34, "ax", %progbits
 /// Need a Thunk to the PLT entry, can use precreated ThunkSection
 .globl callers
 .type callers, %function
callers:
 b.w imported
 beq.w imported
 b.w imported2
// CHECK2: <__ThumbV7PILongThunk_imported>:
// CHECK2-NEXT:  1100008:       f240 0c2c       movw    r12, #44
// CHECK2-NEXT:  110000c:       f2c0 0c00       movt    r12, #0
// CHECK2-NEXT:  1100010:       44fc    add     r12, pc
// CHECK2-NEXT:  1100012:       4760    bx      r12
// CHECK2: <callers>:
// CHECK2-NEXT:  1100014:       f6ff bff6       b.w     0x1000004 <__ThumbV7PILongThunk_imported>
// CHECK2-NEXT:  1100018:       f43f aff6       beq.w   0x1100008 <__ThumbV7PILongThunk_imported>
// CHECK2-NEXT:  110001c:       f6ff bff8       b.w     0x1000010 <__ThumbV7PILongThunk_imported2>

// CHECK3: Disassembly of section .plt:
// CHECK3-EMPTY:
// CHECK3-NEXT: <$a>:
// CHECK3-NEXT:  1100020:       e52de004        str     lr, [sp, #-4]!
// CHECK3-NEXT:  1100024:       e28fe600        add     lr, pc, #0, #12
// CHECK3-NEXT:  1100028:       e28eea20        add     lr, lr, #32
// CHECK3-NEXT:  110002c:       e5bef094        ldr     pc, [lr, #148]!
// CHECK3: <$d>:
// CHECK3-NEXT:  1100030:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK3-NEXT:  1100034:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK3-NEXT:  1100038:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK3-NEXT:  110003c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK3: <$a>:
// CHECK3-NEXT:  1100040:       e28fc600        add     r12, pc, #0, #12
// CHECK3-NEXT:  1100044:       e28cca20        add     r12, r12, #32
// CHECK3-NEXT:  1100048:       e5bcf07c        ldr     pc, [r12, #124]!
// CHECK3: <$d>:
// CHECK3-NEXT:  110004c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK3: <$a>:
// CHECK3-NEXT:  1100050:       e28fc600        add     r12, pc, #0, #12
// CHECK3-NEXT:  1100054:       e28cca20        add     r12, r12, #32
// CHECK3-NEXT:  1100058:       e5bcf070        ldr     pc, [r12, #112]!
// CHECK3: <$d>:
// CHECK3-NEXT:  110005c:       d4 d4 d4 d4     .word   0xd4d4d4d4
