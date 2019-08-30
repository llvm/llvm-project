// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj -S %t | FileCheck %s
// RUN: llvm-readobj -S --symbols %t | FileCheck -check-prefix=SYMBOLS %s
// RUN: llvm-objdump -d --no-show-raw-insn -triple=armv7a-none-linux-gnueabi %t | FileCheck -check-prefix=CODE %s

// Test the R_ARM_GOT_PREL relocation
 .syntax unified
 .text
 .globl _start
 .align 2
_start:
 ldr     r0, .LCPI0_0
.LPC0_0:
 ldr     r0, [pc, r0]
 ldr     r0, [r0]
 bx      lr
.LCPI0_0:
.Ltmp0:
 // Generate R_ARM_GOT_PREL
 .long   val(GOT_PREL)-((.LPC0_0+8)-.Ltmp0)

 .data
 .type   val,%object
 .globl  val
 .align  2
val:
 .long   10
 .size   val, 4

// CHECK: Section {
// CHECK:    Name: .got
// CHECK-NEXT:    Type: SHT_PROGBITS
// CHECK-NEXT:      Flags [
// CHECK-NEXT:      SHF_ALLOC
// CHECK-NEXT:      SHF_WRITE
// CHECK-NEXT:    ]
// CHECK-NEXT:    Address: 0x12128
// CHECK-NEXT:    Offset:
// CHECK-NEXT:    Size: 4
// CHECK-NEXT:    Link:
// CHECK-NEXT:    Info:
// CHECK-NEXT:    AddressAlignment: 4
// CHECK-NEXT:    EntrySize:

// SYMBOLS:    Name: val
// SYMBOLS-NEXT:    Value: 0x1312C
// SYMBOLS-NEXT:    Size: 4
// SYMBOLS-NEXT:    Binding: Global
// SYMBOLS-NEXT:    Type: Object
// SYMBOLS-NEXT:    Other:
// SYMBOLS-NEXT:    Section: .data

// CODE: Disassembly of section .text:
// CODE-EMPTY:
// CODE-NEXT: _start:
// CODE-NEXT:   11114:       ldr     r0, [pc, #8]
// CODE-NEXT:   11118:       ldr     r0, [pc, r0]
// CODE-NEXT:   1111c:       ldr     r0, [r0]
// CODE-NEXT:   11120:       bx      lr
// CODE: $d.1:
// 0x11124 + 0x1008 + 8 = 0x12128 = .got
// CODE-NEXT:   11124:       08 10 00 00
