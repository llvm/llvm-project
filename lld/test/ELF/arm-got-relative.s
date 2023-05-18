// REQUIRES: arm
// RUN: llvm-mc -position-independent -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -shared -o %t
// RUN: llvm-readobj -S --symbols --dyn-relocations %t | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d --triple=armv7a-none-linux-gnueabi %t | FileCheck --check-prefix=CODE %s
 .syntax unified
 .text
 .globl _start
 .align 2
_start:
 .type _start, %function
 ldr r3, .LGOT
 ldr r2, .LGOT+4
.LPIC:
 add r0, pc, r3
 bx lr
 .align 2
.LGOT:
 // gas implicitly uses (R_ARM_BASE_PREL) for _GLOBAL_OFFSET_TABLE_ in PIC
 // llvm-mc generates R_ARM_REL32, this will need updating when MC changes
 .word _GLOBAL_OFFSET_TABLE_ - (.LPIC+8)
 .word function(GOT)

 .globl function
 .align 2
function:
 .type function, %function
 bx lr

// CHECK: Dynamic Relocations {
// CHECK-NEXT:  0x2020C R_ARM_GLOB_DAT function

// CHECK: Name: _GLOBAL_OFFSET_TABLE_
// CHECK-NEXT:    Value: 0x2020C
// CHECK-NEXT:    Size:
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .got

// CODE: Disassembly of section .text:
// CODE-EMPTY:
// CODE-NEXT: <_start>:
// CODE-NEXT:    101a0:        e59f3008       ldr     r3, [pc, #8]
// CODE-NEXT:    101a4:        e59f2008       ldr     r2, [pc, #8]
// CODE-NEXT:    101a8:        e08f0003       add     r0, pc, r3
// CODE-NEXT:    101ac:        e12fff1e       bx      lr
// (_GLOBAL_OFFSET_TABLE_ = 0x220c) - (0x11a8 + 8) = 0x105c
// CODE-NEXT:    101b0:        5c 00 01 00
// (Got(function) - GotBase = 0x0
// CODE-NEXT:    101b4:        00 00 00 00
