// REQUIRES: arm
// RUN: llvm-mc -position-independent -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -shared -o %t
// RUN: llvm-readobj -s -symbols -dyn-relocations %t | FileCheck %s
// RUN: llvm-objdump -d -triple=armv7a-none-linux-gnueabi %t | FileCheck -check-prefix=CODE %s
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
 // gas implicitly uses (GOT_PREL) for _GLOBAL_OFFSET_TABLE_ in PIC
 // llvm-mc needs the (GOT_PREL) suffix or it generates R_ARM_REL32
 .word _GLOBAL_OFFSET_TABLE_(GOT_PREL) - (.LPIC+8)
 .word function(GOT)

 .globl function
 .align 2
function:
 .type function, %function
 bx lr

// CHECK: Dynamic Relocations {
// CHECK-NEXT:  0x204C R_ARM_GLOB_DAT function 0x0

// CHECK: Name: _GLOBAL_OFFSET_TABLE_
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size:
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: Absolute

// CODE: Disassembly of section .text:
// CODE-NEXT: _start:
// CODE-NEXT:    1000:        08 30 9f e5    ldr     r3, [pc, #8]
// CODE-NEXT:    1004:        08 20 9f e5    ldr     r2, [pc, #8]
// CODE-NEXT:    1008:        03 00 8f e0    add     r0, pc, r3
// CODE-NEXT:    100c:        1e ff 2f e1    bx      lr
// CODE:$d.1:
// (_GLOBAL_OFFSET_TABLE_ = 0x2048) - (0x1008 + 8) 0x1038
// CODE-NEXT:    1010:        38 10 00 00
// (Got(function) - GotBase = 0x4
// CODE-NEXT:    1014:        04 00 00 00
