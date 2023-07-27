// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       . = SIZEOF_HEADERS; \
// RUN:       .text_low : { *(.text_low) *(.text_low2) . = . + 0x2000000 ; *(.text_high) *(.text_high2) } \
// RUN:       } " > %t.script
// RUN: ld.lld --no-rosegment --script %t.script %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x94 --stop-address=0xbc | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x20000bc --stop-address=0x20000de | FileCheck --check-prefix=CHECK2 %s

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7aeb-none-linux-gnueabi -mcpu=cortex-a8 %s -o %t
// RUN: ld.lld --no-rosegment --script %t.script %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x94 --stop-address=0xbc | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x20000bc --stop-address=0x20000de | FileCheck --check-prefix=CHECK2 %s
// RUN: ld.lld --be8 --no-rosegment --script %t.script %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x94 --stop-address=0xbc | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x20000bc --stop-address=0x20000de | FileCheck --check-prefix=CHECK2 %s

// Test that range extension thunks can handle location expressions within
// a Section Description
 .syntax unified
 .section .text_low, "ax", %progbits
 .thumb
 .globl _start
_start: bx lr
 .globl low_target
 .type low_target, %function
low_target:
 bl high_target
 bl high_target2

 .section .text_low2, "ax", %progbits
 .thumb
 .globl low_target2
 .type low_target2, %function
low_target2:
 bl high_target
 bl high_target2
// CHECK1: Disassembly of section .text_low:
// CHECK1-EMPTY:
// CHECK1-NEXT: <_start>:
// CHECK1-NEXT:       94:       4770    bx      lr
// CHECK1: <low_target>:
// CHECK1-NEXT:       96:       f000 f803       bl      0xa0 <__Thumbv7ABSLongThunk_high_target>
// CHECK1-NEXT:       9a:       f000 f806       bl      0xaa <__Thumbv7ABSLongThunk_high_target2>
// CHECK1: <__Thumbv7ABSLongThunk_high_target>:
// CHECK1-NEXT:       a0:       f240 0cbd       movw    r12, #189
// CHECK1-NEXT:       a4:       f2c0 2c00       movt    r12, #512
// CHECK1-NEXT:       a8:       4760    bx      r12
// CHECK1: <__Thumbv7ABSLongThunk_high_target2>:
// CHECK1-NEXT:       aa:       f240 0cd9       movw    r12, #217
// CHECK1-NEXT:       ae:       f2c0 2c00       movt    r12, #512
// CHECK1-NEXT:       b2:       4760    bx      r12
// CHECK1: <low_target2>:
// CHECK1-NEXT:       b4:       f7ff fff4       bl      0xa0 <__Thumbv7ABSLongThunk_high_target>
// CHECK1-NEXT:       b8:       f7ff fff7       bl      0xaa <__Thumbv7ABSLongThunk_high_target2>

 .section .text_high, "ax", %progbits
 .thumb
 .globl high_target
 .type high_target, %function
high_target:
 bl low_target
 bl low_target2

 .section .text_high2, "ax", %progbits
 .thumb
 .globl high_target2
 .type high_target2, %function
high_target2:
 bl low_target
 bl low_target2

// CHECK2: <high_target>:
// CHECK2-NEXT:  20000bc:       f000 f802       bl      0x20000c4 <__Thumbv7ABSLongThunk_low_target>
// CHECK2-NEXT:  20000c0:       f000 f805       bl      0x20000ce <__Thumbv7ABSLongThunk_low_target2>
// CHECK2: <__Thumbv7ABSLongThunk_low_target>:
// CHECK2-NEXT:  20000c4:       f240 0c97       movw    r12, #151
// CHECK2-NEXT:  20000c8:       f2c0 0c00       movt    r12, #0
// CHECK2-NEXT:  20000cc:       4760    bx      r12
// CHECK2: <__Thumbv7ABSLongThunk_low_target2>:
// CHECK2-NEXT:  20000ce:       f240 0cb5       movw    r12, #181
// CHECK2-NEXT:  20000d2:       f2c0 0c00       movt    r12, #0
// CHECK2-NEXT:  20000d6:       4760    bx      r12
// CHECK2: <high_target2>:
// CHECK2-NEXT:  20000d8:       f7ff fff4       bl      0x20000c4 <__Thumbv7ABSLongThunk_low_target>
// CHECK2-NEXT:  20000dc:       f7ff fff7       bl      0x20000ce <__Thumbv7ABSLongThunk_low_target2>
