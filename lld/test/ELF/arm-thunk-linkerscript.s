// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       . = SIZEOF_HEADERS; \
// RUN:       .text_low : { *(.text_low) *(.text_low2) } \
// RUN:       .text_high 0x2000000 : { *(.text_high) *(.text_high2) } \
// RUN:       } " > %t.script
// RUN: ld.lld --no-rosegment --script %t.script %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d %t2 | FileCheck %s
// Simple test that we can support range extension thunks with linker scripts
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

// CHECK: Disassembly of section .text_low:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:       94:        4770    bx      lr
// CHECK: <low_target>:
// CHECK-NEXT:       96:        f000 f803       bl      0xa0 <__Thumbv7ABSLongThunk_high_target>
// CHECK-NEXT:       9a:        f000 f806       bl      0xaa <__Thumbv7ABSLongThunk_high_target2>
// CHECK: <__Thumbv7ABSLongThunk_high_target>:
// CHECK-NEXT:       a0:        f240 0c01       movw    r12, #1
// CHECK-NEXT:       a4:        f2c0 2c00       movt    r12, #512
// CHECK-NEXT:       a8:        4760    bx      r12
// CHECK: <__Thumbv7ABSLongThunk_high_target2>:
// CHECK-NEXT:       aa:        f240 0c1d       movw    r12, #29
// CHECK-NEXT:       ae:        f2c0 2c00       movt    r12, #512
// CHECK-NEXT:       b2:        4760    bx      r12
// CHECK: <low_target2>:
// CHECK-NEXT:       b4:        f7ff fff4       bl      0xa0 <__Thumbv7ABSLongThunk_high_target>
// CHECK-NEXT:       b8:        f7ff fff7       bl      0xaa <__Thumbv7ABSLongThunk_high_target2>

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

// CHECK: Disassembly of section .text_high:
// CHECK-EMPTY:
// CHECK-NEXT: <high_target>:
// CHECK-NEXT:  2000000:        f000 f802       bl      0x2000008 <__Thumbv7ABSLongThunk_low_target>
// CHECK-NEXT:  2000004:        f000 f805       bl      0x2000012 <__Thumbv7ABSLongThunk_low_target2>
// CHECK: <__Thumbv7ABSLongThunk_low_target>:
// CHECK-NEXT:  2000008:        f240 0c97       movw    r12, #151
// CHECK-NEXT:  200000c:        f2c0 0c00       movt    r12, #0
// CHECK-NEXT:  2000010:        4760    bx      r12
// CHECK: <__Thumbv7ABSLongThunk_low_target2>:
// CHECK-NEXT:  2000012:        f240 0cb5       movw    r12, #181
// CHECK-NEXT:  2000016:        f2c0 0c00       movt    r12, #0
// CHECK-NEXT:  200001a:        4760    bx      r12
// CHECK: <high_target2>:
// CHECK-NEXT:  200001c:        f7ff fff4       bl      0x2000008 <__Thumbv7ABSLongThunk_low_target>
// CHECK-NEXT:  2000020:        f7ff fff7       bl      0x2000012 <__Thumbv7ABSLongThunk_low_target2>
