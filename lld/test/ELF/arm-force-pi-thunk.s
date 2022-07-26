// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       . = SIZEOF_HEADERS; \
// RUN:       .text_low : { *(.text_low) *(.text_low2) } \
// RUN:       .text_high 0x2000000 : { *(.text_high) *(.text_high2) } \
// RUN:       } " > %t.script
// RUN: ld.lld --pic-veneer --no-rosegment --script %t.script %t -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s

// Test that we can force generation of position independent thunks even when
// inputs are not pic.

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
// CHECK-NEXT:       96:        f000 f803       bl      0xa0 <__ThumbV7PILongThunk_high_target>
// CHECK-NEXT:       9a:        f000 f807       bl      0xac <__ThumbV7PILongThunk_high_target2>
// CHECK-NEXT:       9e:        d4d4 
// CHECK: <__ThumbV7PILongThunk_high_target>:
// CHECK-NEXT:       a0:        f64f 7c55       movw    r12, #65365
// CHECK-NEXT:       a4:        f2c0 1cff       movt    r12, #511
// CHECK-NEXT:       a8:        44fc    add     r12, pc
// CHECK-NEXT:       aa:        4760    bx      r12
// CHECK: <__ThumbV7PILongThunk_high_target2>:
// CHECK-NEXT:       ac:        f64f 7c69       movw    r12, #65385
// CHECK-NEXT:       b0:        f2c0 1cff       movt    r12, #511
// CHECK-NEXT:       b4:        44fc    add     r12, pc
// CHECK-NEXT:       b6:        4760    bx      r12
// CHECK: <low_target2>:
// CHECK-NEXT:       b8:        f7ff fff2       bl      0xa0 <__ThumbV7PILongThunk_high_target>
// CHECK-NEXT:       bc:        f7ff fff6       bl      0xac <__ThumbV7PILongThunk_high_target2>


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
// CHECK-NEXT:  2000000:        f000 f802       bl      0x2000008 <__ThumbV7PILongThunk_low_target>
// CHECK-NEXT:  2000004:        f000 f806       bl      0x2000014 <__ThumbV7PILongThunk_low_target2>
// CHECK: <__ThumbV7PILongThunk_low_target>:
// CHECK-NEXT:  2000008:        f240 0c83       movw    r12, #131
// CHECK-NEXT:  200000c:        f6cf 6c00       movt    r12, #65024
// CHECK-NEXT:  2000010:        44fc    add     r12, pc
// CHECK-NEXT:  2000012:        4760    bx      r12
// CHECK: <__ThumbV7PILongThunk_low_target2>:
// CHECK-NEXT:  2000014:        f240 0c99       movw    r12, #153
// CHECK-NEXT:  2000018:        f6cf 6c00       movt    r12, #65024
// CHECK-NEXT:  200001c:        44fc    add     r12, pc
// CHECK-NEXT:  200001e:        4760    bx      r12
// CHECK: <high_target2>:
// CHECK-NEXT:  2000020:        f7ff fff2       bl      0x2000008 <__ThumbV7PILongThunk_low_target>
// CHECK-NEXT:  2000024:        f7ff fff6       bl      0x2000014 <__ThumbV7PILongThunk_low_target2>
