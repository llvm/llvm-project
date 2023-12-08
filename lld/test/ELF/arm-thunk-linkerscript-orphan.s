// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       .text_low 0x100000 : { *(.text_low) } \
// RUN:       .text_high 0x2000000 : { *(.text_high) } \
// RUN:       .data : { *(.data) } \
// RUN:       }" > %t.script
// RUN: ld.lld --script %t.script %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d %t2 | FileCheck %s
 .syntax unified
 .section .text_low, "ax", %progbits
 .thumb
 .globl _start
_start: bx lr
 .globl low_target
 .type low_target, %function
low_target:
 bl high_target
 bl orphan_target
// CHECK: Disassembly of section .text_low:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:   100000:        4770    bx      lr
// CHECK: <low_target>:
// CHECK-NEXT:   100002:        f000 f803       bl      0x10000c <__Thumbv7ABSLongThunk_high_target>
// CHECK-NEXT:   100006:        f000 f806       bl      0x100016 <__Thumbv7ABSLongThunk_orphan_target>
// CHECK: <__Thumbv7ABSLongThunk_high_target>:
// CHECK-NEXT:   10000c:        f240 0c01       movw    r12, #1
// CHECK-NEXT:   100010:        f2c0 2c00       movt    r12, #512
// CHECK-NEXT:   100014:        4760    bx      r12
// CHECK: <__Thumbv7ABSLongThunk_orphan_target>:
// CHECK-NEXT:   100016:        f240 0c15       movw    r12, #21
// CHECK-NEXT:   10001a:        f2c0 2c00       movt    r12, #512
// CHECK-NEXT:   10001e:        4760    bx      r12
  .section .text_high, "ax", %progbits
 .thumb
 .globl high_target
 .type high_target, %function
high_target:
 bl low_target
 bl orphan_target
// CHECK: Disassembly of section .text_high:
// CHECK-EMPTY:
// CHECK-NEXT: <high_target>:
// CHECK-NEXT:  2000000:        f000 f802       bl      0x2000008 <__Thumbv7ABSLongThunk_low_target>
// CHECK-NEXT:  2000004:        f000 f806       bl      0x2000014 <orphan_target>
// CHECK: <__Thumbv7ABSLongThunk_low_target>:
// CHECK-NEXT:  2000008:        f240 0c03       movw    r12, #3
// CHECK-NEXT:  200000c:        f2c0 0c10       movt    r12, #16
// CHECK-NEXT:  2000010:        4760    bx      r12

 .section orphan, "ax", %progbits
 .thumb
 .globl orphan_target
 .type orphan_target, %function
orphan_target:
 bl low_target
 bl high_target
// CHECK: Disassembly of section orphan:
// CHECK-EMPTY:
// CHECK-NEXT: <orphan_target>:
// CHECK-NEXT:  2000014:        f7ff fff8       bl      0x2000008 <__Thumbv7ABSLongThunk_low_target>
// CHECK-NEXT:  2000018:        f7ff fff2       bl      0x2000000 <high_target>

 .data
 .word 10
