// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:       .text_low 0x2000: { *(.text_low) } \
// RUN:       .text_high 0x8002000 : { *(.text_high) } \
// RUN:       } " > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s

// Check that we have the out of branch range calculation right. The immediate
// field is signed so we have a slightly higher negative displacement.
 .section .text_low, "ax", %progbits
 .globl _start
 .type _start, %function
_start:
 // Need thunk to high_target@plt
 bl high_target
 // Need thunk to .text_high+4
 bl .text_high+4
 ret

 .section .text_high, "ax", %progbits
 .globl high_target
 .type high_target, %function
high_target:
 // No Thunk needed as we are within signed immediate range
 bl _start
 ret

// CHECK: Disassembly of section .text_low:
// CHECK-EMPTY:
// CHECK-NEXT: _start:
// CHECK-NEXT:     2000:       bl      #0x10 <__AArch64AbsLongThunk_high_target>
// CHECK-NEXT:     2004:       bl      #0x1c <__AArch64AbsLongThunk_>
// CHECK-NEXT:                 ret
// CHECK: __AArch64AbsLongThunk_high_target:
// CHECK-NEXT:     2010:       ldr     x16, #0x8
// CHECK-NEXT:                 br      x16
// CHECK: $d:
// CHECK-NEXT:     2018:       00 20 00 08     .word   0x08002000
// CHECK-NEXT:     201c:       00 00 00 00     .word   0x00000000
// CHECK:      __AArch64AbsLongThunk_:
// CHECK-NEXT:     2020:       ldr x16, #0x8
// CHECK-NEXT:     2024:       br x16
// CHECK:      $d:
// CHECK-NEXT:     2028:       04 20 00 08     .word   0x08002004
// CHECK-NEXT:     202c:       00 00 00 00     .word   0x00000000
// CHECK: Disassembly of section .text_high:
// CHECK-EMPTY:
// CHECK-NEXT: high_target:
// CHECK-NEXT:  8002000:       bl      #-0x8000000 <_start>
// CHECK-NEXT:                 ret
