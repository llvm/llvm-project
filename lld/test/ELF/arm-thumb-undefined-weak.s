// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2 2>&1
// RUN: llvm-objdump -triple=thumbv7a-none-linux-gnueabi -d %t2 | FileCheck %s
// REQUIRES: arm

// Check that the ARM ABI rules for undefined weak symbols are applied.
// Branch instructions are resolved to the next instruction. Relative
// relocations are resolved to the place.

 .syntax unified

 .weak target

 .text
 .global _start
_start:
// R_ARM_THM_JUMP19
 beq.w target
// R_ARM_THM_JUMP24
 b.w target
// R_ARM_THM_CALL
 bl target
// R_ARM_THM_CALL with exchange
 blx target
// R_ARM_THM_MOVT_PREL
 movt r0, :upper16:target - .
// R_ARM_THM_MOVW_PREL_NC
 movw r0, :lower16:target - .

// CHECK: Disassembly of section .text:
// 69636 = 0x11004
// CHECK:         11000:       11 f0 02 80     beq.w   #69636
// CHECK-NEXT:    11004:       11 f0 04 b8     b.w     #69640
// CHECK-NEXT:    11008:       11 f0 06 f8     bl      #69644
// blx is transformed into bl so we don't change state
// CHECK-NEXT:    1100c:       11 f0 08 f8     bl      #69648
// CHECK-NEXT:    11010:       c0 f2 00 00     movt    r0, #0
// CHECK-NEXT:    11014:       40 f2 00 00     movw    r0, #0
