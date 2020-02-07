// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -triple=thumbv7a-none-linux-gnueabi -d %t2 | FileCheck %s

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
// CHECK-EMPTY:
// CHECK:         110b4: {{.*}} beq.w   #0 <_start+0x4>
// CHECK-NEXT:    110b8: {{.*}} b.w     #0 <_start+0x8>
// CHECK-NEXT:    110bc: {{.*}} bl      #0
// blx is transformed into bl so we don't change state
// CHECK-NEXT:    110c0: {{.*}} bl      #0
// CHECK-NEXT:    110c4: {{.*}} movt    r0, #0
// CHECK-NEXT:    110c8: {{.*}} movw    r0, #0
