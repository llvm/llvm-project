// REQUIRES: arm
// RUN: llvm-mc --arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump --triple=thumbv7a-none-linux-gnueabi -d %t2 | FileCheck %s

/// Check that the ARM ABI rules for undefined weak symbols are applied.
/// Branch instructions are resolved to the next instruction. Relative
/// relocations are resolved to the place.

 .syntax unified

 .weak target
 .type target, %function

 .text
 .global _start
_start:
/// R_ARM_THM_JUMP19
 beq.w target
/// R_ARM_THM_JUMP24
 b.w target
/// R_ARM_THM_CALL
 bl target
/// R_ARM_THM_CALL with exchange
 blx target
/// R_ARM_THM_MOVT_PREL
 movt r0, :upper16:target - .
/// R_ARM_THM_MOVW_PREL_NC
 movw r0, :lower16:target - .
/// R_ARM_THM_ALU_PREL_11_0
/// adr r0, target
 .inst.w 0xf2af0004
 .reloc 0x18, R_ARM_THM_ALU_PREL_11_0, target
/// R_ARM_THM_PC12
/// ldr r0, target
 .inst.w 0xf85f0004
 .reloc 0x1c, R_ARM_THM_PC12, target
// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK:         200b4: {{.*}} beq.w   #0 <_start+0x4>
// CHECK-NEXT:    200b8: {{.*}} b.w     #0 <_start+0x8>
// CHECK-NEXT:    200bc: {{.*}} bl      #0
/// blx is transformed into bl so we don't change state
// CHECK-NEXT:    200c0: {{.*}} bl      #0
// CHECK-NEXT:    200c4: {{.*}} movt    r0, #0
// CHECK-NEXT:    200c8: {{.*}} movw    r0, #0
// CHECK-NEXT:    200cc: {{.*}} adr.w   r0, #-4
// CHECK-NEXT:    200d0: {{.*}} ldr.w   r0, [pc, #-4]
