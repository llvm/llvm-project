// REQUIRES: arm
// RUN: llvm-mc --triple=thumbv6m-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t --triple=thumbv6m-none-eabi | FileCheck %s

/// Test R_ARM_THM_PC8 as used in the ldr pseudo instruction. Only positive
/// 4-byte aligned offsets are permitted.
 .section .text.01, "ax", %progbits
 .balign 4
 .global _start
 .thumb_func
_start:
/// ldr r0, target1
 .inst.n 0x48ff
 .reloc 0, R_ARM_THM_PC8, target1
/// ldr r1, target2
 .inst.n 0x49ff
 .reloc 2, R_ARM_THM_PC8, target2
 .section .text.02, "ax", %progbits
 .balign 4
 .global target1
 .type target1, %function
target1:
 nop
 bx lr
 .section .text.03, "ax", %progbits
 .balign 4
 .space 1016
 .type target2, %function
target2:
 nop
 bx lr

// CHECK: 000200b4 <_start>:
// CHECK-NEXT: 200b4: ldr     r0, [pc, #0]
// CHECK-NEXT: 200b6: ldr     r1, [pc, #1020]

// CHECK: 000200b8 <target1>:
// CHECK-NEXT: 200b8: nop
// CHECK-NEXT: 200ba: bx      lr

// CHECK: 000204b4 <target2>:
// CHECK-NEXT: 204b4: nop
// CHECK-NEXT: 204b6: bx      lr
