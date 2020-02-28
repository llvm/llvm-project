// REQUIRES: arm
// RUN: llvm-mc --arm-add-build-attributes -filetype=obj -triple=thumbv6a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump --no-show-raw-insn -triple=thumbv6a-none-linux-gnueabi -d %t2

/// Check that the ARM ABI rules for undefined weak symbols are applied.
/// Relative relocations are resolved to the place. Although we can't encode
/// this for R_ARM_THM_PC8 as negative addends are not permitted. Use smallest
/// available value. These are corner cases.
 .syntax unified

 .weak target
 .type target, %function

 .text
 .global _start
_start:
 /// R_ARM_THM_PC8
 adr r0, target
 ldr r0, target

// CHECK: 000110b4 _start:
// CHECK-NEXT: 110b4: adr     r0, #0
// CHECK-NEXT:        ldr     r0, [pc, #0]
