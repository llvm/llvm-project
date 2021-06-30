// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld --image-base=0x10000000 %t -o %t2
// RUN: llvm-objdump --triple=armv7a-none-linux-gnueabi --no-show-raw-insn -d %t2 | FileCheck %s

/// Check that the ARM ABI rules for undefined weak symbols are applied.
/// Branch instructions are resolved to the next instruction. Undefined
/// Symbols in relative are resolved to the place so S - P + A = A.
/// We place the image-base at 0x10000000 to test that a range extensions thunk
/// is not generated.

 .syntax unified

 .weak target
 .type target, %function

 .text
 .global _start
_start:
/// R_ARM_JUMP24
 b target
/// R_ARM_CALL
 bl target
/// R_ARM_CALL with exchange
 blx target
/// R_ARM_MOVT_PREL
 movt r0, :upper16:target - .
/// R_ARM_MOVW_PREL_NC
 movw r0, :lower16:target - .
/// R_ARM_REL32
 .word target - .

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: 100100b4 <_start>:
// CHECK-NEXT: 100100b4: b       {{.+}} @ imm = #-4
// CHECK-NEXT: 100100b8: bl      {{.+}} @ imm = #-4
// CHECK-NEXT: 100100bc: bl      {{.+}} @ imm = #-4
// CHECK-NEXT: 100100c0: movt    r0, #0
// CHECK-NEXT: 100100c4: movw    r0, #0
// CHECK:      100100c8: 00 00 00 00     .word   0x00000000
