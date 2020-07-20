// REQUIRES: arm
// RUN: llvm-mc --triple=armv7a-linux-gnueabihf -arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump --no-show-raw-insn -d %t | FileCheck %s

.syntax unified
.section .arm_target, "ax", %progbits
.balign 0x1000
.arm
arm_func_with_notype:
.type arm_func_with_explicit_notype, %notype
arm_func_with_explicit_notype:
 bx lr

.section .thumb_target, "ax", %progbits
.balign 4
.thumb
thumb_func_with_notype:
.type thumb_func_with_explicit_notype, %notype
thumb_func_with_explicit_notype:
 bx lr

/// All the symbols that are targets of the branch relocations do not have
/// type STT_FUNC. LLD should not insert interworking thunks as non STT_FUNC
/// symbols have no state information, the ABI assumes the user has manually
/// done the interworking.
.section .arm_caller, "ax", %progbits
.balign 4
.arm
.global _start
_start:
 b .arm_target
 b arm_func_with_notype
 b arm_func_with_explicit_notype
 b .thumb_target
 b thumb_func_with_notype
 b thumb_func_with_explicit_notype

 .section .thumb_caller, "ax", %progbits
 .thumb
 .balign 4
 b.w .arm_target
 b.w arm_func_with_notype
 b.w arm_func_with_explicit_notype
 b.w .thumb_target
 b.w thumb_func_with_notype
 b.w thumb_func_with_explicit_notype
 beq.w .arm_target
 beq.w arm_func_with_notype
 beq.w arm_func_with_explicit_notype
 beq.w .thumb_target
 beq.w thumb_func_with_notype
 beq.w thumb_func_with_explicit_notype

// CHECK: 00012008 _start:
// CHECK-NEXT: 12008: b       #-16
// CHECK-NEXT: 1200c: b       #-20
// CHECK-NEXT: 12010: b       #-24
// CHECK-NEXT: 12014: b       #-24
// CHECK-NEXT: 12018: b       #-28
// CHECK-NEXT: 1201c: b       #-32
// CHECK:      12020: b.w     #-36
// CHECK-NEXT: 12024: b.w     #-40
// CHECK-NEXT: 12028: b.w     #-44
// CHECK-NEXT: 1202c: b.w     #-44
// CHECK-NEXT: 12030: b.w     #-48
// CHECK-NEXT: 12034: b.w     #-52
// CHECK-NEXT: 12038: beq.w   #-60
// CHECK-NEXT: 1203c: beq.w   #-64
// CHECK-NEXT: 12040: beq.w   #-68
// CHECK-NEXT: 12044: beq.w   #-68
// CHECK-NEXT: 12048: beq.w   #-72
// CHECK-NEXT: 1204c: beq.w   #-76
