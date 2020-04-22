// REQUIRES: arm
// RUN: llvm-mc -g --triple=armv7a-linux-gnueabihf -arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: ld.lld %t.o -o %t 2>&1 | FileCheck %s --check-prefix=WARN
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
/// done the interworking. For the BL and BLX instructions LLD should
/// preserve the original instruction instead of writing out the correct one
/// for the assumed state at the target.
/// LLD will warn for the BL and BLX cases where the behavior has changed
/// from LLD 10.0
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
 bl .arm_target
 bl arm_func_with_notype
 bl arm_func_with_explicit_notype
 bl .thumb_target
 bl thumb_func_with_notype
 bl thumb_func_with_explicit_notype
// WARN: {{.*}}.s:[[# @LINE+1]]:(.arm_caller+0x30): branch and link relocation: R_ARM_CALL to STT_SECTION symbol .arm_target ; interworking not performed
 blx .arm_target
// WARN: {{.*}}.s:[[# @LINE+1]]:(.arm_caller+0x34): branch and link relocation: R_ARM_CALL to non STT_FUNC symbol: arm_func_with_notype interworking not performed; consider using directive '.type arm_func_with_notype, %function' to give symbol type STT_FUNC if interworking between ARM and Thumb is required
 blx arm_func_with_notype
// WARN: {{.*}}.s:[[# @LINE+1]]:(.arm_caller+0x38): branch and link relocation: R_ARM_CALL to non STT_FUNC symbol: arm_func_with_explicit_notype interworking not performed; consider using directive '.type arm_func_with_explicit_notype, %function' to give symbol type STT_FUNC if interworking between ARM and Thumb is required
 blx arm_func_with_explicit_notype
// WARN: {{.*}}.s:[[# @LINE+1]]:(.arm_caller+0x3C): branch and link relocation: R_ARM_CALL to STT_SECTION symbol .thumb_target ; interworking not performed
 blx .thumb_target
// WARN: {{.*}}.s:[[# @LINE+1]]:(.arm_caller+0x40): branch and link relocation: R_ARM_CALL to non STT_FUNC symbol: thumb_func_with_notype interworking not performed; consider using directive '.type thumb_func_with_notype, %function' to give symbol type STT_FUNC if interworking between ARM and Thumb is required
 blx thumb_func_with_notype
// WARN: {{.*}}.s:[[# @LINE+1]]:(.arm_caller+0x44): branch and link relocation: R_ARM_CALL to non STT_FUNC symbol: thumb_func_with_explicit_notype interworking not performed; consider using directive '.type thumb_func_with_explicit_notype, %function' to give symbol type STT_FUNC if interworking between ARM and Thumb is required
 blx thumb_func_with_explicit_notype

 .section .thumb_caller, "ax", %progbits
 .thumb
 .balign 4
 .global thumb_caller
thumb_caller:
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
// WARN: {{.*}}.s:[[# @LINE+1]]:(.thumb_caller+0x30): branch and link relocation: R_ARM_THM_CALL to STT_SECTION symbol .arm_target ; interworking not performed
 bl .arm_target
// WARN: {{.*}}.s:[[# @LINE+1]]:(.thumb_caller+0x34): branch and link relocation: R_ARM_THM_CALL to non STT_FUNC symbol: arm_func_with_notype interworking not performed; consider using directive '.type arm_func_with_notype, %function' to give symbol type STT_FUNC if interworking between ARM and Thumb is required
 bl arm_func_with_notype
 // WARN: {{.*}}.s:[[# @LINE+1]]:(.thumb_caller+0x38): branch and link relocation: R_ARM_THM_CALL to non STT_FUNC symbol: arm_func_with_explicit_notype interworking not performed; consider using directive '.type arm_func_with_explicit_notype, %function' to give symbol type STT_FUNC if interworking between ARM and Thumb is required
 bl arm_func_with_explicit_notype
// WARN: {{.*}}.s:[[# @LINE+1]]:(.thumb_caller+0x3C): branch and link relocation: R_ARM_THM_CALL to STT_SECTION symbol .thumb_target ; interworking not performed
 bl .thumb_target
// WARN: {{.*}}.s:[[# @LINE+1]]:(.thumb_caller+0x40): branch and link relocation: R_ARM_THM_CALL to non STT_FUNC symbol: thumb_func_with_notype interworking not performed; consider using directive '.type thumb_func_with_notype, %function' to give symbol type STT_FUNC if interworking between ARM and Thumb is required
 bl thumb_func_with_notype
// {{.*}}.s:[[# @LINE+1]]:(.thumb_caller+0x44): branch and link relocation: R_ARM_THM_CALL to non STT_FUNC symbol: thumb_func_with_explicit_notype interworking not performed; consider using directive '.type thumb_func_with_explicit_notype, %function' to give symbol type STT_FUNC if interworking between ARM and Thumb is required
 bl thumb_func_with_explicit_notype
 blx .arm_target
 blx arm_func_with_notype
 blx arm_func_with_explicit_notype
 blx .thumb_target
 blx thumb_func_with_notype
 blx thumb_func_with_explicit_notype

// CHECK: 00021008 <_start>:
// CHECK-NEXT: 21008: b       #-16
// CHECK-NEXT: 2100c: b       #-20
// CHECK-NEXT: 21010: b       #-24
// CHECK-NEXT: 21014: b       #-24
// CHECK-NEXT: 21018: b       #-28
// CHECK-NEXT: 2101c: b       #-32
// CHECK-NEXT: 21020: bl      #-40
// CHECK-NEXT: 21024: bl      #-44
// CHECK-NEXT: 21028: bl      #-48
// CHECK-NEXT: 2102c: bl      #-48
// CHECK-NEXT: 21030: bl      #-52
// CHECK-NEXT: 21034: bl      #-56
// CHECK-NEXT: 21038: blx     #-64
// CHECK-NEXT: 2103c: blx     #-68
// CHECK-NEXT: 21040: blx     #-72
// CHECK-NEXT: 21044: blx     #-72
// CHECK-NEXT: 21048: blx     #-76
// CHECK-NEXT: 2104c: blx     #-80

// CHECK: 00021050 <thumb_caller>:
// CHECK-NEXT: 21050: b.w     #-84
// CHECK-NEXT: 21054: b.w     #-88
// CHECK-NEXT: 21058: b.w     #-92
// CHECK-NEXT: 2105c: b.w     #-92
// CHECK-NEXT: 21060: b.w     #-96
// CHECK-NEXT: 21064: b.w     #-100
// CHECK-NEXT: 21068: beq.w   #-108
// CHECK-NEXT: 2106c: beq.w   #-112
// CHECK-NEXT: 21070: beq.w   #-116
// CHECK-NEXT: 21074: beq.w   #-116
// CHECK-NEXT: 21078: beq.w   #-120
// CHECK-NEXT: 2107c: beq.w   #-124
// CHECK-NEXT: 21080: bl      #-132
// CHECK-NEXT: 21084: bl      #-136
// CHECK-NEXT: 21088: bl      #-140
// CHECK-NEXT: 2108c: bl      #-140
// CHECK-NEXT: 21090: bl      #-144
// CHECK-NEXT: 21094: bl      #-148
// CHECK-NEXT: 21098: blx     #-156
// CHECK-NEXT: 2109c: blx     #-160
// CHECK-NEXT: 210a0: blx     #-164
// CHECK-NEXT: 210a4: blx     #-164
// CHECK-NEXT: 210a8: blx     #-168
// CHECK-NEXT: 210ac: blx     #-172
