// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv6m-none-eabi %t/a.s -o %t/a.o
// RUN: ld.lld --no-rosegment --script %t/a.t %t/a.o -o %t/a
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -d %t/a --triple=armv6m-none-eabi | FileCheck %s
// RUN: not ld.lld --no-rosegment --script %t/a.t %t/a.o -o %t/a2 --pie 2>&1 | FileCheck --check-prefix=CHECK-PI %s
// RUN: rm -f %t/a %t/a2

// Range extension thunks for Arm Architecture v6m. Only Thumb instructions
// are permitted which limits the access to instructions that can access the
// high registers (r8 - r15), this means that the thunks have to spill
// low registers (r0 - r7) in order to perform the transfer of control.

//--- a.t
SECTIONS {
  .text_low  0x11345670 : { *(.text_low) }
  .text_high 0x12345678 : { *(.text_high) }
}

//--- a.s
// The 'y' on the .section directive  means that this section is eXecute Only code
 .syntax unified
 .section .text_low, "axy", %progbits
 .thumb
 .type _start, %function
 .balign 4
 .globl _start
_start:
 bl far

 .section .text_high, "ax", %progbits
 .globl far
 .type far, %function
far:
 bx lr

// CHECK: Disassembly of section .text_low:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT: 11345670:        bl      0x11345674 <__Thumbv6MABSXOLongThunk_far>
// CHECK: <__Thumbv6MABSXOLongThunk_far>:
// CHECK-NEXT:                  push    {r0, r1}
// CHECK-NEXT:                  movs    r0, #18
// CHECK-NEXT:                  lsls    r0, r0, #8
// CHECK-NEXT:                  adds    r0, #52
// CHECK-NEXT:                  lsls    r0, r0, #8
// CHECK-NEXT:                  adds    r0, #86
// CHECK-NEXT:                  lsls    r0, r0, #8
// CHECK-NEXT:                  adds    r0, #121
// CHECK-NEXT:                  str     r0, [sp, #4]
// CHECK-NEXT:                  pop     {r0, pc}
// CHECK: Disassembly of section .text_high:
// CHECK-EMPTY:
// CHECK-NEXT: <far>:
// CHECK-NEXT: 12345678:        bx      lr

// CHECK-PI:  error: relocation R_ARM_THM_CALL to far not supported for Armv6-M targets for position independent and execute only code
