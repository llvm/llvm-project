// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -arm-add-build-attributes -triple=armv6m-none-eabi asm -o v6m.o
// RUN: ld.lld --script=lds v6m.o -o v6m
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -d v6m --triple=armv6m-none-eabi | FileCheck %s

// RUN: llvm-mc -filetype=obj -arm-add-build-attributes -triple=armv8m.base-none-eabi asm -o v8m.o
// RUN: ld.lld --script=lds v8m.o -o v8m
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -d v8m --triple=armv8m.base-none-eabi | FileCheck --check-prefix=CHECKV8BASE %s

/// Test that short thunks are not generated for v6-m as this architecture
/// does not have the B.w instruction.

//--- asm
 .syntax unified

 .section .text_low, "ax", %progbits
 .thumb
 .type _start, %function
 .balign 4
 .globl _start
_start:
 bl far
 .space 0x1000 - (. - _start)

/// Thunks will be inserted here. They are in short thunk range for a B.w
/// instruction. Expect v6-M to use a long thunk as v6-M does not have B.w.
/// Expect v8-m.base to use a short thunk as despite not having Thumb 2 it
/// does have B.w.

// CHECK-LABEL: <__Thumbv6MABSLongThunk_far>:
// CHECK-NEXT: 2000: push    {r0, r1}
// CHECK-NEXT:       ldr     r0, [pc, #4]
// CHECK-NEXT:       str     r0, [sp, #4]
// CHECK-NEXT:       pop     {r0, pc}
// CHECK-NEXT:   01 20 00 01   .word   0x01002001

// CHECKV8BASE-LABEL: <__Thumbv7ABSLongThunk_far>:
// CHECKV8BASE-NEXT: 2000: b.w     0x1002000 <far>

 .section .text_high, "ax", %progbits
 .globl far
 .type far, %function
 .balign 4
far:
 bx lr

//--- lds

PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .text_low  0x1000 : { *(.text_low) }
  .text_high 0x1002000  : { *(.text_high) }
}
