// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv6m-none-eabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       . = SIZEOF_HEADERS; \
// RUN:       .text_low : { *(.text_low) *(.text_low2) } \
// RUN:       .text_high 0x2000000 : { *(.text_high) *(.text_high2) } \
// RUN:       } " > %t.script
// RUN: not ld.lld --script %t.script %t -o %t2 2>&1 | FileCheck %s

// CHECK:  error: thunks not supported for architecture Armv6-m

// Range extension thunks are not currently supported on Armv6-m due to a
// combination of Armv6-m being aimed at low-end microcontrollers that typically
// have < 512 Kilobytes of memory, and the restrictions of the instruction set
// that make thunks inefficient. The main restriction is that the
// interprocedural scratch register r12 (ip) cannot be accessed from many
// instructions so we must use the stack to avoid corrupting the program.
//
// A v6-m Thunk would look like
//     push {r0, r1} ; Make 8-bytes of stack for restoring r0, and destination
//     ldr r0, [pc, #4] ; L1
//     str r0, [sp, #4] ; store destination address into sp + 4
//     pop {r0, pc} ; restore r0 and load pc with destination
// L1: .word destination

 .syntax unified
 .section .text_low, "ax", %progbits
 .thumb
 .type _start, %function
 .globl _start
_start:
 bl far

 .section .text_high, "ax", %progbits
 .globl far
 .type far, %function
far:
 bx lr

