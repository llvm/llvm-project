// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/far-arm-abs.s -o %tfar
// RUN: ld.lld  %t %tfar -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv7a-none-linux-gnueabi %t2 | FileCheck --check-prefix=SHORT %s
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/far-long-arm-abs.s -o %tfarlong
// RUN: ld.lld  %t %tfarlong -o %t3
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv7a-none-linux-gnueabi %t3 | FileCheck --check-prefix=LONG %s
 .syntax unified
 .section .text, "ax",%progbits
 .globl _start
 .balign 0x10000
 .type _start,%function
_start:
 /// address of too_far symbols are just out of range of ARM branch with
 /// 26-bit immediate field and an addend of -8
 bl  too_far1
 b   too_far2
 beq too_far3

// SHORT: 00030000 <_start>:
// SHORT-NEXT:    30000: bl      0x3000c <__ARMv7ABSLongThunk_too_far1>
// SHORT-NEXT:    30004: b       0x30010 <__ARMv7ABSLongThunk_too_far2>
// SHORT-NEXT:    30008: beq     0x30014 <__ARMv7ABSLongThunk_too_far3>
// SHORT:      0003000c <__ARMv7ABSLongThunk_too_far1>:
/// 0x2030008 = too_far1
// SHORT-NEXT:    3000c: b       0x2030008
// SHORT:      00030010 <__ARMv7ABSLongThunk_too_far2>:
/// 0x203000c = too_far2
// SHORT-NEXT:    30010: b       0x203000c
// SHORT:      00030014 <__ARMv7ABSLongThunk_too_far3>:
/// 0x2030010 = too_far3
// SHORT-NEXT:    30014: b       0x2030010

// LONG:      00030000 <_start>:
// LONG-NEXT:    30000: bl      0x3000c <__ARMv7ABSLongThunk_too_far1>
// LONG-NEXT:    30004: b       0x30018 <__ARMv7ABSLongThunk_too_far2>
// LONG-NEXT:    30008: beq     0x30024 <__ARMv7ABSLongThunk_too_far3>
// LONG:      0003000c <__ARMv7ABSLongThunk_too_far1>:
// LONG-NEXT:    3000c: movw    r12, #20
// LONG-NEXT:    30010: movt    r12, #515
// LONG-NEXT:    30014: bx      r12
// LONG:      00030018 <__ARMv7ABSLongThunk_too_far2>:
// LONG-NEXT:    30018: movw    r12, #32
// LONG-NEXT:    3001c: movt    r12, #515
// LONG-NEXT:    30020: bx      r12
// LONG:      00030024 <__ARMv7ABSLongThunk_too_far3>:
// LONG-NEXT:    30024: movw    r12, #44
// LONG-NEXT:    30028: movt    r12, #515
// LONG-NEXT:    3002c: bx      r12
