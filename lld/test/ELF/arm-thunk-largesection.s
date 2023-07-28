// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x21000 --stop-address=0x21006 %t2 | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x22004 --stop-address=0x22008 %t2 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x1021ff8 --stop-address=0x1021ffc %t2 | FileCheck --check-prefix=CHECK3 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x2012ff8 --stop-address=0x2021ffc %t2 | FileCheck --check-prefix=CHECK4 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x3021fec --stop-address=0x3021ff6 %t2 | FileCheck --check-prefix=CHECK5 %s

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7aeb-none-linux-gnueabi -mcpu=cortex-a8 %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x21000 --stop-address=0x21006 %t2 | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x22004 --stop-address=0x22008 %t2 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x1021ff8 --stop-address=0x1021ffc %t2 | FileCheck --check-prefix=CHECK3 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x2012ff8 --stop-address=0x2021ffc %t2 | FileCheck --check-prefix=CHECK4 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x3021fec --stop-address=0x3021ff6 %t2 | FileCheck --check-prefix=CHECK5 %s

// RUN: ld.lld --be8 %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x21000 --stop-address=0x21006 %t2 | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x22004 --stop-address=0x22008 %t2 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x1021ff8 --stop-address=0x1021ffc %t2 | FileCheck --check-prefix=CHECK3 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x2012ff8 --stop-address=0x2021ffc %t2 | FileCheck --check-prefix=CHECK4 %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --start-address=0x3021fec --stop-address=0x3021ff6 %t2 | FileCheck --check-prefix=CHECK5 %s

 .syntax unified
 .balign 0x1000
 .thumb
 .text
 .globl _start
 .type _start, %function
_start:
 bx lr
 .space 0x1000
// CHECK1:      <_start>:
// CHECK1-NEXT:   21000: bx      lr
// CHECK1-NEXT:   21002:       00 00 00 00 .word 0x00000000


// CHECK2:      <__Thumbv7ABSLongThunk__start>:
// CHECK2-NEXT:    22004: b.w     0x21000 <_start>

/// Gigantic section where we need a ThunkSection either side of it
 .section .text.large1, "ax", %progbits
 .balign 4
 .space (16 * 1024 * 1024) - 16
 bl _start
 .space (16 * 1024 * 1024) - 4
 bl _start
 .space (16 * 1024 * 1024) - 16
// CHECK3: 1021ff8: bl      0x22004 <__Thumbv7ABSLongThunk__start>
// CHECK4: 2021ff8: bl      0x3021fec <__Thumbv7ABSLongThunk__start>

// CHECK5:      <__Thumbv7ABSLongThunk__start>:
// CHECK5-NEXT:  3021fec: movw    r12, #4097
// CHECK5-NEXT:  3021ff0: movt    r12, #2
// CHECK5-NEXT:  3021ff4: bx      r12
