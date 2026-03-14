// REQUIRES: arm
// RUN: llvm-mc %s --arm-add-build-attributes --triple=armv7a-linux-gnueabihf --filetype=obj -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d --start-address=0x2200b4 --stop-address=0x2200be %t | FileCheck %s

// RUN: llvm-mc %s --arm-add-build-attributes --triple=armv7aeb-linux-gnueabihf --filetype=obj -mcpu=cortex-a8 -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d --start-address=0x2200b4 --stop-address=0x2200be %t | FileCheck %s
// RUN: ld.lld --be8 %t.o -o %t
// RUN: llvm-objdump -d --start-address=0x2200b4 --stop-address=0x2200be %t | FileCheck %s

        /// Create a conditional branch too far away from a precreated thunk
        /// section. This will need a thunk section created within range.
        .syntax unified
        .thumb

        .section .text.0, "ax", %progbits
        .space 2 * 1024 * 1024
        .globl _start
        .type _start, %function
_start:
        /// Range of +/- 1 Megabyte, new ThunkSection will need creating after
        /// .text.1
        beq.w target
        .section .text.1, "ax", %progbits
        bx lr

// CHECK: <_start>:
// CHECK-NEXT:   2200b4:        f000 8000       beq.w   0x2200b8 <__Thumbv7ABSLongThunk_target>
// CHECK: <__Thumbv7ABSLongThunk_target>:
// CHECK-NEXT:   2200b8:        f000 9001       b.w     0xe200be <target>
// CHECK:        2200bc:        4770            bx      lr

        .section .text.2, "ax", %progbits
        .space 12 * 1024 * 1024
        .globl target
        .type target, %function
target: bx lr
