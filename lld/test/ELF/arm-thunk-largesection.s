// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -d -triple=thumbv7a-none-linux-gnueabi --start-address=0x12000 --stop-address=0x12006 %t2 | FileCheck -check-prefix=CHECK1 %s
// RUN: llvm-objdump -d -triple=thumbv7a-none-linux-gnueabi --start-address=0x13004 --stop-address=0x13008 %t2 | FileCheck -check-prefix=CHECK2 %s
// RUN: llvm-objdump -d -triple=thumbv7a-none-linux-gnueabi --start-address=0x1012ff8 --stop-address=0x1012ffc %t2 | FileCheck -check-prefix=CHECK3 %s
// RUN: llvm-objdump -d -triple=thumbv7a-none-linux-gnueabi --start-address=0x2012ff8 --stop-address=0x2012ffc %t2 | FileCheck -check-prefix=CHECK4 %s
// RUN: llvm-objdump -d -triple=thumbv7a-none-linux-gnueabi --start-address=0x3012fec --stop-address=0x3012ff6 %t2 | FileCheck -check-prefix=CHECK5 %s
 .syntax unified
 .balign 0x1000
 .thumb
 .text
 .globl _start
 .type _start, %function
_start:
 bx lr
 .space 0x1000
// CHECK1: Disassembly of section .text:
// CHECK1-EMPTY:
// CHECK1-NEXT:_start:
// CHECK1-NEXT:   12000:       70 47   bx      lr
// CHECK1-EMPTY:
// CHECK1-NEXT:$d.1:
// CHECK1-NEXT:   12002:       00 00 00 00 .word 0x00000000


// CHECK2: __Thumbv7ABSLongThunk__start:
// CHECK2-NEXT:    13004:       fe f7 fc bf     b.w     #-4104 <_start>

// Gigantic section where we need a ThunkSection either side of it
 .section .text.large1, "ax", %progbits
 .balign 4
 .space (16 * 1024 * 1024) - 16
 bl _start
 .space (16 * 1024 * 1024) - 4
 bl _start
 .space (16 * 1024 * 1024) - 16
// CHECK3: 1012ff8:     00 f4 04 d0     bl      #-16777208
// CHECK4: 2012ff8:     ff f3 f8 d7     bl      #16777200

// CHECK5: __Thumbv7ABSLongThunk__start:
// CHECK5-NEXT:  3012fec:       42 f2 01 0c     movw    r12, #8193
// CHECK5-NEXT:  3012ff0:       c0 f2 01 0c     movt    r12, #1
// CHECK5-NEXT:  3012ff4:       60 47   bx      r12
