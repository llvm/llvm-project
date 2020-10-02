// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d --start-address=0x81d1008 --stop-address=0x81d1014 --no-show-raw-insn %t | FileCheck %s
// RUN: rm %t.o %t
// Check that the range extension thunks are dumped close to the aarch64 branch
// range of 128 MiB
 .section .text.1, "ax", %progbits
 .balign 0x1000
 .globl _start
_start:
 bl high_target
 ret

 .section .text.2, "ax", %progbits
 .space 0x2000000

 .section .text.2, "ax", %progbits
 .space 0x2000000

 .section .text.3, "ax", %progbits
 .space 0x2000000

 .section .text.4, "ax", %progbits
 .space 0x2000000 - 0x40000

 .section .text.5, "ax", %progbits
 .space 0x40000

 .section .text.6, "ax", %progbits
 .balign 0x1000

 .globl high_target
 .type high_target, %function
high_target:
 ret

// CHECK: <__AArch64AbsLongThunk_high_target>:
// CHECK-NEXT:  81d1008:       ldr     x16, 0x81d1010
// CHECK-NEXT:  81d100c:       br      x16
// CHECK: <$d>:
// CHECK-NEXT:  81d1010:       00 20 21 08     .word   0x08212000
