// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: echo 'SECTIONS { \
// RUN:         .text.1 0x10000 : { *(.text.1) } \
// RUN:         .text.2 0x200000000 : AT(0x20000) { *(.text.2) } \
// RUN:       } ' > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s

// The word should be an offset to the range extension thunk.
// CHECK-LABEL: <_start>:
// CHECK-NEXT:    10000:       04 00 00 00     .word   0x00000004

// The thunk redirects to the address of callee.
// CHECK-LABEL: <__AArch64AbsLongThunk_callee>:
// CHECK-NEXT:    10004:       ldr     x16, 0x1000c <$d>
// CHECK-NEXT:    10008:       br      x16

// CHECK-LABEL: <$d>:
// CHECK-NEXT:    1000c:       00 00 00 00     .word   0x00000000
// CHECK-NEXT:    10010:       02 00 00 00     .word   0x00000002

// CHECK-LABEL: <callee>:
// CHECK-NEXT:    200000000:      ret

  .section .text.1, "ax", %progbits
  .global _start
  .type _start, %function
_start:
  .word callee@PLT - .

  .section .text.2, "ax", %progbits
  .global callee
  .type callee, %function
callee:
  ret
