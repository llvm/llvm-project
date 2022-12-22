// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv4t-none-linux-gnueabi %t/a.s -o %t/a.o
// RUN: ld.lld %t/a.o --script %t/far.lds -o %t/a-far
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-far | FileCheck %s --check-prefixes=FAR
// RUN: ld.lld %t/a.o --script %t/near.lds -o %t/a-near
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-near | FileCheck %s --check-prefixes=NEAR

/// On Arm v4t there is no blx instruction so all interworking must go via
/// a thunk.

#--- a.s
 .text
 .syntax unified
 .cpu    arm7tdmi

 .section .low, "ax", %progbits
 .arm
 .globl _start
 .type   _start,%function
 .p2align       2
_start:
  bl target
  bx lr

// FAR-LABEL: <_start>:
// FAR-NEXT:   1000000:       bl      0x1000008 <__ARMv4ABSLongBXThunk_target> @ imm = #0
// FAR-NEXT:                  bx      lr
// FAR-EMPTY:
// FAR-NEXT:  <__ARMv4ABSLongBXThunk_target>:
// FAR-NEXT:   1000008:       ldr     r12, [pc]               @ 0x1000010 <__ARMv4ABSLongBXThunk_target+0x8>
// FAR-NEXT:                  bx      r12
// FAR-EMPTY:
// FAR-NEXT:  <$d>:
// FAR-NEXT:   1000010: 01 00 00 06   .word   0x06000001

// NEAR-LABEL: <_start>:
// NEAR-NEXT:   1000000:       bl      0x1000008 <__ARMv4ABSLongBXThunk_target> @ imm = #0
// NEAR-NEXT:                  bx      lr
// NEAR-EMPTY:
// NEAR-NEXT:  <__ARMv4ABSLongBXThunk_target>:
// NEAR-NEXT:   1000008:       ldr     r12, [pc]               @ 0x1000010 <__ARMv4ABSLongBXThunk_target+0x8>
// NEAR-NEXT:                  bx      r12
// NEAR-EMPTY:
// NEAR-NEXT:  <$d>:
// NEAR-NEXT:  1000010: 15 00 00 01   .word   0x01000015


.section .high, "ax", %progbits
.thumb
 .globl target
 .type target,%function
target:
  bl _start
  bx lr

// FAR-LABEL: <target>:
// FAR-NEXT:   6000000:       bl      0x6000008 <__Thumbv4ABSLongBXThunk__start> @ imm = #4
// FAR-NEXT:                  bx      lr
// FAR-NEXT:                  bmi     0x5ffffb2 <__ARMv4ABSLongBXThunk_target+0x4ffffaa> @ imm = #-88
// FAR-EMPTY:
// FAR-NEXT:  <__Thumbv4ABSLongBXThunk__start>:
// FAR-NEXT:   6000008:       bx      pc
// FAR-NEXT:                  b       0x6000008 <__Thumbv4ABSLongBXThunk__start> @ imm = #-6
// FAR-EMPTY:
// FAR-NEXT:  <$a>:
// FAR-NEXT:   600000c:       ldr     pc, [pc, #-4]           @ 0x6000010 <__Thumbv4ABSLongBXThunk__start+0x8>
// FAR-EMPTY:
// FAR-NEXT: <$d>:
// FAR-NEXT:  6000010: 00 00 00 01   .word   0x01000000

// NEAR-LABEL: <target>:
// NEAR-NEXT:   1000014:       bl      0x100001c <__Thumbv4ABSLongBXThunk__start> @ imm = #4
// NEAR-NEXT:   1000018:       bx      lr
// NEAR-NEXT:   100001a:       bmi     0xffffc6                @ imm = #-88
// NEAR-EMPTY:
// NEAR-NEXT:  <__Thumbv4ABSLongBXThunk__start>:
// NEAR-NEXT:   100001c:       bx      pc
// NEAR-NEXT:   100001e:       b       0x100001c <__Thumbv4ABSLongBXThunk__start> @ imm = #-6
// NEAR-EMPTY:
// NEAR-NEXT:  <$a>:
// NEAR-NEXT:   1000020:       ldr     pc, [pc, #-4]           @ 0x1000024 <__Thumbv4ABSLongBXThunk__start+0x8>
// NEAR-EMPTY:
// NEAR-NEXT:  <$d>:
// NEAR-NEXT:   1000024: 00 00 00 01   .word   0x01000000

#--- far.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .low 0x01000000 : { *(.low) }
  .high 0x06000000 : { *(.high) }
}

#--- near.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .all 0x01000000 : { *(.low) *(.high) }
}
