// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv4-none-linux-gnueabi %t/a.s -o %t/a.o
// RUN: ld.lld %t/a.o --script %t/far.lds -o %t/a-far
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4-none-linux-gnueabi %t/a-far | FileCheck %s --check-prefixes=FAR
// RUN: ld.lld %t/a.o --script %t/near.lds -o %t/a-near
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4-none-linux-gnueabi %t/a-near | FileCheck %s --check-prefixes=NEAR
// RUN: ld.lld %t/a.o -pie --script %t/far.lds -o %t/a-far-pie
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4-none-linux-gnueabi %t/a-far-pie | FileCheck %s --check-prefixes=FAR-PIE
// RUN: ld.lld %t/a.o -pie --script %t/near.lds -o %t/a-near-pie
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4-none-linux-gnueabi %t/a-near-pie | FileCheck %s --check-prefixes=NEAR

/// On Armv4 there is no blx instruction so long branch/exchange looks slightly
/// different.

//--- a.s
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
  mov pc, lr

// FAR-LABEL: <_start>:
// FAR-NEXT:   1000000:      	bl	0x1000008 <__ARMv5LongLdrPcThunk_target> @ imm = #0
// FAR-NEXT:                	mov pc, lr
// FAR-EMPTY:
// FAR-NEXT:  <__ARMv5LongLdrPcThunk_target>:
// FAR-NEXT:   1000008:      	ldr	pc, [pc, #-4]           @ 0x100000c <__ARMv5LongLdrPcThunk_target+0x4>
// FAR-EMPTY:
// FAR-NEXT:  <$d>:
// FAR-NEXT:   100000c: 00 00 00 06  	.word	0x06000000

// FAR-PIE-LABEL: <_start>:
// FAR-PIE-NEXT:   1000000:      	bl	0x1000008 <__ARMv4PILongThunk_target> @ imm = #0
// FAR-PIE-NEXT:                	mov pc, lr
// FAR-PIE-EMPTY:
// FAR-PIE-NEXT:  <__ARMv4PILongThunk_target>:
// FAR-PIE-NEXT:   1000008:      	ldr	r12, [pc]               @ 0x1000010 <__ARMv4PILongThunk_target+0x8>
// FAR-PIE-NEXT:                	add	pc, pc, r12
// FAR-PIE-EMPTY:
// FAR-PIE-NEXT:  <$d>:
// FAR-PIE-NEXT:   1000010: ec ff ff 04  	.word	0x04ffffec

// NEAR-LABEL: <_start>:
// NEAR-NEXT:  1000000:      	bl 0x1000008 <target> @ imm = #0
// NEAR-NEXT:               	mov pc, lr

.section .high, "ax", %progbits
 .arm
 .globl target
 .type target,%function
target:
  mov pc, lr

// FAR-LABEL: <target>:
// FAR-NEXT:   6000000:      	mov pc, lr

// FAR-PIE-LABEL: <target>:
// FAR-PIE-NEXT:   6000000:     mov pc, lr
                                         
// NEAR-LABEL: <target>:
// NEAR-LABEL:  1000008:      	mov pc, lr

//--- far.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .low 0x01000000 : { *(.low) }
  .high 0x06000000 : { *(.high) }
}

//--- near.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .all 0x01000000 : { *(.low) *(.high) }
}
