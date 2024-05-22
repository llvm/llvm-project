// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv4t-none-linux-gnueabi %t/a.s -o %t/a.o
// RUN: ld.lld %t/a.o --script %t/far.lds -o %t/a-far
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-far | FileCheck %s --check-prefixes=FAR
// RUN: ld.lld %t/a.o --script %t/near.lds -o %t/a-near
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-near | FileCheck %s --check-prefixes=NEAR
// RUN: ld.lld %t/a.o -pie --script %t/far.lds -o %t/a-far-pie
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-far-pie | FileCheck %s --check-prefixes=FAR-PIE
// RUN: ld.lld %t/a.o -pie --script %t/near.lds -o %t/a-near-pie
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-near-pie | FileCheck %s --check-prefixes=NEAR

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv4teb-none-linux-gnueabi %t/a.s -o %t/a.o
// RUN: ld.lld %t/a.o --script %t/far.lds -o %t/a-far
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4teb-none-linux-gnueabi %t/a-far | FileCheck %s --check-prefixes=FAR-EB
// RUN: ld.lld %t/a.o --script %t/near.lds -o %t/a-near
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4teb-none-linux-gnueabi %t/a-near | FileCheck %s --check-prefixes=NEAR
// RUN: ld.lld %t/a.o -pie --script %t/far.lds -o %t/a-far-pie
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4teb-none-linux-gnueabi %t/a-far-pie | FileCheck %s --check-prefixes=FAR-EB-PIE
// RUN: ld.lld %t/a.o -pie --script %t/near.lds -o %t/a-near-pie
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4teb-none-linux-gnueabi %t/a-near-pie | FileCheck %s --check-prefixes=NEAR

/// On Armv4T there is no blx instruction so long branch/exchange looks slightly
/// different.

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

 .thumb
 .globl thumb_start
 .type  thumb_start,%function
 .p2align       2
thumb_start:
  bl thumb_target
  bx lr

// FAR-LABEL: <_start>:
// FAR-NEXT:   1000000:      	bl	0x1000010 <__ARMv5LongLdrPcThunk_target> @ imm = #8
// FAR-NEXT:                	bx	lr
// FAR-EMPTY:
// FAR-LABEL: <thumb_start>:
// FAR-NEXT:   1000008:      	bl	0x1000018 <__Thumbv4ABSLongThunk_thumb_target> @ imm = #12
// FAR-NEXT:                	bx	lr
// FAR-NEXT:                	bmi	0xffffba                @ imm = #-88
// FAR-EMPTY:
// FAR-NEXT:  <__ARMv5LongLdrPcThunk_target>:
// FAR-NEXT:   1000010:      	ldr	pc, [pc, #-4]           @ 0x1000014 <__ARMv5LongLdrPcThunk_target+0x4>
// FAR-NEXT:   1000014: 00 00 00 06  	.word	0x06000000
// FAR-EMPTY:
// FAR-NEXT:  <__Thumbv4ABSLongThunk_thumb_target>:
// FAR-NEXT:   1000018:      	bx	pc
// FAR-NEXT:                	b	0x1000018 <__Thumbv4ABSLongThunk_thumb_target> @ imm = #-6
// FAR-NEXT:   100001c:      	ldr	r12, [pc]               @ 0x1000024 <__Thumbv4ABSLongThunk_thumb_target+0xc>
// FAR-NEXT:                	bx	r12
// FAR-NEXT:   1000024: 05 00 00 06  	.word	0x06000005

// FAR-EB-LABEL: <_start>:
// FAR-EB-NEXT:   1000000:      	bl	0x1000010 <__ARMv5LongLdrPcThunk_target> @ imm = #8
// FAR-EB-NEXT:                	bx	lr
// FAR-EB-EMPTY:
// FAR-EB-LABEL: <thumb_start>:
// FAR-EB-NEXT:   1000008:      	bl	0x1000018 <__Thumbv4ABSLongThunk_thumb_target> @ imm = #12
// FAR-EB-NEXT:                	bx	lr
// FAR-EB-NEXT:                	bmi	0xffffba                @ imm = #-88
// FAR-EB-EMPTY:
// FAR-EB-NEXT:  <__ARMv5LongLdrPcThunk_target>:
// FAR-EB-NEXT:   1000010:      	ldr	pc, [pc, #-4]           @ 0x1000014 <__ARMv5LongLdrPcThunk_target+0x4>
// FAR-EB-NEXT:   1000014: 06 00 00 00  	.word	0x06000000
// FAR-EB-EMPTY:
// FAR-EB-NEXT:  <__Thumbv4ABSLongThunk_thumb_target>:
// FAR-EB-NEXT:   1000018:      	bx	pc
// FAR-EB-NEXT:                	b	0x1000018 <__Thumbv4ABSLongThunk_thumb_target> @ imm = #-6
// FAR-EB-NEXT:   100001c:      	ldr	r12, [pc]               @ 0x1000024 <__Thumbv4ABSLongThunk_thumb_target+0xc>
// FAR-EB-NEXT:                	bx	r12
// FAR-EB-NEXT:   1000024: 06 00 00 05  	.word	0x06000005

// FAR-PIE-LABEL: <_start>:
// FAR-PIE-NEXT:   1000000:      	bl	0x1000010 <__ARMv4PILongThunk_target> @ imm = #8
// FAR-PIE-NEXT:                	bx	lr
// FAR-PIE-EMPTY:
// FAR-PIE-NEXT:  <thumb_start>:
// FAR-PIE-NEXT:   1000008:      	bl	0x100001c <__Thumbv4PILongThunk_thumb_target> @ imm = #16
// FAR-PIE-NEXT:                	bx	lr
// FAR-PIE-NEXT:                	bmi	0xffffba                @ imm = #-88
// FAR-PIE-EMPTY:
// FAR-PIE-NEXT:  <__ARMv4PILongThunk_target>:
// FAR-PIE-NEXT:   1000010:      	ldr	r12, [pc]               @ 0x1000018 <__ARMv4PILongThunk_target+0x8>
// FAR-PIE-NEXT:                	add	pc, pc, r12
// FAR-PIE-NEXT:   1000018: e4 ff ff 04  	.word	0x04ffffe4
// FAR-PIE-EMPTY:
// FAR-PIE-NEXT:  <__Thumbv4PILongThunk_thumb_target>:
// FAR-PIE-NEXT:   100001c:      	bx	pc
// FAR-PIE-NEXT:                	b	0x100001c <__Thumbv4PILongThunk_thumb_target> @ imm = #-6
// FAR-PIE-NEXT:   1000020:      	ldr	r12, [pc, #4]           @ 0x100002c <__Thumbv4PILongThunk_thumb_target+0x10>
// FAR-PIE-NEXT:                	add	r12, pc, r12
// FAR-PIE-NEXT:                	bx	r12
// FAR-PIE-NEXT:   100002c: d9 ff ff 04  	.word	0x04ffffd9

// FAR-EB-PIE-LABEL: <_start>:
// FAR-EB-PIE-NEXT:   1000000:      	bl	0x1000010 <__ARMv4PILongThunk_target> @ imm = #8
// FAR-EB-PIE-NEXT:                	bx	lr
// FAR-EB-PIE-EMPTY:
// FAR-EB-PIE-NEXT:  <thumb_start>:
// FAR-EB-PIE-NEXT:   1000008:      	bl	0x100001c <__Thumbv4PILongThunk_thumb_target> @ imm = #16
// FAR-EB-PIE-NEXT:                	bx	lr
// FAR-EB-PIE-NEXT:                	bmi	0xffffba                @ imm = #-88
// FAR-EB-PIE-EMPTY:
// FAR-EB-PIE-NEXT:  <__ARMv4PILongThunk_target>:
// FAR-EB-PIE-NEXT:   1000010:      	ldr	r12, [pc]               @ 0x1000018 <__ARMv4PILongThunk_target+0x8>
// FAR-EB-PIE-NEXT:                	add	pc, pc, r12
// FAR-EB-PIE-NEXT:   1000018: 04 ff ff e4  	.word	0x04ffffe4
// FAR-EB-PIE-EMPTY:
// FAR-EB-PIE-NEXT:  <__Thumbv4PILongThunk_thumb_target>:
// FAR-EB-PIE-NEXT:   100001c:      	bx	pc
// FAR-EB-PIE-NEXT:                	b	0x100001c <__Thumbv4PILongThunk_thumb_target> @ imm = #-6
// FAR-EB-PIE-NEXT:   1000020:      	ldr	r12, [pc, #4]           @ 0x100002c <__Thumbv4PILongThunk_thumb_target+0x10>
// FAR-EB-PIE-NEXT:                	add	r12, pc, r12
// FAR-EB-PIE-NEXT:                	bx	r12
// FAR-EB-PIE-NEXT:   100002c: 04 ff ff d9  	.word	0x04ffffd9

// NEAR-LABEL: <_start>:
// NEAR-NEXT:  1000000:      	bl	0x1000010 <target> @ imm = #8
// NEAR-NEXT:               	bx	lr
// NEAR-EMPTY:
// NEAR-LABEL: <thumb_start>:
// NEAR-NEXT:  1000008:      	bl	0x1000014 <thumb_target> @ imm = #8
// NEAR-NEXT:               	bx	lr

.section .high, "ax", %progbits
 .arm
 .globl target
 .type target,%function
target:
  bx lr

.thumb
 .globl thumb_target
 .type thumb_target,%function
thumb_target:
  bx lr

// FAR-LABEL: <target>:
// FAR-NEXT:   6000000:      	bx	lr
// FAR-EMPTY:
// FAR-LABEL: <thumb_target>:
// FAR-NEXT:   6000004:      	bx	lr

// FAR-PIE-LABEL: <target>:
// FAR-PIE-NEXT:   6000000:     bx	lr
// FAR-PIE-EMPTY:
// FAR-PIE-LABEL: <thumb_target>:
// FAR-PIE-NEXT:   6000004:     bx	lr

// NEAR-LABEL: <target>:
// NEAR-LABEL:  1000010:      	bx	lr
// NEAR-EMPTY:
// NEAR-NEXT: <thumb_target>:
// NEAR-NEXT:  1000014:      	bx	lr

                                     
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
