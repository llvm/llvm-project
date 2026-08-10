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
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-near-pie | FileCheck %s --check-prefixes=NEAR-PIE

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv4teb-none-linux-gnueabi %t/a.s -o %t/a.o
// RUN: ld.lld %t/a.o --script %t/far.lds -o %t/a-far
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4teb-none-linux-gnueabi %t/a-far | FileCheck %s --check-prefixes=FAR-EB
// RUN: ld.lld %t/a.o --script %t/near.lds -o %t/a-near
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4teb-none-linux-gnueabi %t/a-near | FileCheck %s --check-prefixes=NEAR-EB
// RUN: ld.lld %t/a.o -pie --script %t/far.lds -o %t/a-far-pie
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4teb-none-linux-gnueabi %t/a-far-pie | FileCheck %s --check-prefixes=FAR-EB-PIE
// RUN: ld.lld %t/a.o -pie --script %t/near.lds -o %t/a-near-pie
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4teb-none-linux-gnueabi %t/a-near-pie | FileCheck %s --check-prefixes=NEAR-EB-PIE

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
// FAR-NEXT:   1000010: 01 00 00 06   .word   0x06000001

// FAR-EB-LABEL: <_start>:
// FAR-EB-NEXT:   1000000:       bl      0x1000008 <__ARMv4ABSLongBXThunk_target> @ imm = #0
// FAR-EB-NEXT:                  bx      lr
// FAR-EB-EMPTY:
// FAR-EB-NEXT:  <__ARMv4ABSLongBXThunk_target>:
// FAR-EB-NEXT:   1000008:       ldr     r12, [pc]               @ 0x1000010 <__ARMv4ABSLongBXThunk_target+0x8>
// FAR-EB-NEXT:                  bx      r12
// FAR-EB-NEXT:   1000010: 06 00 00 01   .word   0x06000001

// NEAR-LABEL: <_start>:
// NEAR-NEXT:   1000000:       bl      0x1000008 <__ARMv4ABSLongBXThunk_target> @ imm = #0
// NEAR-NEXT:                  bx      lr
// NEAR-EMPTY:
// NEAR-NEXT:  <__ARMv4ABSLongBXThunk_target>:
// NEAR-NEXT:   1000008:       ldr     r12, [pc]               @ 0x1000010 <__ARMv4ABSLongBXThunk_target+0x8>
// NEAR-NEXT:                  bx      r12
// NEAR-NEXT:  1000010: 15 00 00 01   .word   0x01000015

// NEAR-EB-LABEL: <_start>:
// NEAR-EB-NEXT:   1000000:       bl      0x1000008 <__ARMv4ABSLongBXThunk_target> @ imm = #0
// NEAR-EB-NEXT:                  bx      lr
// NEAR-EB-EMPTY:
// NEAR-EB-NEXT:  <__ARMv4ABSLongBXThunk_target>:
// NEAR-EB-NEXT:   1000008:       ldr     r12, [pc]               @ 0x1000010 <__ARMv4ABSLongBXThunk_target+0x8>
// NEAR-EB-NEXT:                  bx      r12
// NEAR-EB-NEXT:  1000010: 01 00 00 15   .word   0x01000015

// FAR-PIE-LABEL: <_start>:
// FAR-PIE-NEXT:   1000000:    	bl	0x1000008 <__ARMv4PILongBXThunk_target> @ imm = #0
// FAR-PIE-NEXT:               	bx	lr
// FAR-PIE-EMPTY:
// FAR-PIE-NEXT:  <__ARMv4PILongBXThunk_target>:
// FAR-PIE-NEXT:   1000008:     ldr	r12, [pc, #4]           @ 0x1000014 <__ARMv4PILongBXThunk_target+0xc>
// FAR-PIE-NEXT:                add	r12, pc, r12
// FAR-PIE-NEXT:                bx	r12
// FAR-PIE-NEXT:   1000014: ed ff ff 04  	.word	0x04ffffed

// FAR-EB-PIE-LABEL: <_start>:
// FAR-EB-PIE-NEXT:   1000000:    	bl	0x1000008 <__ARMv4PILongBXThunk_target> @ imm = #0
// FAR-EB-PIE-NEXT:               	bx	lr
// FAR-EB-PIE-EMPTY:
// FAR-EB-PIE-NEXT:  <__ARMv4PILongBXThunk_target>:
// FAR-EB-PIE-NEXT:   1000008:     ldr	r12, [pc, #4]           @ 0x1000014 <__ARMv4PILongBXThunk_target+0xc>
// FAR-EB-PIE-NEXT:                add	r12, pc, r12
// FAR-EB-PIE-NEXT:                bx	r12
// FAR-EB-PIE-NEXT:   1000014: 04 ff ff ed  	.word	0x04ffffed

// NEAR-PIE-LABEL: <_start>:
// NEAR-PIE-NEXT:   1000000:    bl	0x1000008 <__ARMv4PILongBXThunk_target> @ imm = #0
// NEAR-PIE-NEXT:               bx	lr
// NEAR-PIE-EMPTY:
// NEAR-PIE-NEXT:  <__ARMv4PILongBXThunk_target>:
// NEAR-PIE-NEXT:   1000008:    ldr	r12, [pc, #4]           @ 0x1000014 <__ARMv4PILongBXThunk_target+0xc>
// NEAR-PIE-NEXT:               add	r12, pc, r12
// NEAR-PIE-NEXT:               bx	r12
// NEAR-PIE-NEXT:   1000014: 05 00 00 00  	.word	0x00000005

// NEAR-EB-PIE-LABEL: <_start>:
// NEAR-EB-PIE-NEXT:   1000000:    bl	0x1000008 <__ARMv4PILongBXThunk_target> @ imm = #0
// NEAR-EB-PIE-NEXT:               bx	lr
// NEAR-EB-PIE-EMPTY:
// NEAR-EB-PIE-NEXT:  <__ARMv4PILongBXThunk_target>:
// NEAR-EB-PIE-NEXT:   1000008:    ldr	r12, [pc, #4]           @ 0x1000014 <__ARMv4PILongBXThunk_target+0xc>
// NEAR-EB-PIE-NEXT:               add	r12, pc, r12
// NEAR-EB-PIE-NEXT:               bx	r12
// NEAR-EB-PIE-NEXT:   1000014: 00 00 00 05  	.word	0x00000005

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
// FAR-NEXT:   600000c:       ldr     pc, [pc, #-4]           @ 0x6000010 <__Thumbv4ABSLongBXThunk__start+0x8>
// FAR-NEXT:  6000010: 00 00 00 01   .word   0x01000000

// FAR-EB-LABEL: <target>:
// FAR-EB-NEXT:   6000000:       bl      0x6000008 <__Thumbv4ABSLongBXThunk__start> @ imm = #4
// FAR-EB-NEXT:                  bx      lr
// FAR-EB-NEXT:                  bmi     0x5ffffb2 <__ARMv4ABSLongBXThunk_target+0x4ffffaa> @ imm = #-88
// FAR-EB-EMPTY:
// FAR-EB-NEXT:  <__Thumbv4ABSLongBXThunk__start>:
// FAR-EB-NEXT:   6000008:       bx      pc
// FAR-EB-NEXT:                  b       0x6000008 <__Thumbv4ABSLongBXThunk__start> @ imm = #-6
// FAR-EB-NEXT:   600000c:       ldr     pc, [pc, #-4]           @ 0x6000010 <__Thumbv4ABSLongBXThunk__start+0x8>
// FAR-EB-NEXT:  6000010: 01 00 00 00   .word   0x01000000

// NEAR-LABEL: <target>:
// NEAR-NEXT:   1000014:       bl      0x100001c <__Thumbv4ABSLongBXThunk__start> @ imm = #4
// NEAR-NEXT:                  bx      lr
// NEAR-NEXT:                  bmi     0xffffc6                @ imm = #-88
// NEAR-EMPTY:
// NEAR-NEXT:  <__Thumbv4ABSLongBXThunk__start>:
// NEAR-NEXT:   100001c:       bx      pc
// NEAR-NEXT:                  b       0x100001c <__Thumbv4ABSLongBXThunk__start> @ imm = #-6
// NEAR-NEXT:   1000020:       ldr     pc, [pc, #-4]           @ 0x1000024 <__Thumbv4ABSLongBXThunk__start+0x8>
// NEAR-NEXT:   1000024: 00 00 00 01   .word   0x01000000

// NEAR-EB-LABEL: <target>:
// NEAR-EB-NEXT:   1000014:       bl      0x100001c <__Thumbv4ABSLongBXThunk__start> @ imm = #4
// NEAR-EB-NEXT:                  bx      lr
// NEAR-EB-NEXT:                  bmi     0xffffc6                @ imm = #-88
// NEAR-EB-EMPTY:
// NEAR-EB-NEXT:  <__Thumbv4ABSLongBXThunk__start>:
// NEAR-EB-NEXT:   100001c:       bx      pc
// NEAR-EB-NEXT:                  b       0x100001c <__Thumbv4ABSLongBXThunk__start> @ imm = #-6
// NEAR-EB-NEXT:   1000020:       ldr     pc, [pc, #-4]           @ 0x1000024 <__Thumbv4ABSLongBXThunk__start+0x8>
// NEAR-EB-NEXT:   1000024: 01 00 00 00   .word   0x01000000

// FAR-PIE-LABEL: <target>:
// FAR-PIE-NEXT:   6000000:       	bl	0x6000008 <__Thumbv4PILongBXThunk__start> @ imm = #4
// FAR-PIE-NEXT:                	bx  lr
// FAR-PIE-NEXT:                	bmi 0x5ffffb2 <__ARMv4PILongBXThunk_target+0x4ffffaa> @ imm = #-88
// FAR-PIE-EMPTY:
// FAR-PIE-NEXT:  <__Thumbv4PILongBXThunk__start>:
// FAR-PIE-NEXT:   6000008:      	bx	pc
// FAR-PIE-NEXT:                	b	0x6000008 <__Thumbv4PILongBXThunk__start> @ imm = #-6
// FAR-PIE-NEXT:   600000c:      	ldr	r12, [pc]               @ 0x6000014 <__Thumbv4PILongBXThunk__start+0xc>
// FAR-PIE-NEXT:                	add	pc, r12, pc
// FAR-PIE-NEXT:   6000014: e8 ff ff fa  	.word	0xfaffffe8

// FAR-EB-PIE-LABEL: <target>:
// FAR-EB-PIE-NEXT:   6000000:       	bl	0x6000008 <__Thumbv4PILongBXThunk__start> @ imm = #4
// FAR-EB-PIE-NEXT:                	bx  lr
// FAR-EB-PIE-NEXT:                	bmi 0x5ffffb2 <__ARMv4PILongBXThunk_target+0x4ffffaa> @ imm = #-88
// FAR-EB-PIE-EMPTY:
// FAR-EB-PIE-NEXT:  <__Thumbv4PILongBXThunk__start>:
// FAR-EB-PIE-NEXT:   6000008:      	bx	pc
// FAR-EB-PIE-NEXT:                	b	0x6000008 <__Thumbv4PILongBXThunk__start> @ imm = #-6
// FAR-EB-PIE-NEXT:   600000c:      	ldr	r12, [pc]               @ 0x6000014 <__Thumbv4PILongBXThunk__start+0xc>
// FAR-EB-PIE-NEXT:                	add	pc, r12, pc
// FAR-EB-PIE-NEXT:   6000014: fa ff ff e8  	.word	0xfaffffe8

// NEAR-PIE-LABEL: <target>:
// NEAR-PIE-NEXT:   1000018:      	bl	0x1000020 <__Thumbv4PILongBXThunk__start> @ imm = #4
// NEAR-PIE-NEXT:               	bx	lr
// NEAR-PIE-NEXT:               	bmi	0xffffca                @ imm = #-88
// NEAR-PIE-EMPTY:
// NEAR-PIE-NEXT:  <__Thumbv4PILongBXThunk__start>:
// NEAR-PIE-NEXT:   1000020:      	bx	pc
// NEAR-PIE-NEXT:               	b	0x1000020 <__Thumbv4PILongBXThunk__start> @ imm = #-6
// NEAR-PIE-NEXT:   1000024:      	ldr	r12, [pc]               @ 0x100002c <__Thumbv4PILongBXThunk__start+0xc>
// NEAR-PIE-NEXT:               	add	pc, r12, pc
// NEAR-PIE-NEXT:   100002c: d0 ff ff ff  	.word	0xffffffd0

// NEAR-EB-PIE-LABEL: <target>:
// NEAR-EB-PIE-NEXT:   1000018:      	bl	0x1000020 <__Thumbv4PILongBXThunk__start> @ imm = #4
// NEAR-EB-PIE-NEXT:               	bx	lr
// NEAR-EB-PIE-NEXT:               	bmi	0xffffca                @ imm = #-88
// NEAR-EB-PIE-EMPTY:
// NEAR-EB-PIE-NEXT:  <__Thumbv4PILongBXThunk__start>:
// NEAR-EB-PIE-NEXT:   1000020:      	bx	pc
// NEAR-EB-PIE-NEXT:               	b	0x1000020 <__Thumbv4PILongBXThunk__start> @ imm = #-6
// NEAR-EB-PIE-NEXT:   1000024:      	ldr	r12, [pc]               @ 0x100002c <__Thumbv4PILongBXThunk__start+0xc>
// NEAR-EB-PIE-NEXT:               	add	pc, r12, pc
// NEAR-EB-PIE-NEXT:   100002c: ff ff ff d0  	.word	0xffffffd0

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
