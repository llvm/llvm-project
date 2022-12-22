// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv4t-none-linux-gnueabi %t/a.s -o %t/a.o
// RUN: ld.lld %t/a.o --script %t/far.lds -o %t/a-far
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-far | FileCheck %s --check-prefixes=FAR
// RUN: ld.lld %t/a.o --script %t/near.lds -o %t/a-near
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv4t-none-linux-gnueabi %t/a-near | FileCheck %s --check-prefixes=NEAR

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
// FAR-EMPTY: 
// FAR-NEXT:  <$d>:
// FAR-NEXT:   1000014: 00 00 00 06  	.word	0x06000000
// FAR-EMPTY:
// FAR-NEXT:  <__Thumbv4ABSLongThunk_thumb_target>:
// FAR-NEXT:   1000018:      	bx	pc
// FAR-NEXT:                	b	0x1000018 <__Thumbv4ABSLongThunk_thumb_target> @ imm = #-6
// FAR-EMPTY:
// FAR-NEXT:  <$a>:
// FAR-NEXT:   100001c:      	ldr	r12, [pc]               @ 0x1000024 <__Thumbv4ABSLongThunk_thumb_target+0xc>
// FAR-NEXT:                	bx	r12
// FAR-EMPTY:
// FAR-NEXT:  <$d>:
// FAR-NEXT:   1000024: 05 00 00 06  	.word	0x06000005

// NEAR-LABEL: <_start>:
// NEAR-NEXT:  1000000:      	bl	0x100000c <thumb_start+0x4> @ imm = #4
// NEAR-NEXT:               	bx	lr
// NEAR-EMPTY:
// NEAR-LABEL: <thumb_start>:
// NEAR-NEXT:  1000008:      	bl	0x1000012 <thumb_target> @ imm = #6
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

// NEAR-LABEL: <target>:
// NEAR-LABEL:  100000e:      	bx	lr
// NEAR-EMPTY:
// NEAR-NEXT: <thumb_target>:
// NEAR-NEXT:  1000012:      	bx	lr

                                     
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
