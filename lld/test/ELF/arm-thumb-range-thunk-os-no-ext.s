// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv4t-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t2
/// The output file is large, most of it zeroes. We dissassemble only the
/// parts we need to speed up the test and avoid a large output file.
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0x100000 --stop-address=0x10000c | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0x200000 --stop-address=0x200002 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0x300000 --stop-address=0x300002 | FileCheck --check-prefix=CHECK3 %s
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0x400000 --stop-address=0x400028 | FileCheck --check-prefix=CHECK4 %s
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0x700000 --stop-address=0x700014 | FileCheck --check-prefix=CHECK5 %s
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0x800000 --stop-address=0x800004 | FileCheck --check-prefix=CHECK6 %s
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0x900000 --stop-address=0x900004 | FileCheck --check-prefix=CHECK7 %s
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0xa00000 --stop-address=0xa00018 | FileCheck --check-prefix=CHECK8 %s
// RUN: llvm-objdump --no-show-raw-insn -d %t2 --start-address=0xb00000 --stop-address=0xb00004 | FileCheck --check-prefix=CHECK9 %s

/// Test the Range extension Thunks for Thumb when all the code is in a single
/// OutputSection. The Thumb BL instruction has a range of 4Mb. We create a
/// series of functions a megabyte apart. We expect range extension thunks to be
/// created when a branch is out of range. Thunks will be reused whenever they
/// are in range.
.syntax unified

/// Define a function aligned on a megabyte boundary.
.macro FUNCTION suff
  .section .text.\suff\(), "ax", %progbits
  .thumb
  .balign 0x100000
  .globl tfunc\suff\()
  .type  tfunc\suff\(), %function
  tfunc\suff\():
  bx lr
.endm

.section .text, "ax", %progbits
.thumb
.globl _start
_start:
  /// tfunc00 and tfunc03 are within 4MB, so no Range Thunks expected.
  bl tfunc00
  bl tfunc03
  /// tfunc04 is > 4MB away, expect a Range Thunk to be generated, to go into
  /// the first of the pre-created ThunkSections.
  bl tfunc04

// CHECK1-LABEL: <_start>:
// CHECK1-NEXT:   100000:       bl      0x200000 <tfunc00>      @ imm = #0xffffc
// CHECK1-NEXT:                 bl      0x500000 <tfunc03>      @ imm = #0x3ffff8
// CHECK1-NEXT:                 bl      0x400008 <__Thumbv4ABSLongThunk_tfunc04> @ imm = #0x2ffffc

FUNCTION 00
// CHECK2-LABEL: <tfunc00>:
// CHECK2-NEXT:   200000:       bx      lr

FUNCTION 01
// CHECK3-LABEL: <tfunc01>:
// CHECK3-NEXT:   300000:       bx      lr

FUNCTION 02
  bl tfunc07
  /// The thunks should not generate a v7-style short thunk.
// CHECK4-LABEL: <tfunc02>:
// CHECK4-NEXT:   400000:       bx      lr
// CHECK4-NEXT:                 bl      0x400018 <__Thumbv4ABSLongThunk_tfunc07> @ imm = #0x12
// CHECK4-NEXT:                 bmi     0x3fffb2 <tfunc01+0xfffb2> @ imm = #-0x58
// CHECK4-EMPTY:
// CHECK4-NEXT:  <__Thumbv4ABSLongThunk_tfunc04>:
// CHECK4-NEXT:   400008:      	bx      pc
// CHECK4-NEXT:                 b       0x400008 <__Thumbv4ABSLongThunk_tfunc04> @ imm = #-0x6
// CHECK4-NEXT:  40000c:        ldr	    r12, [pc]               @ 0x400014 <__Thumbv4ABSLongThunk_tfunc04+0xc>
// CHECK4-NEXT:                 bx	    r12
// CHECK4-NEXT:   400014: 01 00 60 00  	.word	0x00600001
// CHECK4-EMPTY:
// CHECK4-NEXT:  <__Thumbv4ABSLongThunk_tfunc07>:
// CHECK4-NEXT:   400018:      	bx	    pc
// CHECK4-NEXT:             	b	    0x400018 <__Thumbv4ABSLongThunk_tfunc07> @ imm = #-0x6
// CHECK4-NEXT:   40001c:      	ldr	r12, [pc]                   @ 0x400024 <__Thumbv4ABSLongThunk_tfunc07+0xc>
// CHECK4-NEXT:             	bx	r12
// CHECK4-NEXT:   400024: 01 00 90 00  	.word	0x00900001

FUNCTION 03
FUNCTION 04
FUNCTION 05
// CHECK5-LABEL: <tfunc05>:
// CHECK5-NEXT:   700000:      	bx	lr
// CHECK5-NEXT:             	bmi	0x6fffae <tfunc04+0xfffae> @ imm = #-0x58
// CHECK5-EMPTY:
// CHECK5-NEXT:  <__Thumbv4ABSLongThunk_tfunc03>:
// CHECK5-NEXT:   700004:      	bx	pc
// CHECK5-NEXT:             	b	0x700004 <__Thumbv4ABSLongThunk_tfunc03> @ imm = #-0x6
// CHECK5-NEXT:   700008:      	ldr	r12, [pc]               @ 0x700010 <__Thumbv4ABSLongThunk_tfunc03+0xc>
// CHECK5-NEXT:             	bx	r12
// CHECK5-NEXT:   700010: 01 00 50 00  	.word	0x00500001

FUNCTION 06
  /// The backwards branch is within range, so no range extension necessary.
  bl tfunc04
// CHECK6-LABEL: <tfunc06>:
// CHECK6-NEXT:   800000:      	bx	lr
// CHECK6-NEXT:  	            bl	0x600000 <tfunc04>      @ imm = #-0x200006

FUNCTION 07
  /// The backwards branch is out of range.
  bl tfunc03
// CHECK7-LABEL: <tfunc07>:
// CHECK7-NEXT:   900000:      	bx	lr
// CHECK7-NEXT:               	bl	0x700004 <__Thumbv4ABSLongThunk_tfunc03> @ imm = #-0x200002

FUNCTION 08
  /// 2nd backwards branch outside of range to same fn. Should share thunk with
  /// previous call.
  bl tfunc03
// CHECK8-LABEL: <tfunc08>:
// CHECK8-NEXT:   a00000:      	bx	lr
// CHECK8-NEXT:             	bl	0x700004 <__Thumbv4ABSLongThunk_tfunc03> @ imm = #-0x300002
// CHECK8-NEXT:             	bmi	0x9fffb2 <tfunc07+0xfffb2> @ imm = #-0x58
// CHECK8-EMPTY:
// CHECK8-NEXT:  <__Thumbv4ABSLongThunk_tfunc03>:
// CHECK8-NEXT:   a00008:      	bx	pc
// CHECK8-NEXT:   a0000a:      	b	0xa00008 <__Thumbv4ABSLongThunk_tfunc03> @ imm = #-0x6
// CHECK8-NEXT:   a0000c:      	ldr	r12, [pc]               @ 0xa00014 <__Thumbv4ABSLongThunk_tfunc03+0xc>
// CHECK8-NEXT:   a00010:      	bx	r12
// CHECK8-NEXT:   a00014: 01 00 50 00  	.word	0x00500001

FUNCTION 09
  /// This call is out of range of ThunkSection at 0700004.
  /// These 3 calls to tfunc03 could have used the same thunk (section), but
  /// we are not that sophisticated.
  bl tfunc03
// CHECK9-LABEL: <tfunc09>:
// CHECK9-NEXT:   b00000:      	bx	lr
// CHECK9-NEXT:      	        bl	0xa00008 <__Thumbv4ABSLongThunk_tfunc03> @ imm = #-0xffffe
FUNCTION 10
FUNCTION 11
