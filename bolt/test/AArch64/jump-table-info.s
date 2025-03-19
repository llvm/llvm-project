## Check parsing of a .llvm_jump_table_info section
## The assembly is produced from the following code snippet:
## volatile int g;
## struct FooB {
##   int a;
##   float b;
## };
## static const struct FooB F = { 5, 42.42 };
## void switchy(int x) {
##   switch (x) {
##   case 0: g--; break;
##   case 1: g++; break;
##   case 2: g = 42; break;
##   case 3: g += 17; break;
##   case 4: g -= 66; break;
##   case 5: g++; g--; break;
##   case 6: g--; g++; break;
##   case 66: g-=3; g++; break;
##   case 8: g+=5; g--; break;
##   case 10: g+=5; g--; break;
##   case 12: g+=42; g--; break;
##   case 15: g+=99; g--; break;
##   case 20: switchy(g); break;
##   case 21: g -= 1234; break;
##   default:
##           g = 0;break;
##   }
## }
## const void *getf(void) {
##   return &F;
## }

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.null -print-jump-tables | FileCheck %s

# Confirm 67 entries are parsed:
# CHECK:      jump tables for function _Z7switchyi:
# CHECK-NEXT: Jump table {{.*}} for function _Z7switchyi
# CHECK:      0x0042 : .Ltmp16

	.text
	.globl	_Z7switchyi                     // -- Begin function _Z7switchyi
	.p2align	2
	.type	_Z7switchyi,@function
_Z7switchyi:                            // @_Z7switchyi
	.cfi_startproc
// %bb.0:                               // %entry
	adrp	x8, g
	cmp	w0, #20
	b.ne	.LBB0_2
.LBB0_1:                                // %sw.bb26
                                        // =>This Inner Loop Header: Depth=1
	ldr	w0, [x8, :lo12:g]
	cmp	w0, #20
	b.eq	.LBB0_1
.LBB0_2:                                // %tailrecurse
	cmp	w0, #66
	b.hi	.LBB0_18
// %bb.3:                               // %tailrecurse
	mov	w9, w0
	adrp	x10, .LJTI0_0
	add	x10, x10, :lo12:.LJTI0_0
	adr	x11, .LBB0_4
.Ltmp0:
	ldrb	w12, [x10, x9]
	add	x11, x11, x12, lsl #2
.Ltmp1:
	br	x11
.LBB0_4:                                // %sw.bb17
	ldr	w9, [x8, :lo12:g]
	add	w9, w9, #5
	b	.LBB0_13
.LBB0_5:                                // %sw.bb11
	ldr	w9, [x8, :lo12:g]
	sub	w9, w9, #3
	b	.LBB0_10
.LBB0_6:                                // %sw.bb5
	ldr	w9, [x8, :lo12:g]
	add	w9, w9, #1
	b	.LBB0_13
.LBB0_7:                                // %sw.bb3
	ldr	w9, [x8, :lo12:g]
	add	w9, w9, #17
	str	w9, [x8, :lo12:g]
	ret
.LBB0_8:                                // %sw.bb23
	ldr	w9, [x8, :lo12:g]
	add	w9, w9, #99
	b	.LBB0_13
.LBB0_9:                                // %sw.bb8
	ldr	w9, [x8, :lo12:g]
	sub	w9, w9, #1
.LBB0_10:                               // %sw.epilog
	str	w9, [x8, :lo12:g]
.LBB0_11:                               // %sw.bb1
	ldr	w9, [x8, :lo12:g]
	add	w9, w9, #1
	str	w9, [x8, :lo12:g]
	ret
.LBB0_12:                               // %sw.bb20
	ldr	w9, [x8, :lo12:g]
	add	w9, w9, #42
.LBB0_13:                               // %sw.epilog
	str	w9, [x8, :lo12:g]
.LBB0_14:                               // %sw.bb
	ldr	w9, [x8, :lo12:g]
	sub	w9, w9, #1
	str	w9, [x8, :lo12:g]
	ret
.LBB0_15:                               // %sw.epilog.loopexit
	mov	w9, #42                         // =0x2a
	str	w9, [x8, :lo12:g]
	ret
.LBB0_16:                               // %sw.bb27
	ldr	w9, [x8, :lo12:g]
	sub	w9, w9, #1234
	str	w9, [x8, :lo12:g]
	ret
.LBB0_17:                               // %sw.bb4
	ldr	w9, [x8, :lo12:g]
	sub	w9, w9, #66
	str	w9, [x8, :lo12:g]
	ret
.LBB0_18:                               // %sw.epilog.loopexit29
	str	wzr, [x8, :lo12:g]
	ret
.Lfunc_end0:
	.size	_Z7switchyi, .Lfunc_end0-_Z7switchyi
	.cfi_endproc
	.section	.rodata,"a",@progbits
.LJTI0_0:
	.byte	(.LBB0_14-.LBB0_4)>>2
	.byte	(.LBB0_11-.LBB0_4)>>2
	.byte	(.LBB0_15-.LBB0_4)>>2
	.byte	(.LBB0_7-.LBB0_4)>>2
	.byte	(.LBB0_17-.LBB0_4)>>2
	.byte	(.LBB0_6-.LBB0_4)>>2
	.byte	(.LBB0_9-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_4-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_4-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_12-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_8-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_16-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_18-.LBB0_4)>>2
	.byte	(.LBB0_5-.LBB0_4)>>2
	.section	.llvm_jump_table_info,"",@0x6fff4c0e
	.byte	2                               // format 2: 1b relative; shr 2
	.xword	.LJTI0_0
	.xword	.LBB0_4                         // Base
	.xword	.Ltmp0                          // Load Instruction
	.xword	.Ltmp1                          // Branch Instruction
	.byte	67                              // Number of Entries
                                        // -- End function
	.text
	.globl	_Z4getfv                        // -- Begin function _Z4getfv
	.p2align	2
	.type	_Z4getfv,@function
_Z4getfv:                               // @_Z4getfv
	.cfi_startproc
// %bb.0:                               // %entry
	adrp	x0, _ZL1F
	add	x0, x0, :lo12:_ZL1F
	ret
.Lfunc_end1:
	.size	_Z4getfv, .Lfunc_end1-_Z4getfv
	.cfi_endproc
                                        // -- End function
	.type	g,@object                       // @g
	.bss
	.globl	g
	.p2align	2, 0x0
g:
	.word	0                               // 0x0
	.size	g, 4
	.type	_ZL1F,@object                   // @_ZL1F
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
_ZL1F:
	.word	5                               // 0x5
	.word	0x4229ae14                      // float 42.4199982
	.size	_ZL1F, 8
	.section	".note.GNU-stack","",@progbits
