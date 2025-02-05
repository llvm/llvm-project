# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t && cd %t

## Check for incompatible cases
# RUN: not ld.lld %t --irpgo-profile=/dev/null --bp-startup-sort=function --call-graph-ordering-file=/dev/null 2>&1 | FileCheck %s --check-prefix=BP-STARTUP-CALLGRAPH-ERR
# RUN: not ld.lld --bp-compression-sort=function --call-graph-ordering-file /dev/null 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-CALLGRAPH-ERR
# RUN: not ld.lld --bp-startup-sort=function 2>&1 | FileCheck %s --check-prefix=BP-STARTUP-ERR
# RUN: not ld.lld --bp-compression-sort-startup-functions 2>&1 | FileCheck %s --check-prefix=BP-STARTUP-COMPRESSION-ERR
# RUN: not ld.lld --bp-startup-sort=invalid --bp-compression-sort=invalid 2>&1 | FileCheck %s --check-prefix=BP-INVALID

# BP-STARTUP-CALLGRAPH-ERR: error: --bp-startup-sort=function is incompatible with --call-graph-ordering-file
# BP-COMPRESSION-CALLGRAPH-ERR: error: --bp-compression-sort is incompatible with --call-graph-ordering-file
# BP-STARTUP-ERR: error: --bp-startup-sort=function must be used with --irpgo-profile
# BP-STARTUP-COMPRESSION-ERR: error: --bp-compression-sort-startup-functions must be used with --irpgo-profile

# BP-INVALID: error: --bp-compression-sort=: expected [none|function|data|both]
# BP-INVALID: error: --bp-startup-sort=: expected [none|function]

# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: llvm-profdata merge a.proftext -o a.profdata
# RUN: ld.lld a.o --irpgo-profile=a.profdata --bp-startup-sort=function --verbose-bp-section-orderer --icf=all 2>&1 | FileCheck %s --check-prefix=STARTUP-FUNC-ORDER

# STARTUP-FUNC-ORDER: Ordered 3 sections using balanced partitioning
# STARTUP-FUNC-ORDER: Total area under the page fault curve: 3.

# RUN: ld.lld -o out.s a.o --irpgo-profile=a.profdata --bp-startup-sort=function
# RUN: llvm-nm -jn out.s | tr '\n' , | FileCheck %s --check-prefix=STARTUP
# STARTUP: s5,s4,s3,s2,s1,A,B,C,F,E,D,_start,d4,d3,d2,d1,{{$}}

# RUN: ld.lld -o out.os a.o --irpgo-profile=a.profdata --bp-startup-sort=function --symbol-ordering-file a.txt
# RUN: llvm-nm -jn out.os | tr '\n' , | FileCheck %s --check-prefix=ORDER-STARTUP
# ORDER-STARTUP: s2,s1,s5,s4,s3,A,F,E,D,B,C,_start,d3,d2,d4,d1,{{$}}

# RUN: ld.lld -o out.cf a.o --verbose-bp-section-orderer --bp-compression-sort=function 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-FUNC
# RUN: llvm-nm -jn out.cf | tr '\n' , | FileCheck %s --check-prefix=CFUNC
# CFUNC: s5,s4,s3,s2,s1,F,C,E,D,B,A,_start,d4,d3,d2,d1,{{$}}

# RUN: ld.lld -o out.cd a.o --verbose-bp-section-orderer --bp-compression-sort=data 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-DATA
# RUN: llvm-nm -jn out.cd | tr '\n' , | FileCheck %s --check-prefix=CDATA
# CDATA: s5,s3,s4,s2,s1,F,C,E,D,B,A,_start,d4,d1,d3,d2,{{$}}

# RUN: ld.lld -o out.cb a.o --verbose-bp-section-orderer --bp-compression-sort=both 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-BOTH
# RUN: llvm-nm -jn out.cb | tr '\n' , | FileCheck %s --check-prefix=CDATA

# RUN: ld.lld -o out.cbs a.o --verbose-bp-section-orderer --bp-compression-sort=both --irpgo-profile=a.profdata --bp-startup-sort=function 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-BOTH
# RUN: llvm-nm -jn out.cbs | tr '\n' , | FileCheck %s --check-prefix=CBOTH-STARTUP
# CBOTH-STARTUP: s5,s3,s4,s2,s1,A,B,C,F,E,D,_start,d4,d1,d3,d2,{{$}}

# BP-COMPRESSION-FUNC: Ordered 7 sections using balanced partitioning
# BP-COMPRESSION-DATA: Ordered 9 sections using balanced partitioning
# BP-COMPRESSION-BOTH: Ordered 16 sections using balanced partitioning

#--- a.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
A, B, C

A
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1

B
# Func Hash:
2222
# Num Counters:
1
# Counter Values:
1

C
# Func Hash:
3333
# Num Counters:
1
# Counter Values:
1

D
# Func Hash:
4444
# Num Counters:
1
# Counter Values:
1

#--- a.txt
A
F
E
D
s2
s1
d3
d2

#--- a.c
const char s5[] = "engineering";
const char s4[] = "computer program";
const char s3[] = "hardware engineer";
const char s2[] = "computer software";
const char s1[] = "hello world program";
int d4[] = {1,2,3,4,5,6};
int d3[] = {5,6,7,8};
int d2[] = {7,8,9,10};
int d1[] = {3,4,5,6};

int C(int a);
int B(int a);
void A();

int F(int a) { return C(a + 3); }
int E(int a) { return C(a + 2); }
int D(int a) { return B(a + 2); }
int C(int a) { A(); return a + 2; }
int B(int a) { A(); return a + 1; }
void A() {}

int _start() { return 0; }

#--- gen
clang --target=aarch64-linux-gnu -O0 -ffunction-sections -fdata-sections -fno-asynchronous-unwind-tables -S a.c -o -
;--- a.s
	.file	"a.c"
	.section	.text.F,"ax",@progbits
	.globl	F                               // -- Begin function F
	.p2align	2
	.type	F,@function
F:                                      // @F
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #3
	bl	C
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end0:
	.size	F, .Lfunc_end0-F
                                        // -- End function
	.section	.text.C,"ax",@progbits
	.globl	C                               // -- Begin function C
	.p2align	2
	.type	C,@function
C:                                      // @C
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	bl	A
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end1:
	.size	C, .Lfunc_end1-C
                                        // -- End function
	.section	.text.E,"ax",@progbits
	.globl	E                               // -- Begin function E
	.p2align	2
	.type	E,@function
E:                                      // @E
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	bl	C
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end2:
	.size	E, .Lfunc_end2-E
                                        // -- End function
	.section	.text.D,"ax",@progbits
	.globl	D                               // -- Begin function D
	.p2align	2
	.type	D,@function
D:                                      // @D
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	bl	B
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end3:
	.size	D, .Lfunc_end3-D
                                        // -- End function
	.section	.text.B,"ax",@progbits
	.globl	B                               // -- Begin function B
	.p2align	2
	.type	B,@function
B:                                      // @B
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	bl	A
	ldur	w8, [x29, #-4]
	add	w0, w8, #1
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end4:
	.size	B, .Lfunc_end4-B
                                        // -- End function
	.section	.text.A,"ax",@progbits
	.globl	A                               // -- Begin function A
	.p2align	2
	.type	A,@function
A:                                      // @A
// %bb.0:                               // %entry
	ret
.Lfunc_end5:
	.size	A, .Lfunc_end5-A
                                        // -- End function
	.section	.text._start,"ax",@progbits
	.globl	_start                          // -- Begin function _start
	.p2align	2
	.type	_start,@function
_start:                                 // @_start
// %bb.0:                               // %entry
	mov	w0, wzr
	ret
.Lfunc_end6:
	.size	_start, .Lfunc_end6-_start
                                        // -- End function
	.type	s5,@object                      // @s5
	.section	.rodata.s5,"a",@progbits
	.globl	s5
s5:
	.asciz	"engineering"
	.size	s5, 12

	.type	s4,@object                      // @s4
	.section	.rodata.s4,"a",@progbits
	.globl	s4
s4:
	.asciz	"computer program"
	.size	s4, 17

	.type	s3,@object                      // @s3
	.section	.rodata.s3,"a",@progbits
	.globl	s3
s3:
	.asciz	"hardware engineer"
	.size	s3, 18

	.type	s2,@object                      // @s2
	.section	.rodata.s2,"a",@progbits
	.globl	s2
s2:
	.asciz	"computer software"
	.size	s2, 18

	.type	s1,@object                      // @s1
	.section	.rodata.s1,"a",@progbits
	.globl	s1
s1:
	.asciz	"hello world program"
	.size	s1, 20

	.type	d4,@object                      // @d4
	.section	.data.d4,"aw",@progbits
	.globl	d4
	.p2align	2, 0x0
d4:
	.word	1                               // 0x1
	.word	2                               // 0x2
	.word	3                               // 0x3
	.word	4                               // 0x4
	.word	5                               // 0x5
	.word	6                               // 0x6
	.size	d4, 24

	.type	d3,@object                      // @d3
	.section	.data.d3,"aw",@progbits
	.globl	d3
	.p2align	2, 0x0
d3:
	.word	5                               // 0x5
	.word	6                               // 0x6
	.word	7                               // 0x7
	.word	8                               // 0x8
	.size	d3, 16

	.type	d2,@object                      // @d2
	.section	.data.d2,"aw",@progbits
	.globl	d2
	.p2align	2, 0x0
d2:
	.word	7                               // 0x7
	.word	8                               // 0x8
	.word	9                               // 0x9
	.word	10                              // 0xa
	.size	d2, 16

	.type	d1,@object                      // @d1
	.section	.data.d1,"aw",@progbits
	.globl	d1
	.p2align	2, 0x0
d1:
	.word	3                               // 0x3
	.word	4                               // 0x4
	.word	5                               // 0x5
	.word	6                               // 0x6
	.size	d1, 16

	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym C
	.addrsig_sym B
	.addrsig_sym A
