// Test that basic ELF static initializers work. The main function in this
// test returns the value of 'x', which is initially 1 in the data section,
// and reset to 0 if the _static_init function is run. If the static initializer
// does not run then main will return 1, causing the test to be treated as a
// failure.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
	.text
	.abiversion 2
	.file	"init.c"
	.globl	static_init
	.p2align	4
	.type	static_init,@function
static_init:
.Lfunc_begin0:
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	.localentry	static_init, .Lfunc_lep0-.Lfunc_gep0
	addis 3, 2, x@toc@ha
	li 4, 0
	stw 4, x@toc@l(3)
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	static_init, .Lfunc_end0-.Lfunc_begin0

	.globl	main
	.p2align	4
	.type	main,@function
main:
.Lfunc_begin1:
.Lfunc_gep1:
	addis 2, 12, .TOC.-.Lfunc_gep1@ha
	addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
	.localentry	main, .Lfunc_lep1-.Lfunc_gep1
	addis 3, 2, x@toc@ha
	lwa 3, x@toc@l(3)
	blr
	.long	0
	.quad	0
.Lfunc_end1:
	.size	main, .Lfunc_end1-.Lfunc_begin1

	.type	x,@object
	.data
	.globl	x
	.p2align	2, 0x0
x:
	.long	1
	.size	x, 4

	.section	.init_array.0,"aw",@init_array
	.p2align	3
	.quad	static_init
