// Test that basic ELF static initializers work. The main function in this
// test returns the value of 'x', which is initially 1 in the data section,
// and reset to 0 if the _static_init function is run. If the static initializer
// does not run then main will return 1, causing the test to be treated as a
// failure.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

	.text
	.globl	main
	.p2align	2
	.type	main,@function
main:

	adrp	x8, :got:x
	ldr	x8, [x8, :got_lo12:x]
	ldr	w0, [x8]
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

// static initializer sets the value of 'x' to zero.

	.section	.text.startup,"ax",@progbits
	.p2align	2
	.type	static_init,@function
static_init:

	adrp	x8, :got:x
	ldr	x8, [x8, :got_lo12:x]
	str	wzr, [x8]
	ret
.Lfunc_end1:
	.size	static_init, .Lfunc_end1-static_init

	.type	x,@object
	.data
	.globl	x
	.p2align	2
x:
	.word	1
	.size	x, 4

	.section	.init_array,"aw",@init_array
	.p2align	3
	.xword	static_init
