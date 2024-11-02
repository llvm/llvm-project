// Test that the runtime correctly interposes ___cxa_atexit.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

	.text
// Destructor resets the test result override to zero.
	.globl	on_exit
	.p2align	2
	.type	on_exit,@function
on_exit:

	mov	w0, wzr
	b	llvm_jitlink_setTestResultOverride
.Lfunc_end0:
	.size	on_exit, .Lfunc_end0-on_exit

// main registers the atexit and sets the test result to one.
	.globl	main
	.p2align	2
	.type	main,@function
main:

	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	adrp	x8, :got:__dso_handle
	adrp	x0, :got:on_exit
	mov	x1, xzr
	ldr	x8, [x8, :got_lo12:__dso_handle]
	ldr	x2, [x8]
	ldr	x0, [x0, :got_lo12:on_exit]
	bl	__cxa_atexit
	mov	w0, #1
	bl	llvm_jitlink_setTestResultOverride
	mov	w0, wzr
	ldp	x29, x30, [sp], #16
	ret
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
