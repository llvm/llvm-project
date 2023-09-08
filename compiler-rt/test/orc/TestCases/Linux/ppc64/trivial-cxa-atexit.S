// Test that the runtime correctly interposes ___cxa_atexit.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
	.text
	.abiversion 2
	.file	"cxa-atexit.c"
	.globl	on_exit_hook
	.p2align	4
	.type	on_exit_hook,@function
on_exit_hook:
.Lfunc_begin0:
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	.localentry	on_exit_hook, .Lfunc_lep0-.Lfunc_gep0
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	li 3, 0
	bl llvm_jitlink_setTestResultOverride
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	on_exit_hook, .Lfunc_end0-.Lfunc_begin0

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
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	addis 3, 2, on_exit_hook@toc@ha
	addi 3, 3, on_exit_hook@toc@l
	li 4, 0
	addis 5, 2, __dso_handle@toc@ha
	addi 5, 5, __dso_handle@toc@l
	bl __cxa_atexit
	nop
	li 3, 1
	bl llvm_jitlink_setTestResultOverride
	nop
	li 3, 0
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end1:
	.size	main, .Lfunc_end1-.Lfunc_begin1

  .hidden __dso_handle
