// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 11, 3

	.globl	_main
	.p2align	4, 0x90
_main:
	xorl	%eax, %eax
	retq

.subsections_via_symbols
