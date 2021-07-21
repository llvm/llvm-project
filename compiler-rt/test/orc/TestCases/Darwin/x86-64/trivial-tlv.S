// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
//
// Test that basic MachO TLVs work by adding together TLVs with values
// 0, 1, and -1, and returning the result (0 for success). This setup
// tests both zero-initialized (__thread_bss) and non-zero-initialized
// (__thread_data) secitons.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0

        .globl  _main
        .p2align        4, 0x90
_main:
	pushq	%rax
	movq	_x@TLVP(%rip), %rdi
	callq	*(%rdi)
	movq	%rax, %rcx
	movq	_y@TLVP(%rip), %rdi
	callq	*(%rdi)
	movl	(%rax), %edx
	addl	(%rcx), %edx
	movq	_z@TLVP(%rip), %rdi
	callq	*(%rdi)
	addl	(%rax), %edx
	movl	%edx, %eax
	popq	%rcx
	retq

.tbss _x$tlv$init, 4, 2

	.section	__DATA,__thread_vars,thread_local_variables
	.globl	_x
_x:
	.quad	__tlv_bootstrap
	.quad	0
	.quad	_x$tlv$init

	.section	__DATA,__thread_data,thread_local_regular
	.p2align	2
_y$tlv$init:
	.long	4294967295

	.section	__DATA,__thread_vars,thread_local_variables
	.globl	_y
_y:
	.quad	__tlv_bootstrap
	.quad	0
	.quad	_y$tlv$init

	.section	__DATA,__thread_data,thread_local_regular
	.p2align	2
_z$tlv$init:
	.long	1

	.section	__DATA,__thread_vars,thread_local_variables
	.globl	_z
_z:
	.quad	__tlv_bootstrap
	.quad	0
	.quad	_z$tlv$init

.subsections_via_symbols
