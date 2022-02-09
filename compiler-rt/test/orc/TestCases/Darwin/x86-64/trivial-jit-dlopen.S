// Test that __orc_rt_macho_jit_dlopen and __orc_rt_macho_jit_dlclose run
// constructors and destructors as expected.
//
// This test calls dlopen and dlclose twice. We expect the inner calls to be
// no-ops.
//
// RUN: %clang -c -o %t.inits.o %p/Inputs/standalone-ctor-and-cxa-atexit-dtor.S
// RUN: %clang -c -o %t.test.o %s
// RUN: %llvm_jitlink \
// RUN:   -alias _dlopen=___orc_rt_macho_jit_dlopen \
// RUN:   -alias _dlclose=___orc_rt_macho_jit_dlclose \
// RUN:   %t.test.o -jd inits %t.inits.o -lmain | FileCheck %s
//
// CHECK: entering main
// CHECK-NEXT: first dlopen
// CHECK-NEXT: constructor
// CHECK-NEXT: second dlopen
// CHECK-NEXT: first dlclose
// CHECK-NEXT: second dlclose
// CHECK-NEXT: destructor
// CHECK-NEXT: leaving main

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0	sdk_version 13, 0
	.globl	_main
	.p2align	4, 0x90
_main:

	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	pushq	%rax
	leaq	L_str(%rip), %rdi
	callq	_puts
	leaq	L_str.9(%rip), %rdi
	callq	_puts
	leaq	L_.str.2(%rip), %rdi
	movl	$1, %esi
	callq	_dlopen
	movl	$-1, %r14d
	leaq	L_str.17(%rip), %r15
	testq	%rax, %rax
	je	LBB0_6

	movq	%rax, %rbx
	leaq	L_str.11(%rip), %rdi
	callq	_puts
	leaq	L_.str.2(%rip), %rdi
	movl	$1, %esi
	callq	_dlopen
	testq	%rax, %rax
	je	LBB0_6

	cmpq	%rbx, %rax
	je	LBB0_4

	leaq	L_str.18(%rip), %r15
	jmp	LBB0_6
LBB0_4:
	leaq	L_str.13(%rip), %rdi
	callq	_puts
	movq	%rbx, %rdi
	callq	_dlclose
	cmpl	$-1, %eax
	je	LBB0_6

	leaq	L_str.14(%rip), %rdi
	callq	_puts
	movq	%rbx, %rdi
	callq	_dlclose
	xorl	%r14d, %r14d
	cmpl	$-1, %eax
	sete	%r14b
	leaq	L_str.17(%rip), %rax
	leaq	L_str.15(%rip), %r15
	cmoveq	%rax, %r15
	negl	%r14d
LBB0_6:
	movq	%r15, %rdi
	callq	_puts
	movl	%r14d, %eax
	addq	$8, %rsp
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
	retq

	.section	__TEXT,__cstring,cstring_literals
L_.str.2:
	.asciz	"inits"

L_str:
	.asciz	"entering main"

L_str.9:
	.asciz	"first dlopen"

L_str.11:
	.asciz	"second dlopen"

L_str.13:
	.asciz	"first dlclose"

L_str.14:
	.asciz	"second dlclose"

L_str.15:
	.asciz	"leaving main"

L_str.17:
	.asciz	"failed"

L_str.18:
	.asciz	"handles do not match"

.subsections_via_symbols
