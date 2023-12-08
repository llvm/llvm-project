// Test that the runtime correctly interposes ___cxa_atexit.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 11, 3

// OnExit destructor resets the test result override to zero.
	.globl	__ZN6OnExitD1Ev
	.weak_def_can_be_hidden	__ZN6OnExitD1Ev
	.p2align	4, 0x90
__ZN6OnExitD1Ev:
	xorl	%edi, %edi
        jmp	_llvm_jitlink_setTestResultOverride

// main registers the atexit and sets the test result to one.
	.globl	_main
	.p2align	4, 0x90
_main:
	pushq	%rbp
	movq	%rsp, %rbp

	movq	__ZN6OnExitD1Ev@GOTPCREL(%rip), %rdi
	leaq	_onExit(%rip), %rsi
	leaq	___dso_handle(%rip), %rdx
	callq   ___cxa_atexit

	movl	$1, %edi
	callq	_llvm_jitlink_setTestResultOverride
	xorl	%eax, %eax
	popq	%rbp
	retq

	.globl	_onExit
.zerofill __DATA,__common,_onExit,1,0

.subsections_via_symbols
