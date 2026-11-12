// Test that the runtime correctly interposes atexit.
//
// REQUIRES: disabled
//   This test is disabled until a proper atexit interpose can be implemented:
//   the current one assumes that atexit is defined in the dylib that calls it,
//   which is not true in general. See
//   https://github.com/llvm/llvm-project/issues/74641
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

	.text
// OnExit destructor resets the test result override to zero.
	.section	.text._ZN6OnExitD2Ev,"axG",@progbits,_ZN6OnExitD2Ev,comdat
	.p2align	4, 0x90
	.type	_ZN6OnExitD2Ev,@function
_ZN6OnExitD2Ev:                         # @_ZN6OnExitD2Ev
	.cfi_startproc
	xorl	%edi, %edi
	jmp	llvm_jitlink_setTestResultOverride@PLT # TAILCALL
	.cfi_endproc

// main registers the atexit and sets the test result to one.
	.globl main
	.p2align	4, 0x90                         # -- Begin function main
	.type	main,@function
main:              # @main
	.cfi_startproc
# %bb.0:
	movq	_ZN6OnExitD2Ev@GOTPCREL(%rip), %rdi
	callq	atexit@PLT
	movl	$1, %edi
	callq	llvm_jitlink_setTestResultOverride@PLT
	xorl	%eax, %eax
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.type	_ZL6onExit,@object              # @_ZL6onExit
	.local	_ZL6onExit
	.comm	_ZL6onExit,1,1
