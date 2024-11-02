// Test that basic COFF static initializers work. The main function in this
// test returns the expression a + b + c + 1. a is initialzed as -1 and c is
// initialized as -2 by static initializers. b is initialized as 2 by constant
// value. The test will succeeds only if static initializers are ran successfully.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"static-initializer.cpp"
	.def	init;
	.scl	2;
	.type	32;
	.endef
	.globl	init                            # -- Begin function init
	.p2align	4, 0x90
init:                                   # @init
# %bb.0:
	movl	$4294967295, %eax               # imm = 0xFFFFFFFF
	retq
                                        # -- End function
	.def	init2;
	.scl	2;
	.type	32;
	.endef
	.globl	init2                           # -- Begin function init2
	.p2align	4, 0x90
init2:                                  # @init2
# %bb.0:
	movl	$4294967294, %eax               # imm = 0xFFFFFFFE
	retq
                                        # -- End function
	.def	"??__Ea@@YAXXZ";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90                         # -- Begin function ??__Ea@@YAXXZ
"??__Ea@@YAXXZ":                        # @"??__Ea@@YAXXZ"
.seh_proc "??__Ea@@YAXXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	callq	init
	movl	%eax, a(%rip)
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??__Ec@@YAXXZ";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90                         # -- Begin function ??__Ec@@YAXXZ
"??__Ec@@YAXXZ":                        # @"??__Ec@@YAXXZ"
.seh_proc "??__Ec@@YAXXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	callq	init2
	movl	%eax, c(%rip)
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.seh_proc main
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movl	$0, 52(%rsp)
	movq	%rdx, 40(%rsp)
	movl	%ecx, 36(%rsp)
	movl	a(%rip), %ecx
	addl	b(%rip), %ecx
	addl	c(%rip), %ecx
	addl	$1, %ecx
	callq	*__imp_llvm_jitlink_setTestResultOverride(%rip)
	xorl	%eax, %eax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	_GLOBAL__sub_I_static_initializer.cpp;
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_static_initializer.cpp
_GLOBAL__sub_I_static_initializer.cpp:  # @_GLOBAL__sub_I_static_initializer.cpp
.seh_proc _GLOBAL__sub_I_static_initializer.cpp
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	callq	"??__Ea@@YAXXZ"
	callq	"??__Ec@@YAXXZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.bss
	.globl	a                               # @a
	.p2align	2
a:
	.long	0                               # 0x0

	.data
	.globl	b                               # @b
	.p2align	2
b:
	.long	2                               # 0x2

	.bss
	.globl	c                               # @c
	.p2align	2
c:
	.long	0                               # 0x0

	.section	.CRT$XCU,"dr"
	.p2align	3
	.quad	_GLOBAL__sub_I_static_initializer.cpp
