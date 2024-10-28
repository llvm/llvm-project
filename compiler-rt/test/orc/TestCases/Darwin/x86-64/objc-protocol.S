// Test that we register protocol metadata. Without registration, messaging
// a protocol metatype object crashes.
//
// RUN: %clang -c -o %t.o %s
// RUN: %llvm_jitlink -preload libobjc.A.dylib %t.o
//
// REQUIRES: jit-compatible-osx-swift-runtime

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0	sdk_version 15, 0
	.globl	_main                           ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	$0, -4(%rbp)
	movq	__OBJC_PROTOCOL_REFERENCE_$_P(%rip), %rdi
	callq	*_objc_retain@GOTPCREL(%rip)
	xorl	%eax, %eax
	addq	$16, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__objc_classname,cstring_literals
L_OBJC_CLASS_NAME_:                     ## @OBJC_CLASS_NAME_
	.asciz	"P"

	.private_extern	__OBJC_PROTOCOL_$_P     ## @"_OBJC_PROTOCOL_$_P"
	.section	__DATA,__data
	.globl	__OBJC_PROTOCOL_$_P
	.weak_definition	__OBJC_PROTOCOL_$_P
	.p2align	3, 0x0
__OBJC_PROTOCOL_$_P:
	.quad	0
	.quad	L_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	96                              ## 0x60
	.long	0                               ## 0x0
	.quad	0
	.quad	0
	.quad	0

	.private_extern	__OBJC_LABEL_PROTOCOL_$_P ## @"_OBJC_LABEL_PROTOCOL_$_P"
	.section	__DATA,__objc_protolist,coalesced,no_dead_strip
	.globl	__OBJC_LABEL_PROTOCOL_$_P
	.weak_definition	__OBJC_LABEL_PROTOCOL_$_P
	.p2align	3, 0x0
__OBJC_LABEL_PROTOCOL_$_P:
	.quad	__OBJC_PROTOCOL_$_P

	.private_extern	__OBJC_PROTOCOL_REFERENCE_$_P ## @"_OBJC_PROTOCOL_REFERENCE_$_P"
	.section	__DATA,__objc_protorefs,coalesced,no_dead_strip
	.globl	__OBJC_PROTOCOL_REFERENCE_$_P
	.weak_definition	__OBJC_PROTOCOL_REFERENCE_$_P
	.p2align	3, 0x0
__OBJC_PROTOCOL_REFERENCE_$_P:
	.quad	__OBJC_PROTOCOL_$_P

	.no_dead_strip	__OBJC_PROTOCOL_$_P
	.no_dead_strip	__OBJC_LABEL_PROTOCOL_$_P
	.no_dead_strip	__OBJC_PROTOCOL_REFERENCE_$_P
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
