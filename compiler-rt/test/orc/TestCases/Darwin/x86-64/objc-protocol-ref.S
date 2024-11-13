// Test that we register protocol reference metadata. Without registration,
// protocol pointers will not be uniqued correctly and @protocol(NSObject) will
// not match the runtime's pointer.
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
	movq	__OBJC_PROTOCOL_REFERENCE_$_NSObject(%rip), %rax
	movq	%rax, -16(%rbp)                 ## 8-byte Spill
	leaq	L_.str(%rip), %rdi
	callq	_objc_getProtocol
	movq	-16(%rbp), %rdx                 ## 8-byte Reload
	movq	%rax, %rsi
	movl	$1, %eax
	xorl	%ecx, %ecx
	cmpq	%rsi, %rdx
	cmovel	%ecx, %eax
	addq	$16, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__objc_classname,cstring_literals
L_OBJC_CLASS_NAME_:                     ## @OBJC_CLASS_NAME_
	.asciz	"NSObject"

	.private_extern	__OBJC_PROTOCOL_$_NSObject ## @"_OBJC_PROTOCOL_$_NSObject"
	.section	__DATA,__data
	.globl	__OBJC_PROTOCOL_$_NSObject
	.weak_definition	__OBJC_PROTOCOL_$_NSObject
	.p2align	3, 0x0
__OBJC_PROTOCOL_$_NSObject:
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

	.private_extern	__OBJC_LABEL_PROTOCOL_$_NSObject ## @"_OBJC_LABEL_PROTOCOL_$_NSObject"
	.section	__DATA,__objc_protolist,coalesced,no_dead_strip
	.globl	__OBJC_LABEL_PROTOCOL_$_NSObject
	.weak_definition	__OBJC_LABEL_PROTOCOL_$_NSObject
	.p2align	3, 0x0
__OBJC_LABEL_PROTOCOL_$_NSObject:
	.quad	__OBJC_PROTOCOL_$_NSObject

	.private_extern	__OBJC_PROTOCOL_REFERENCE_$_NSObject ## @"_OBJC_PROTOCOL_REFERENCE_$_NSObject"
	.section	__DATA,__objc_protorefs,coalesced,no_dead_strip
	.globl	__OBJC_PROTOCOL_REFERENCE_$_NSObject
	.weak_definition	__OBJC_PROTOCOL_REFERENCE_$_NSObject
	.p2align	3, 0x0
__OBJC_PROTOCOL_REFERENCE_$_NSObject:
	.quad	__OBJC_PROTOCOL_$_NSObject

	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"NSObject"

	.no_dead_strip	__OBJC_PROTOCOL_$_NSObject
	.no_dead_strip	__OBJC_LABEL_PROTOCOL_$_NSObject
	.no_dead_strip	__OBJC_PROTOCOL_REFERENCE_$_NSObject
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
