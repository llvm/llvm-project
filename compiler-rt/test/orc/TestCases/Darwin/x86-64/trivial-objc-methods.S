// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink -preload libobjc.A.dylib %t
//
// Test that Objective-C class and instance methods work.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 11, 5
	.p2align	4, 0x90
"-[ZeroGen objZero]":
	pushq	%rbp
	movq	%rsp, %rbp
	xorl	%eax, %eax
	popq	%rbp
	retq

	.p2align	4, 0x90
"+[ZeroGen classZero]":
	pushq	%rbp
	movq	%rsp, %rbp
	xorl	%eax, %eax
	popq	%rbp
	retq

	.globl	_main
	.p2align	4, 0x90
_main:
	pushq	%rbp
	movq	%rsp, %rbp
	movq	_OBJC_CLASSLIST_REFERENCES_$_(%rip), %rdi
	callq	_objc_alloc_init
	movq	_OBJC_SELECTOR_REFERENCES_(%rip), %rsi
	movq	%rax, %rdi
	callq	*_objc_msgSend@GOTPCREL(%rip)
	testl	%eax, %eax
	jne	LBB2_3

	movq	_OBJC_CLASSLIST_REFERENCES_$_(%rip), %rdi
	movq	_OBJC_SELECTOR_REFERENCES_.2(%rip), %rsi
	callq	*_objc_msgSend@GOTPCREL(%rip)
	testl	%eax, %eax
	je	LBB2_2
LBB2_3:
	movl	$1, %eax
	popq	%rbp
	retq
LBB2_2:
	xorl	%eax, %eax
	popq	%rbp
	retq

	.section	__TEXT,__objc_classname,cstring_literals
L_OBJC_CLASS_NAME_:
	.asciz	"ZeroGen"

	.section	__TEXT,__objc_methname,cstring_literals
L_OBJC_METH_VAR_NAME_:
	.asciz	"classZero"

	.section	__TEXT,__objc_methtype,cstring_literals
L_OBJC_METH_VAR_TYPE_:
	.asciz	"i16@0:8"

	.section	__DATA,__objc_const
	.p2align	3
__OBJC_$_CLASS_METHODS_ZeroGen:
	.long	24
	.long	1
	.quad	L_OBJC_METH_VAR_NAME_
	.quad	L_OBJC_METH_VAR_TYPE_
	.quad	"+[ZeroGen classZero]"

	.p2align	3
__OBJC_METACLASS_RO_$_ZeroGen:
	.long	1
	.long	40
	.long	40
	.space	4
	.quad	0
	.quad	L_OBJC_CLASS_NAME_
	.quad	__OBJC_$_CLASS_METHODS_ZeroGen
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_METACLASS_$_ZeroGen
	.p2align	3
_OBJC_METACLASS_$_ZeroGen:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_ZeroGen

	.section	__TEXT,__objc_methname,cstring_literals
L_OBJC_METH_VAR_NAME_.1:
	.asciz	"objZero"

	.section	__DATA,__objc_const
	.p2align	3
__OBJC_$_INSTANCE_METHODS_ZeroGen:
	.long	24
	.long	1
	.quad	L_OBJC_METH_VAR_NAME_.1
	.quad	L_OBJC_METH_VAR_TYPE_
	.quad	"-[ZeroGen objZero]"

	.p2align	3
__OBJC_CLASS_RO_$_ZeroGen:
	.long	0
	.long	8
	.long	8
	.space	4
	.quad	0
	.quad	L_OBJC_CLASS_NAME_
	.quad	__OBJC_$_INSTANCE_METHODS_ZeroGen
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_ZeroGen
	.p2align	3
_OBJC_CLASS_$_ZeroGen:
	.quad	_OBJC_METACLASS_$_ZeroGen
	.quad	_OBJC_CLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_CLASS_RO_$_ZeroGen

	.section	__DATA,__objc_classrefs,regular,no_dead_strip
	.p2align	3
_OBJC_CLASSLIST_REFERENCES_$_:
	.quad	_OBJC_CLASS_$_ZeroGen

	.section	__DATA,__objc_selrefs,literal_pointers,no_dead_strip
	.p2align	3
_OBJC_SELECTOR_REFERENCES_:
	.quad	L_OBJC_METH_VAR_NAME_.1

	.p2align	3
_OBJC_SELECTOR_REFERENCES_.2:
	.quad	L_OBJC_METH_VAR_NAME_

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3
L_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_ZeroGen

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
