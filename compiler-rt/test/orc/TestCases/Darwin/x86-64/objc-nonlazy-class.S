// Test that we register non-lazy classes by ensuring a +load is called even if
// the class is never referenced.
//
// RUN: %clang -c -o %t.o %s
// RUN: %llvm_jitlink -preload libobjc.A.dylib %t.o
//
// REQUIRES: jit-compatible-osx-swift-runtime

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0	sdk_version 15, 0
	.p2align	4, 0x90                         ## -- Begin function +[C load]
"+[C load]":                            ## @"\01+[C load]"
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	$7, _x(%rip)
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
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
	movl	$0, -4(%rbp)
	movl	_x(%rip), %edx
	movl	$1, %eax
	xorl	%ecx, %ecx
	cmpl	$7, %edx
	cmovel	%ecx, %eax
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	_x                              ## @x
.zerofill __DATA,__common,_x,4,2
	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_C                 ## @"OBJC_CLASS_$_C"
	.p2align	3, 0x0
_OBJC_CLASS_$_C:
	.quad	_OBJC_METACLASS_$_C
	.quad	0
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_CLASS_RO_$_C

	.globl	_OBJC_METACLASS_$_C             ## @"OBJC_METACLASS_$_C"
	.p2align	3, 0x0
_OBJC_METACLASS_$_C:
	.quad	_OBJC_METACLASS_$_C
	.quad	_OBJC_CLASS_$_C
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_C

	.section	__TEXT,__objc_classname,cstring_literals
L_OBJC_CLASS_NAME_:                     ## @OBJC_CLASS_NAME_
	.asciz	"C"

	.section	__TEXT,__objc_methname,cstring_literals
L_OBJC_METH_VAR_NAME_:                  ## @OBJC_METH_VAR_NAME_
	.asciz	"load"

	.section	__TEXT,__objc_methtype,cstring_literals
L_OBJC_METH_VAR_TYPE_:                  ## @OBJC_METH_VAR_TYPE_
	.asciz	"v16@0:8"

	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ## @"_OBJC_$_CLASS_METHODS_C"
__OBJC_$_CLASS_METHODS_C:
	.long	24                              ## 0x18
	.long	1                               ## 0x1
	.quad	L_OBJC_METH_VAR_NAME_
	.quad	L_OBJC_METH_VAR_TYPE_
	.quad	"+[C load]"

	.p2align	3, 0x0                          ## @"_OBJC_METACLASS_RO_$_C"
__OBJC_METACLASS_RO_$_C:
	.long	3                               ## 0x3
	.long	40                              ## 0x28
	.long	40                              ## 0x28
	.space	4
	.quad	0
	.quad	L_OBJC_CLASS_NAME_
	.quad	__OBJC_$_CLASS_METHODS_C
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.p2align	3, 0x0                          ## @"_OBJC_CLASS_RO_$_C"
__OBJC_CLASS_RO_$_C:
	.long	2                               ## 0x2
	.long	0                               ## 0x0
	.long	0                               ## 0x0
	.space	4
	.quad	0
	.quad	L_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0                          ## @"OBJC_LABEL_CLASS_$"
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_C

	.section	__DATA,__objc_nlclslist,regular,no_dead_strip
	.p2align	3, 0x0                          ## @"OBJC_LABEL_NONLAZY_CLASS_$"
l_OBJC_LABEL_NONLAZY_CLASS_$:
	.quad	_OBJC_CLASS_$_C

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
