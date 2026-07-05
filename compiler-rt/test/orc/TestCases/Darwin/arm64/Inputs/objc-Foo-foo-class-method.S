	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0	sdk_version 14, 0
	.p2align	2
"+[Foo foo]":
	mov	w0, wzr
	ret

	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:
	.asciz	"Foo"

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:
	.asciz	"foo"

	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:
	.asciz	"i16@0:8"

	.section	__DATA,__objc_const
	.p2align	3
__OBJC_$_CLASS_METHODS_Foo:
	.long	24
	.long	1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[Foo foo]"

	.p2align	3
__OBJC_METACLASS_RO_$_Foo:
	.long	1
	.long	40
	.long	40
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_CLASS_METHODS_Foo
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_METACLASS_$_Foo
	.p2align	3
_OBJC_METACLASS_$_Foo:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_Foo

	.section	__DATA,__objc_const
	.p2align	3
__OBJC_CLASS_RO_$_Foo:
	.long	0
	.long	8
	.long	8
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_Foo
	.p2align	3
_OBJC_CLASS_$_Foo:
	.quad	_OBJC_METACLASS_$_Foo
	.quad	_OBJC_CLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_CLASS_RO_$_Foo

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_Foo

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
