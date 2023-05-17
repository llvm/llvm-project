// Test that we can handle calls to methods on categories.
// The following assembly defines an ObjC class Foo with an instance method
// -foo, then uses a category (Bar) to add an extra instance method -bar.
// The main function calls both -foo and -bar on an instance of Foo to check
// that the calls behave as expected.
//
// RUN: %clang -c -o %t.o %s
// RUN: %llvm_jitlink -preload libobjc.A.dylib %t.o
//
// REQUIRES: jit-compatible-osx-swift-runtime

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0
	.p2align	2
"-[Foo foo]":
	mov	w0, #1
	ret

	.p2align	2
"-[Foo(Bar) bar]":
	mov	w0, #1
	ret

	.globl	_main
	.p2align	2
_main:
	stp	x20, x19, [sp, #-32]!
	stp	x29, x30, [sp, #16]
	add	x29, sp, #16
Lloh0:
	adrp	x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGE
Lloh1:
	ldr	x0, [x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGEOFF]
	bl	_objc_alloc_init
	mov	x19, x0
Lloh2:
	adrp	x8, _OBJC_SELECTOR_REFERENCES_@PAGE
Lloh3:
	ldr	x1, [x8, _OBJC_SELECTOR_REFERENCES_@PAGEOFF]
	bl	_objc_msgSend
	cmp	w0, #1
	b.ne	LBB2_2

Lloh4:
	adrp	x8, _OBJC_SELECTOR_REFERENCES_.3@PAGE
Lloh5:
	ldr	x1, [x8, _OBJC_SELECTOR_REFERENCES_.3@PAGEOFF]
	mov	x0, x19
	bl	_objc_msgSend
	cmp	w0, #1
	cset	w0, ne
	ldp	x29, x30, [sp, #16]
	ldp	x20, x19, [sp], #32
	ret
LBB2_2:
	mov	w0, #1
	ldp	x29, x30, [sp, #16]
	ldp	x20, x19, [sp], #32
	ret
	.loh AdrpLdr	Lloh2, Lloh3
	.loh AdrpLdr	Lloh0, Lloh1
	.loh AdrpLdr	Lloh4, Lloh5

	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:
	.asciz	"Foo"

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__OBJC_METACLASS_RO_$_Foo:
	.long	1
	.long	40
	.long	40
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_METACLASS_$_Foo
	.p2align	3, 0x0
_OBJC_METACLASS_$_Foo:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_Foo

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:
	.asciz	"foo"

	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:
	.asciz	"i16@0:8"

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__OBJC_$_INSTANCE_METHODS_Foo:
	.long	24
	.long	1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[Foo foo]"

	.p2align	3, 0x0
__OBJC_CLASS_RO_$_Foo:
	.long	0
	.long	8
	.long	8
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_INSTANCE_METHODS_Foo
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_Foo
	.p2align	3, 0x0
_OBJC_CLASS_$_Foo:
	.quad	_OBJC_METACLASS_$_Foo
	.quad	_OBJC_CLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_CLASS_RO_$_Foo

	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_.1:
	.asciz	"Bar"

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.2:
	.asciz	"bar"

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Bar:
	.long	24
	.long	1
	.quad	l_OBJC_METH_VAR_NAME_.2
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[Foo(Bar) bar]"

	.p2align	3, 0x0
__OBJC_$_CATEGORY_Foo_$_Bar:
	.quad	l_OBJC_CLASS_NAME_.1
	.quad	_OBJC_CLASS_$_Foo
	.quad	__OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Bar
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	64
	.space	4

	.section	__DATA,__objc_classrefs,regular,no_dead_strip
	.p2align	3, 0x0
_OBJC_CLASSLIST_REFERENCES_$_:
	.quad	_OBJC_CLASS_$_Foo

	.section	__DATA,__objc_selrefs,literal_pointers,no_dead_strip
	.p2align	3, 0x0
_OBJC_SELECTOR_REFERENCES_:
	.quad	l_OBJC_METH_VAR_NAME_

	.p2align	3, 0x0
_OBJC_SELECTOR_REFERENCES_.3:
	.quad	l_OBJC_METH_VAR_NAME_.2

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_Foo

	.section	__DATA,__objc_catlist,regular,no_dead_strip
	.p2align	3, 0x0
l_OBJC_LABEL_CATEGORY_$:
	.quad	__OBJC_$_CATEGORY_Foo_$_Bar

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
