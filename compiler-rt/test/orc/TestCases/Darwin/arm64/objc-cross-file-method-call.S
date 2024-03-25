// Test that calls to Objective-C methods in other files work.
//
// RUN: %clang -c -o %t.o %s
// RUN: %clang -c -o %t.aux.o %S/Inputs/objc-Foo-foo-class-method.S
// RUN: %llvm_jitlink -preload libobjc.A.dylib %t.o %t.aux.o
//
// REQUIRES: jit-compatible-osx-swift-runtime

        .section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0	sdk_version 14, 0
	.globl	_main
	.p2align	2
_main:
	adrp	x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGE
	ldr	x0, [x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGEOFF]
	adrp	x8, _OBJC_SELECTOR_REFERENCES_@PAGE
	ldr	x1, [x8, _OBJC_SELECTOR_REFERENCES_@PAGEOFF]
	b	_objc_msgSend

	.section	__DATA,__objc_classrefs,regular,no_dead_strip
	.p2align	3, 0x0
_OBJC_CLASSLIST_REFERENCES_$_:
	.quad	_OBJC_CLASS_$_Foo

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:
	.asciz	"foo"

	.section	__DATA,__objc_selrefs,literal_pointers,no_dead_strip
	.p2align	3, 0x0
_OBJC_SELECTOR_REFERENCES_:
	.quad	l_OBJC_METH_VAR_NAME_

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
