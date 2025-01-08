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
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	wzr, [x29, #-4]
	adrp	x8, __OBJC_PROTOCOL_REFERENCE_$_NSObject@PAGE
	ldr	x8, [x8, __OBJC_PROTOCOL_REFERENCE_$_NSObject@PAGEOFF]
	str	x8, [sp]                        ; 8-byte Folded Spill
	adrp	x0, l_.str@PAGE
	add	x0, x0, l_.str@PAGEOFF
	bl	_objc_getProtocol
	ldr	x8, [sp]                        ; 8-byte Folded Reload
	subs	x8, x8, x0
	cset	w8, eq
	and	w8, w8, #0x1
	ands	w8, w8, #0x1
	cset	w0, eq
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"NSObject"

	.private_extern	__OBJC_PROTOCOL_$_NSObject ; @"_OBJC_PROTOCOL_$_NSObject"
	.section	__DATA,__data
	.globl	__OBJC_PROTOCOL_$_NSObject
	.weak_definition	__OBJC_PROTOCOL_$_NSObject
	.p2align	3, 0x0
__OBJC_PROTOCOL_$_NSObject:
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	96                              ; 0x60
	.long	0                               ; 0x0
	.quad	0
	.quad	0
	.quad	0

	.private_extern	__OBJC_LABEL_PROTOCOL_$_NSObject ; @"_OBJC_LABEL_PROTOCOL_$_NSObject"
	.section	__DATA,__objc_protolist,coalesced,no_dead_strip
	.globl	__OBJC_LABEL_PROTOCOL_$_NSObject
	.weak_definition	__OBJC_LABEL_PROTOCOL_$_NSObject
	.p2align	3, 0x0
__OBJC_LABEL_PROTOCOL_$_NSObject:
	.quad	__OBJC_PROTOCOL_$_NSObject

	.private_extern	__OBJC_PROTOCOL_REFERENCE_$_NSObject ; @"_OBJC_PROTOCOL_REFERENCE_$_NSObject"
	.section	__DATA,__objc_protorefs,coalesced,no_dead_strip
	.globl	__OBJC_PROTOCOL_REFERENCE_$_NSObject
	.weak_definition	__OBJC_PROTOCOL_REFERENCE_$_NSObject
	.p2align	3, 0x0
__OBJC_PROTOCOL_REFERENCE_$_NSObject:
	.quad	__OBJC_PROTOCOL_$_NSObject

	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"NSObject"

	.no_dead_strip	__OBJC_PROTOCOL_$_NSObject
	.no_dead_strip	__OBJC_LABEL_PROTOCOL_$_NSObject
	.no_dead_strip	__OBJC_PROTOCOL_REFERENCE_$_NSObject
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
