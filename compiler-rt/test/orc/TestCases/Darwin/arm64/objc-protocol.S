// Test that we register protocol metadata. Without registration, messaging
// a protocol metatype object crashes.
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
	mov	w8, #0                          ; =0x0
	str	w8, [sp, #8]                    ; 4-byte Folded Spill
	stur	wzr, [x29, #-4]
	adrp	x8, __OBJC_PROTOCOL_REFERENCE_$_P@PAGE
	ldr	x0, [x8, __OBJC_PROTOCOL_REFERENCE_$_P@PAGEOFF]
	bl	_objc_retain
	ldr	w0, [sp, #8]                    ; 4-byte Folded Reload
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"P"

	.private_extern	__OBJC_PROTOCOL_$_P     ; @"_OBJC_PROTOCOL_$_P"
	.section	__DATA,__data
	.globl	__OBJC_PROTOCOL_$_P
	.weak_definition	__OBJC_PROTOCOL_$_P
	.p2align	3, 0x0
__OBJC_PROTOCOL_$_P:
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

	.private_extern	__OBJC_LABEL_PROTOCOL_$_P ; @"_OBJC_LABEL_PROTOCOL_$_P"
	.section	__DATA,__objc_protolist,coalesced,no_dead_strip
	.globl	__OBJC_LABEL_PROTOCOL_$_P
	.weak_definition	__OBJC_LABEL_PROTOCOL_$_P
	.p2align	3, 0x0
__OBJC_LABEL_PROTOCOL_$_P:
	.quad	__OBJC_PROTOCOL_$_P

	.private_extern	__OBJC_PROTOCOL_REFERENCE_$_P ; @"_OBJC_PROTOCOL_REFERENCE_$_P"
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
