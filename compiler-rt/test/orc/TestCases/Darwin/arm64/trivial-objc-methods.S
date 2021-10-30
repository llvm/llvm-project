// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink -dlopen libobjc.A.dylib %t
//
// Test that Objective-C class and instance methods work.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0	sdk_version 12, 0
	.p2align	2                               ; -- Begin function +[ZeroGen classZero]
"+[ZeroGen classZero]":                 ; @"\01+[ZeroGen classZero]"
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #16                     ; =16
	.cfi_def_cfa_offset 16
	str	x0, [sp, #8]
	str	x1, [sp]
	mov	w0, #0
	add	sp, sp, #16                     ; =16
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[ZeroGen objZero]
"-[ZeroGen objZero]":                   ; @"\01-[ZeroGen objZero]"
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #16                     ; =16
	.cfi_def_cfa_offset 16
	str	x0, [sp, #8]
	str	x1, [sp]
	mov	w0, #0
	add	sp, sp, #16                     ; =16
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48                     ; =48
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	wzr, [x29, #-4]
	adrp	x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGE
	str	x8, [sp, #8]                    ; 8-byte Folded Spill
	ldr	x0, [x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGEOFF]
	bl	_objc_alloc_init
	ldr	x8, [sp, #8]                    ; 8-byte Folded Reload
	str	x0, [sp, #16]
	ldr	x0, [x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGEOFF]
	adrp	x8, _OBJC_SELECTOR_REFERENCES_@PAGE
	ldr	x1, [x8, _OBJC_SELECTOR_REFERENCES_@PAGEOFF]
	bl	_objc_msgSend
	cbz	w0, LBB2_2
; %bb.1:
	mov	w8, #1
	stur	w8, [x29, #-4]
	b	LBB2_5
LBB2_2:
	ldr	x0, [sp, #16]
	adrp	x8, _OBJC_SELECTOR_REFERENCES_.2@PAGE
	ldr	x1, [x8, _OBJC_SELECTOR_REFERENCES_.2@PAGEOFF]
	bl	_objc_msgSend
	cbz	w0, LBB2_4
; %bb.3:
	mov	w8, #1
	stur	w8, [x29, #-4]
	b	LBB2_5
LBB2_4:
	stur	wzr, [x29, #-4]
LBB2_5:
	ldur	w0, [x29, #-4]
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48                     ; =48
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"ZeroGen"

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:                  ; @OBJC_METH_VAR_NAME_
	.asciz	"classZero"

	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:                  ; @OBJC_METH_VAR_TYPE_
	.asciz	"i16@0:8"

	.section	__DATA,__objc_const
	.p2align	3                               ; @"_OBJC_$_CLASS_METHODS_ZeroGen"
__OBJC_$_CLASS_METHODS_ZeroGen:
	.long	24                              ; 0x18
	.long	1                               ; 0x1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[ZeroGen classZero]"

	.p2align	3                               ; @"_OBJC_METACLASS_RO_$_ZeroGen"
__OBJC_METACLASS_RO_$_ZeroGen:
	.long	1                               ; 0x1
	.long	40                              ; 0x28
	.long	40                              ; 0x28
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_CLASS_METHODS_ZeroGen
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_METACLASS_$_ZeroGen       ; @"OBJC_METACLASS_$_ZeroGen"
	.p2align	3
_OBJC_METACLASS_$_ZeroGen:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_ZeroGen

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.1:                ; @OBJC_METH_VAR_NAME_.1
	.asciz	"objZero"

	.section	__DATA,__objc_const
	.p2align	3                               ; @"_OBJC_$_INSTANCE_METHODS_ZeroGen"
__OBJC_$_INSTANCE_METHODS_ZeroGen:
	.long	24                              ; 0x18
	.long	1                               ; 0x1
	.quad	l_OBJC_METH_VAR_NAME_.1
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[ZeroGen objZero]"

	.p2align	3                               ; @"_OBJC_CLASS_RO_$_ZeroGen"
__OBJC_CLASS_RO_$_ZeroGen:
	.long	0                               ; 0x0
	.long	8                               ; 0x8
	.long	8                               ; 0x8
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_INSTANCE_METHODS_ZeroGen
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_ZeroGen           ; @"OBJC_CLASS_$_ZeroGen"
	.p2align	3
_OBJC_CLASS_$_ZeroGen:
	.quad	_OBJC_METACLASS_$_ZeroGen
	.quad	_OBJC_CLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_CLASS_RO_$_ZeroGen

	.section	__DATA,__objc_classrefs,regular,no_dead_strip
	.p2align	3                               ; @"OBJC_CLASSLIST_REFERENCES_$_"
_OBJC_CLASSLIST_REFERENCES_$_:
	.quad	_OBJC_CLASS_$_ZeroGen

	.section	__DATA,__objc_selrefs,literal_pointers,no_dead_strip
	.p2align	3                               ; @OBJC_SELECTOR_REFERENCES_
_OBJC_SELECTOR_REFERENCES_:
	.quad	l_OBJC_METH_VAR_NAME_

	.p2align	3                               ; @OBJC_SELECTOR_REFERENCES_.2
_OBJC_SELECTOR_REFERENCES_.2:
	.quad	l_OBJC_METH_VAR_NAME_.1

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3                               ; @"OBJC_LABEL_CLASS_$"
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_ZeroGen

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
