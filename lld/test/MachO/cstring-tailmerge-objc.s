; REQUIRES: aarch64
; RUN: rm -rf %t && split-file %s %t

; Test that ObjC method names are tail merged and
; ObjCSelRefsHelper::makeSelRef() still works correctly

; RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o
; RUN: %lld -dylib -arch arm64 --tail-merge-strings %t/a.o -o %t/a
; RUN: llvm-objdump --macho --section="__TEXT,__objc_methname" %t/a | FileCheck %s --implicit-check-not=error

; RUN: %lld -dylib -arch arm64 --no-tail-merge-strings %t/a.o -o %t/nomerge
; RUN: llvm-objdump --macho --section="__TEXT,__objc_methname" %t/nomerge | FileCheck %s --check-prefixes=CHECK,NOMERGE --implicit-check-not=error

; CHECK: withBar:error:
; NOMERGE: error:

;--- a.mm
__attribute__((objc_root_class))
@interface Foo
- (void)withBar:(int)bar error:(int)error;
- (void)error:(int)error;
@end

@implementation Foo
- (void)withBar:(int)bar error:(int)error {}
- (void)error:(int)error {}
@end

void *_objc_empty_cache;
void *_objc_empty_vtable;
;--- gen
clang -Oz -target arm64-apple-darwin a.mm -S -o -
;--- a.s
	.build_version macos, 11, 0
	.section	__TEXT,__text,regular,pure_instructions
	.p2align	2                               ; -- Begin function -[Foo withBar:error:]
"-[Foo withBar:error:]":                ; @"\01-[Foo withBar:error:]"
	.cfi_startproc
; %bb.0:
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[Foo error:]
"-[Foo error:]":                        ; @"\01-[Foo error:]"
	.cfi_startproc
; %bb.0:
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__objc_empty_vtable             ; @_objc_empty_vtable
.zerofill __DATA,__common,__objc_empty_vtable,8,3
	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_Foo               ; @"OBJC_CLASS_$_Foo"
	.p2align	3, 0x0
_OBJC_CLASS_$_Foo:
	.quad	_OBJC_METACLASS_$_Foo
	.quad	0
	.quad	__objc_empty_cache
	.quad	__objc_empty_vtable
	.quad	__OBJC_CLASS_RO_$_Foo

	.globl	_OBJC_METACLASS_$_Foo           ; @"OBJC_METACLASS_$_Foo"
	.p2align	3, 0x0
_OBJC_METACLASS_$_Foo:
	.quad	_OBJC_METACLASS_$_Foo
	.quad	_OBJC_CLASS_$_Foo
	.quad	__objc_empty_cache
	.quad	__objc_empty_vtable
	.quad	__OBJC_METACLASS_RO_$_Foo

	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"Foo"

	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_METACLASS_RO_$_Foo"
__OBJC_METACLASS_RO_$_Foo:
	.long	3                               ; 0x3
	.long	40                              ; 0x28
	.long	40                              ; 0x28
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:                  ; @OBJC_METH_VAR_NAME_
	.asciz	"withBar:error:"

	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:                  ; @OBJC_METH_VAR_TYPE_
	.asciz	"v24@0:8i16i20"

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.1:                ; @OBJC_METH_VAR_NAME_.1
	.asciz	"error:"

	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_.2:                ; @OBJC_METH_VAR_TYPE_.2
	.asciz	"v20@0:8i16"

	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_INSTANCE_METHODS_Foo"
__OBJC_$_INSTANCE_METHODS_Foo:
	.long	24                              ; 0x18
	.long	2                               ; 0x2
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[Foo withBar:error:]"
	.quad	l_OBJC_METH_VAR_NAME_.1
	.quad	l_OBJC_METH_VAR_TYPE_.2
	.quad	"-[Foo error:]"

	.p2align	3, 0x0                          ; @"_OBJC_CLASS_RO_$_Foo"
__OBJC_CLASS_RO_$_Foo:
	.long	2                               ; 0x2
	.long	0                               ; 0x0
	.long	0                               ; 0x0
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_INSTANCE_METHODS_Foo
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.globl	__objc_empty_cache              ; @_objc_empty_cache
.zerofill __DATA,__common,__objc_empty_cache,8,3
	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0                          ; @"OBJC_LABEL_CLASS_$"
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_Foo

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
