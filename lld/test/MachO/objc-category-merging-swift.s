# REQUIRES: aarch64
# RUN: rm -rf %t; mkdir %t && cd %t

############ Test merging multiple categories into a single category ############
## Apply category merging to swiftc code just make sure we can handle addends
## and don't erase category names for swift -- in order to not crash
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o cat_swift.o %s
# RUN: %lld -arch arm64 -dylib -o cat_swift.dylib cat_swift.o -objc_category_merging -no_objc_relative_method_lists
# RUN: llvm-objdump --objc-meta-data --macho cat_swift.dylib | FileCheck %s --check-prefixes=CHECK-MERGE

; CHECK-MERGE:      Contents of (__DATA_CONST,__objc_classlist) section
; CHECK-MERGE-NEXT: _$s11SimpleClassAACN
; CHECK-MERGE-NEXT:            isa {{.+}} _OBJC_METACLASS_$__TtC11SimpleClass11SimpleClass
; CHECK-MERGE-NEXT:     superclass 0x0
; CHECK-MERGE-NEXT:          cache 0x0
; CHECK-MERGE-NEXT:         vtable 0x0
; CHECK-MERGE-NEXT:           data {{.+}} (struct class_ro_t *) Swift class
; CHECK-MERGE-NEXT:                     flags 0x80
; CHECK-MERGE-NEXT:             instanceStart 8
; CHECK-MERGE-NEXT:              instanceSize 8
; CHECK-MERGE-NEXT:                  reserved 0x0
; CHECK-MERGE-NEXT:                ivarLayout 0x0
; CHECK-MERGE-NEXT:                      name {{.+}} _TtC11SimpleClass11SimpleClass
; CHECK-MERGE-NEXT:               baseMethods {{.+}} (struct method_list_t *)
; CHECK-MERGE-NEXT:                    entsize 24
; CHECK-MERGE-NEXT:                      count 3
; CHECK-MERGE-NEXT:                       name {{.+}} categoryInstanceMethod
; CHECK-MERGE-NEXT:                      types {{.+}} q16@0:8
; CHECK-MERGE-NEXT:                        imp _$s11SimpleClassAAC22categoryInstanceMethodSiyFTo
; CHECK-MERGE-NEXT:                       name {{.+}} baseClassInstanceMethod
; CHECK-MERGE-NEXT:                      types {{.+}} i16@0:8
; CHECK-MERGE-NEXT:                        imp _$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyFTo
; CHECK-MERGE-NEXT:                       name {{.+}} init
; CHECK-MERGE-NEXT:                      types {{.+}} @16@0:8
; CHECK-MERGE-NEXT:                        imp _$s11SimpleClassAACABycfcTo
; CHECK-MERGE-NEXT:             baseProtocols 0x0
; CHECK-MERGE-NEXT:                     ivars 0x0
; CHECK-MERGE-NEXT:            weakIvarLayout 0x0
; CHECK-MERGE-NEXT:            baseProperties 0x0
; CHECK-MERGE-NEXT: Meta Class
; CHECK-MERGE-NEXT:            isa 0x0
; CHECK-MERGE-NEXT:     superclass 0x0
; CHECK-MERGE-NEXT:          cache 0x0
; CHECK-MERGE-NEXT:         vtable 0x0
; CHECK-MERGE-NEXT:           data {{.+}} (struct class_ro_t *)
; CHECK-MERGE-NEXT:                     flags 0x81 RO_META
; CHECK-MERGE-NEXT:             instanceStart 40
; CHECK-MERGE-NEXT:              instanceSize 40
; CHECK-MERGE-NEXT:                  reserved 0x0
; CHECK-MERGE-NEXT:                ivarLayout 0x0
; CHECK-MERGE-NEXT:                      name {{.+}} _TtC11SimpleClass11SimpleClass
; CHECK-MERGE-NEXT:               baseMethods 0x0 (struct method_list_t *)
; CHECK-MERGE-NEXT:             baseProtocols 0x0
; CHECK-MERGE-NEXT:                     ivars 0x0
; CHECK-MERGE-NEXT:            weakIvarLayout 0x0
; CHECK-MERGE-NEXT:            baseProperties 0x0
; CHECK-MERGE-NEXT: Contents of (__DATA_CONST,__objc_imageinfo) section
; CHECK-MERGE-NEXT:   version 0
; CHECK-MERGE-NEXT:     flags 0x740 OBJC_IMAGE_HAS_CATEGORY_CLASS_PROPERTIES Swift 5 or later

;  ================== Generated from Swift: ==================
;; > xcrun swiftc --version
;; swift-driver version: 1.109.2 Apple Swift version 6.0 (swiftlang-6.0.0.3.300 clang-1600.0.20.10)
;; > xcrun swiftc -S SimpleClass.swift -o SimpleClass.s
; import Foundation
; @objc class SimpleClass: NSObject {
;     @objc func baseClassInstanceMethod() -> Int32 {
;         return 2
;     }
; }
; extension SimpleClass {
;     @objc func categoryInstanceMethod() -> Int {
;         return 3
;     }
; }

;  ================== Generated from Swift: ==================
	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 12, 0
	.globl	_main
	.p2align	2
_main:
	.cfi_startproc
	mov	w0, #0
	ret
	.cfi_endproc

	.private_extern	_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyF
	.globl	_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyF
	.p2align	2
_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyF:
	.cfi_startproc
	ret
	.cfi_endproc

	.p2align	2
_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyFTo:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11SimpleClassAACABycfC
	.globl	_$s11SimpleClassAACABycfC
	.p2align	2
_$s11SimpleClassAACABycfC:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11SimpleClassAACABycfc
	.globl	_$s11SimpleClassAACABycfc
	.p2align	2
_$s11SimpleClassAACABycfc:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11SimpleClassAACMa
	.globl	_$s11SimpleClassAACMa
	.p2align	2
_$s11SimpleClassAACMa:
	ret

	.p2align	2
_$s11SimpleClassAACABycfcTo:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11SimpleClassAACfD
	.globl	_$s11SimpleClassAACfD
	.p2align	2
_$s11SimpleClassAACfD:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11SimpleClassAAC22categoryInstanceMethodSiyF
	.globl	_$s11SimpleClassAAC22categoryInstanceMethodSiyF
	.p2align	2
_$s11SimpleClassAAC22categoryInstanceMethodSiyF:
	.cfi_startproc
	ret
	.cfi_endproc

	.p2align	2
_$s11SimpleClassAAC22categoryInstanceMethodSiyFTo:
	.cfi_startproc
	ret
	.cfi_endproc

	.section	__TEXT,__objc_methname,cstring_literals
"L_selector_data(init)":
	.asciz	"init"

	.section	__DATA,__objc_selrefs,literal_pointers,no_dead_strip
	.p2align	3, 0x0
"L_selector(init)":
	.quad	"L_selector_data(init)"

	.section	__TEXT,__objc_methname,cstring_literals
"L_selector_data(dealloc)":
	.asciz	"dealloc"

	.section	__DATA,__objc_selrefs,literal_pointers,no_dead_strip
	.p2align	3, 0x0
"L_selector(dealloc)":
	.quad	"L_selector_data(dealloc)"

	.section	__TEXT,__swift5_entry,regular,no_dead_strip
	.p2align	2, 0x0
l_entry_point:
	.long	_main-l_entry_point
	.long	0

	.private_extern	_OBJC_METACLASS_$__TtC11SimpleClass11SimpleClass
	.section	__DATA,__data
	.globl	_OBJC_METACLASS_$__TtC11SimpleClass11SimpleClass
	.p2align	3, 0x0
_OBJC_METACLASS_$__TtC11SimpleClass11SimpleClass:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__METACLASS_DATA__TtC11SimpleClass11SimpleClass

	.section	__TEXT,__cstring,cstring_literals
	.p2align	4, 0x0
l_.str.30._TtC11SimpleClass11SimpleClass:
	.asciz	"_TtC11SimpleClass11SimpleClass"

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__METACLASS_DATA__TtC11SimpleClass11SimpleClass:
	.long	129
	.long	40
	.long	40
	.long	0
	.quad	0
	.quad	l_.str.30._TtC11SimpleClass11SimpleClass
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__TEXT,__objc_methname,cstring_literals
"L_selector_data(baseClassInstanceMethod)":
	.asciz	"baseClassInstanceMethod"

	.section	__TEXT,__cstring,cstring_literals
"l_.str.7.i16@0:8":
	.asciz	"i16@0:8"

"l_.str.7.@16@0:8":
	.asciz	"@16@0:8"

	.section	__DATA,__objc_data
	.p2align	3, 0x0
__INSTANCE_METHODS__TtC11SimpleClass11SimpleClass:
	.long	24
	.long	2
	.quad	"L_selector_data(baseClassInstanceMethod)"
	.quad	"l_.str.7.i16@0:8"
	.quad	_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyFTo
	.quad	"L_selector_data(init)"
	.quad	"l_.str.7.@16@0:8"
	.quad	_$s11SimpleClassAACABycfcTo

	.p2align	3, 0x0
__DATA__TtC11SimpleClass11SimpleClass:
	.long	128
	.long	8
	.long	8
	.long	0
	.quad	0
	.quad	l_.str.30._TtC11SimpleClass11SimpleClass
	.quad	__INSTANCE_METHODS__TtC11SimpleClass11SimpleClass
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__TEXT,__const
l_.str.11.SimpleClass:
	.asciz	"SimpleClass"

	.private_extern	_$s11SimpleClassMXM
	.section	__TEXT,__constg_swiftt
	.globl	_$s11SimpleClassMXM
	.weak_definition	_$s11SimpleClassMXM
	.p2align	2, 0x0
_$s11SimpleClassMXM:
	.long	0
	.long	0
	.long	(l_.str.11.SimpleClass-_$s11SimpleClassMXM)-8

	.private_extern	"_symbolic So8NSObjectC"
	.section	__TEXT,__swift5_typeref
	.globl	"_symbolic So8NSObjectC"
	.weak_definition	"_symbolic So8NSObjectC"
	.p2align	1, 0x0
"_symbolic So8NSObjectC":
	.ascii	"So8NSObjectC"
	.byte	0

	.private_extern	_$s11SimpleClassAACMn
	.section	__TEXT,__constg_swiftt
	.globl	_$s11SimpleClassAACMn
	.p2align	2, 0x0
_$s11SimpleClassAACMn:
	.long	2147483728
	.long	(_$s11SimpleClassMXM-_$s11SimpleClassAACMn)-4
	.long	(l_.str.11.SimpleClass-_$s11SimpleClassAACMn)-8
	.long	(_$s11SimpleClassAACMa-_$s11SimpleClassAACMn)-12
	.long	(_$s11SimpleClassAACMF-_$s11SimpleClassAACMn)-16
	.long	("_symbolic So8NSObjectC"-_$s11SimpleClassAACMn)-20
	.long	3
	.long	11
	.long	1
	.long	0
	.long	10
	.long	10
	.long	1
	.long	16
	.long	(_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyF-_$s11SimpleClassAACMn)-56

	.section	__DATA,__objc_data
	.p2align	3, 0x0
_$s11SimpleClassAACMf:
	.quad	0
	.quad	_$s11SimpleClassAACfD
	.quad	_$sBOWV
	.quad	_OBJC_METACLASS_$__TtC11SimpleClass11SimpleClass
	.quad	_OBJC_CLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__DATA__TtC11SimpleClass11SimpleClass+2
	.long	0
	.long	0
	.long	8
	.short	7
	.short	0
	.long	112
	.long	24
	.quad	_$s11SimpleClassAACMn
	.quad	0
	.quad	_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyF

	.private_extern	"_symbolic _____ 11SimpleClassAAC"
	.section	__TEXT,__swift5_typeref
	.globl	"_symbolic _____ 11SimpleClassAAC"
	.weak_definition	"_symbolic _____ 11SimpleClassAAC"
	.p2align	1, 0x0
"_symbolic _____ 11SimpleClassAAC":
	.byte	1
	.long	(_$s11SimpleClassAACMn-"_symbolic _____ 11SimpleClassAAC")-1
	.byte	0

	.section	__TEXT,__swift5_fieldmd
	.p2align	2, 0x0
_$s11SimpleClassAACMF:
	.long	"_symbolic _____ 11SimpleClassAAC"-_$s11SimpleClassAACMF
	.long	("_symbolic So8NSObjectC"-_$s11SimpleClassAACMF)-4
	.short	7
	.short	12
	.long	0

	.section	__TEXT,__objc_methname,cstring_literals
"L_selector_data(categoryInstanceMethod)":
	.asciz	"categoryInstanceMethod"

	.section	__TEXT,__cstring,cstring_literals
"l_.str.7.q16@0:8":
	.asciz	"q16@0:8"

	.section	__DATA,__objc_data
	.p2align	3, 0x0
__CATEGORY_INSTANCE_METHODS__TtC11SimpleClass11SimpleClass_$_SimpleClass:
	.long	24
	.long	1
	.quad	"L_selector_data(categoryInstanceMethod)"
	.quad	"l_.str.7.q16@0:8"
	.quad	_$s11SimpleClassAAC22categoryInstanceMethodSiyFTo

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__CATEGORY__TtC11SimpleClass11SimpleClass_$_SimpleClass:
	.quad	l_.str.11.SimpleClass
	.quad	_$s11SimpleClassAACMf+24
	.quad	__CATEGORY_INSTANCE_METHODS__TtC11SimpleClass11SimpleClass_$_SimpleClass
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	60
	.space	4

	.section	__TEXT,__swift5_types
	.p2align	2, 0x0
l_$s11SimpleClassAACHn:
	.long	_$s11SimpleClassAACMn-l_$s11SimpleClassAACHn

	.private_extern	___swift_reflection_version
	.section	__TEXT,__const
	.globl	___swift_reflection_version
	.weak_definition	___swift_reflection_version
	.p2align	1, 0x0
___swift_reflection_version:
	.short	3

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0
_objc_classes_$s11SimpleClassAACN:
	.quad	_$s11SimpleClassAACN

	.section	__DATA,__objc_catlist,regular,no_dead_strip
	.p2align	3, 0x0
_objc_categories:
	.quad	__CATEGORY__TtC11SimpleClass11SimpleClass_$_SimpleClass

	.no_dead_strip	_main
	.no_dead_strip	l_entry_point
	.no_dead_strip	_$s11SimpleClassAACMF
	.no_dead_strip	l_$s11SimpleClassAACHn
	.no_dead_strip	___swift_reflection_version
	.no_dead_strip	_objc_classes_$s11SimpleClassAACN
	.no_dead_strip	_objc_categories
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	100665152

	.globl	_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyFTq
	.private_extern	_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyFTq
	.alt_entry	_$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyFTq
.set _$s11SimpleClassAAC04baseB14InstanceMethods5Int32VyFTq, _$s11SimpleClassAACMn+52
	.globl	_$s11SimpleClassAACN
	.private_extern	_$s11SimpleClassAACN
	.alt_entry	_$s11SimpleClassAACN
.set _$s11SimpleClassAACN, _$s11SimpleClassAACMf+24
	.globl	_OBJC_CLASS_$__TtC11SimpleClass11SimpleClass
	.private_extern	_OBJC_CLASS_$__TtC11SimpleClass11SimpleClass
.subsections_via_symbols

_OBJC_CLASS_$_NSObject:
_OBJC_METACLASS_$_NSObject:
__objc_empty_cache:
_$sBOWV:
  .quad 0
