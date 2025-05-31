# REQUIRES: aarch64
# RUN: rm -rf %t; mkdir %t && cd %t

############ Test swift category merging into @objc class, with protocol ############
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o cat_swift.o %s
# RUN: %lld -arch arm64 -dylib -o cat_swift.dylib cat_swift.o -objc_category_merging
# RUN: llvm-objdump --objc-meta-data --macho cat_swift.dylib | FileCheck %s --check-prefixes=CHECK-MERGE


; CHECK-MERGE:      Contents of (__DATA_CONST,__objc_classlist) section

; CHECK-MERGE-NEXT: [[#%x,]] 0x[[#%x,]] _OBJC_CLASS_$__TtC11MyTestClass11MyTestClass
; CHECK-MERGE-NEXT:            isa 0x[[#%x,]] _OBJC_METACLASS_$__TtC11MyTestClass11MyTestClass
; CHECK-MERGE-NEXT:     superclass 0x0
; CHECK-MERGE-NEXT:          cache 0x0
; CHECK-MERGE-NEXT:         vtable 0x0
; CHECK-MERGE-NEXT:           data 0x[[#%x,]] (struct class_ro_t *) Swift class
; CHECK-MERGE-NEXT:                     flags 0x80
; CHECK-MERGE-NEXT:             instanceStart 8
; CHECK-MERGE-NEXT:              instanceSize 8
; CHECK-MERGE-NEXT:                  reserved 0x0
; CHECK-MERGE-NEXT:                ivarLayout 0x0
; CHECK-MERGE-NEXT:                      name 0x[[#%x,]] _TtC11MyTestClass11MyTestClass
; CHECK-MERGE-NEXT:               baseMethods 0x[[#%x,]] (struct method_list_t *)
; CHECK-MERGE-NEXT: 		   entsize 24
; CHECK-MERGE-NEXT: 		     count 1
; CHECK-MERGE-NEXT: 		      name 0x[[#%x,]] init
; CHECK-MERGE-NEXT: 		     types 0x[[#%x,]] @16@0:8
; CHECK-MERGE-NEXT: 		       imp _$s11MyTestClassAACABycfcTo
; CHECK-MERGE-NEXT:             baseProtocols 0x0
; CHECK-MERGE-NEXT:                     ivars 0x0
; CHECK-MERGE-NEXT:            weakIvarLayout 0x0
; CHECK-MERGE-NEXT:            baseProperties 0x0
; CHECK-MERGE-NEXT: Meta Class
; CHECK-MERGE-NEXT:            isa 0x0
; CHECK-MERGE-NEXT:     superclass 0x0
; CHECK-MERGE-NEXT:          cache 0x0
; CHECK-MERGE-NEXT:         vtable 0x0
; CHECK-MERGE-NEXT:           data 0x[[#%x,]] (struct class_ro_t *)
; CHECK-MERGE-NEXT:                     flags 0x81 RO_META
; CHECK-MERGE-NEXT:             instanceStart 40
; CHECK-MERGE-NEXT:              instanceSize 40
; CHECK-MERGE-NEXT:                  reserved 0x0
; CHECK-MERGE-NEXT:                ivarLayout 0x0
; CHECK-MERGE-NEXT:                      name 0x[[#%x,]] _TtC11MyTestClass11MyTestClass
; CHECK-MERGE-NEXT:               baseMethods 0x0 (struct method_list_t *)
; CHECK-MERGE-NEXT:             baseProtocols 0x0
; CHECK-MERGE-NEXT:                     ivars 0x0
; CHECK-MERGE-NEXT:            weakIvarLayout 0x0
; CHECK-MERGE-NEXT:            baseProperties 0x0


;  ================== Generated from Swift: ==================
;; > xcrun swiftc --version
;; swift-driver version: 1.109.2 Apple Swift version 6.0 (swiftlang-6.0.0.3.300 clang-1600.0.20.10)
;; > xcrun swiftc -S MyTestClass.swift -o MyTestClass.s
;;
; import Foundation
;
; protocol MyProtocol {
;     func protocolMethod()
; }
;
; @objc class MyTestClass: NSObject, MyProtocol {
;     func protocolMethod() {
;     }
; }
;
; extension MyTestClass {
;     public func extensionMethod() {
;     }
; }
;  ================== Generated from Swift: ==================


	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 10, 0
	.globl	_main
	.p2align	2
_main:
	.cfi_startproc
	mov	w0, #0
	ret
	.cfi_endproc

	.private_extern	_$s11MyTestClassAAC14protocolMethodyyF
	.globl	_$s11MyTestClassAAC14protocolMethodyyF
	.p2align	2
_$s11MyTestClassAAC14protocolMethodyyF:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11MyTestClassAACABycfC
	.globl	_$s11MyTestClassAACABycfC
	.p2align	2
_$s11MyTestClassAACABycfC:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11MyTestClassAACABycfc
	.globl	_$s11MyTestClassAACABycfc
	.p2align	2
_$s11MyTestClassAACABycfc:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11MyTestClassAACMa
	.globl	_$s11MyTestClassAACMa
	.p2align	2
_$s11MyTestClassAACMa:
	ret

	.p2align	2
_$s11MyTestClassAACABycfcTo:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11MyTestClassAACfD
	.globl	_$s11MyTestClassAACfD
	.p2align	2
_$s11MyTestClassAACfD:
	.cfi_startproc
	ret
	.cfi_endproc

	.p2align	2
_$s11MyTestClassAACAA0A8ProtocolA2aCP14protocolMethodyyFTW:
	.cfi_startproc
	ret
	.cfi_endproc

	.private_extern	_$s11MyTestClassAAC15extensionMethodyyF
	.globl	_$s11MyTestClassAAC15extensionMethodyyF
	.p2align	2
_$s11MyTestClassAAC15extensionMethodyyF:
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

	.private_extern	_$s11MyTestClassAACAA0A8ProtocolAAMc
	.section	__TEXT,__const
	.globl	_$s11MyTestClassAACAA0A8ProtocolAAMc
	.p2align	2, 0x0
_$s11MyTestClassAACAA0A8ProtocolAAMc:
	.long	_$s11MyTestClass0A8ProtocolMp-_$s11MyTestClassAACAA0A8ProtocolAAMc
	.long	(_$s11MyTestClassAACMn-_$s11MyTestClassAACAA0A8ProtocolAAMc)-4
	.long	(_$s11MyTestClassAACAA0A8ProtocolAAWP-_$s11MyTestClassAACAA0A8ProtocolAAMc)-8
	.long	0

	.private_extern	_$s11MyTestClassAACAA0A8ProtocolAAWP
	.section	__DATA,__const
	.globl	_$s11MyTestClassAACAA0A8ProtocolAAWP
	.p2align	3, 0x0
_$s11MyTestClassAACAA0A8ProtocolAAWP:
	.quad	_$s11MyTestClassAACAA0A8ProtocolAAMc
	.quad	_$s11MyTestClassAACAA0A8ProtocolA2aCP14protocolMethodyyFTW

	.section	__TEXT,__swift5_entry,regular,no_dead_strip
	.p2align	2, 0x0
l_entry_point:
	.long	_main-l_entry_point
	.long	0

	.private_extern	"_symbolic $s11MyTestClass0A8ProtocolP"
	.section	__TEXT,__swift5_typeref
	.globl	"_symbolic $s11MyTestClass0A8ProtocolP"
	.weak_definition	"_symbolic $s11MyTestClass0A8ProtocolP"
	.p2align	1, 0x0
"_symbolic $s11MyTestClass0A8ProtocolP":
	.ascii	"$s11MyTestClass0A8ProtocolP"
	.byte	0

	.section	__TEXT,__swift5_fieldmd
	.p2align	2, 0x0
_$s11MyTestClass0A8Protocol_pMF:
	.long	"_symbolic $s11MyTestClass0A8ProtocolP"-_$s11MyTestClass0A8Protocol_pMF
	.long	0
	.short	4
	.short	12
	.long	0

	.section	__TEXT,__const
l_.str.11.MyTestClass:
	.asciz	"MyTestClass"

	.private_extern	_$s11MyTestClassMXM
	.section	__TEXT,__constg_swiftt
	.globl	_$s11MyTestClassMXM
	.weak_definition	_$s11MyTestClassMXM
	.p2align	2, 0x0
_$s11MyTestClassMXM:
	.long	0
	.long	0
	.long	(l_.str.11.MyTestClass-_$s11MyTestClassMXM)-8

	.section	__TEXT,__const
l_.str.10.MyProtocol:
	.asciz	"MyProtocol"

	.private_extern	_$s11MyTestClass0A8ProtocolMp
	.section	__TEXT,__constg_swiftt
	.globl	_$s11MyTestClass0A8ProtocolMp
	.p2align	2, 0x0
_$s11MyTestClass0A8ProtocolMp:
	.long	65603
	.long	(_$s11MyTestClassMXM-_$s11MyTestClass0A8ProtocolMp)-4
	.long	(l_.str.10.MyProtocol-_$s11MyTestClass0A8ProtocolMp)-8
	.long	0
	.long	1
	.long	0
	.long	17
	.long	0

	.private_extern	_OBJC_METACLASS_$__TtC11MyTestClass11MyTestClass
	.section	__DATA,__data
	.globl	_OBJC_METACLASS_$__TtC11MyTestClass11MyTestClass
	.p2align	3, 0x0
_OBJC_METACLASS_$__TtC11MyTestClass11MyTestClass:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__METACLASS_DATA__TtC11MyTestClass11MyTestClass

	.section	__TEXT,__cstring,cstring_literals
	.p2align	4, 0x0
l_.str.30._TtC11MyTestClass11MyTestClass:
	.asciz	"_TtC11MyTestClass11MyTestClass"

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__METACLASS_DATA__TtC11MyTestClass11MyTestClass:
	.long	129
	.long	40
	.long	40
	.long	0
	.quad	0
	.quad	l_.str.30._TtC11MyTestClass11MyTestClass
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__TEXT,__cstring,cstring_literals
"l_.str.7.@16@0:8":
	.asciz	"@16@0:8"

	.section	__DATA,__objc_data
	.p2align	3, 0x0
__INSTANCE_METHODS__TtC11MyTestClass11MyTestClass:
	.long	24
	.long	1
	.quad	"L_selector_data(init)"
	.quad	"l_.str.7.@16@0:8"
	.quad	_$s11MyTestClassAACABycfcTo

	.p2align	3, 0x0
__DATA__TtC11MyTestClass11MyTestClass:
	.long	128
	.long	8
	.long	8
	.long	0
	.quad	0
	.quad	l_.str.30._TtC11MyTestClass11MyTestClass
	.quad	__INSTANCE_METHODS__TtC11MyTestClass11MyTestClass
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.private_extern	"_symbolic So8NSObjectC"
	.section	__TEXT,__swift5_typeref
	.globl	"_symbolic So8NSObjectC"
	.weak_definition	"_symbolic So8NSObjectC"
	.p2align	1, 0x0
"_symbolic So8NSObjectC":
	.ascii	"So8NSObjectC"
	.byte	0

	.private_extern	_$s11MyTestClassAACMn
	.section	__TEXT,__constg_swiftt
	.globl	_$s11MyTestClassAACMn
	.p2align	2, 0x0
_$s11MyTestClassAACMn:
	.long	2147483728
	.long	(_$s11MyTestClassMXM-_$s11MyTestClassAACMn)-4
	.long	(l_.str.11.MyTestClass-_$s11MyTestClassAACMn)-8
	.long	(_$s11MyTestClassAACMa-_$s11MyTestClassAACMn)-12
	.long	(_$s11MyTestClassAACMF-_$s11MyTestClassAACMn)-16
	.long	("_symbolic So8NSObjectC"-_$s11MyTestClassAACMn)-20
	.long	3
	.long	11
	.long	1
	.long	0
	.long	10
	.long	10
	.long	1
	.long	16
	.long	(_$s11MyTestClassAAC14protocolMethodyyF-_$s11MyTestClassAACMn)-56

	.section	__DATA,__objc_data
	.p2align	3, 0x0
_$s11MyTestClassAACMf:
	.quad	0
	.quad	_$s11MyTestClassAACfD
	.quad	_$sBOWV
	.quad	_OBJC_METACLASS_$__TtC11MyTestClass11MyTestClass
	.quad	_OBJC_CLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__DATA__TtC11MyTestClass11MyTestClass+2
	.long	0
	.long	0
	.long	8
	.short	7
	.short	0
	.long	112
	.long	24
	.quad	_$s11MyTestClassAACMn
	.quad	0
	.quad	_$s11MyTestClassAAC14protocolMethodyyF

	.private_extern	"_symbolic _____ 11MyTestClassAAC"
	.section	__TEXT,__swift5_typeref
	.globl	"_symbolic _____ 11MyTestClassAAC"
	.weak_definition	"_symbolic _____ 11MyTestClassAAC"
	.p2align	1, 0x0
"_symbolic _____ 11MyTestClassAAC":
	.byte	1
	.long	(_$s11MyTestClassAACMn-"_symbolic _____ 11MyTestClassAAC")-1
	.byte	0

	.section	__TEXT,__swift5_fieldmd
	.p2align	2, 0x0
_$s11MyTestClassAACMF:
	.long	"_symbolic _____ 11MyTestClassAAC"-_$s11MyTestClassAACMF
	.long	("_symbolic So8NSObjectC"-_$s11MyTestClassAACMF)-4
	.short	7
	.short	12
	.long	0

	.section	__TEXT,__swift5_protos
	.p2align	2, 0x0
l_$s11MyTestClass0A8ProtocolHr:
	.long	_$s11MyTestClass0A8ProtocolMp-l_$s11MyTestClass0A8ProtocolHr

	.section	__TEXT,__swift5_proto
	.p2align	2, 0x0
l_$s11MyTestClassAACAA0A8ProtocolAAHc:
	.long	_$s11MyTestClassAACAA0A8ProtocolAAMc-l_$s11MyTestClassAACAA0A8ProtocolAAHc

	.section	__TEXT,__swift5_types
	.p2align	2, 0x0
l_$s11MyTestClassAACHn:
	.long	_$s11MyTestClassAACMn-l_$s11MyTestClassAACHn

	.private_extern	___swift_reflection_version
	.section	__TEXT,__const
	.globl	___swift_reflection_version
	.weak_definition	___swift_reflection_version
	.p2align	1, 0x0
___swift_reflection_version:
	.short	3

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0
_objc_classes_$s11MyTestClassAACN:
	.quad	_$s11MyTestClassAACN

	.no_dead_strip	_main
	.no_dead_strip	l_entry_point
	.no_dead_strip	_$s11MyTestClass0A8Protocol_pMF
	.no_dead_strip	_$s11MyTestClassAACMF
	.no_dead_strip	__swift_FORCE_LOAD_$_swiftFoundation_$_MyTestClass
	.no_dead_strip	__swift_FORCE_LOAD_$_swiftDarwin_$_MyTestClass
	.no_dead_strip	__swift_FORCE_LOAD_$_swiftObjectiveC_$_MyTestClass
	.no_dead_strip	__swift_FORCE_LOAD_$_swiftCoreFoundation_$_MyTestClass
	.no_dead_strip	__swift_FORCE_LOAD_$_swiftDispatch_$_MyTestClass
	.no_dead_strip	__swift_FORCE_LOAD_$_swiftXPC_$_MyTestClass
	.no_dead_strip	__swift_FORCE_LOAD_$_swiftIOKit_$_MyTestClass
	.no_dead_strip	l_$s11MyTestClass0A8ProtocolHr
	.no_dead_strip	l_$s11MyTestClassAACAA0A8ProtocolAAHc
	.no_dead_strip	l_$s11MyTestClassAACHn
	.no_dead_strip	___swift_reflection_version
	.no_dead_strip	_objc_classes_$s11MyTestClassAACN
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	100665152

	.globl	_$s11MyTestClass0A8ProtocolTL
	.private_extern	_$s11MyTestClass0A8ProtocolTL
	.alt_entry	_$s11MyTestClass0A8ProtocolTL
.set _$s11MyTestClass0A8ProtocolTL, (_$s11MyTestClass0A8ProtocolMp+24)-8
	.globl	_$s11MyTestClassAAC14protocolMethodyyFTq
	.private_extern	_$s11MyTestClassAAC14protocolMethodyyFTq
	.alt_entry	_$s11MyTestClassAAC14protocolMethodyyFTq
.set _$s11MyTestClassAAC14protocolMethodyyFTq, _$s11MyTestClassAACMn+52
	.globl	_$s11MyTestClassAACN
	.private_extern	_$s11MyTestClassAACN
	.alt_entry	_$s11MyTestClassAACN
.set _$s11MyTestClassAACN, _$s11MyTestClassAACMf+24
	.globl	_OBJC_CLASS_$__TtC11MyTestClass11MyTestClass
	.private_extern	_OBJC_CLASS_$__TtC11MyTestClass11MyTestClass
.set _OBJC_CLASS_$__TtC11MyTestClass11MyTestClass, _$s11MyTestClassAACN
	.weak_reference __swift_FORCE_LOAD_$_swiftFoundation
	.weak_reference __swift_FORCE_LOAD_$_swiftDarwin
	.weak_reference __swift_FORCE_LOAD_$_swiftObjectiveC
	.weak_reference __swift_FORCE_LOAD_$_swiftCoreFoundation
	.weak_reference __swift_FORCE_LOAD_$_swiftDispatch
	.weak_reference __swift_FORCE_LOAD_$_swiftXPC
	.weak_reference __swift_FORCE_LOAD_$_swiftIOKit
.subsections_via_symbols

_OBJC_CLASS_$_NSObject:
_OBJC_METACLASS_$_NSObject:
__objc_empty_cache:
_$sBOWV:
  .quad 0
