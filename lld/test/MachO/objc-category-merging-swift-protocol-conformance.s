# REQUIRES: aarch64

## Regression test for the crash reported at
## https://github.com/llvm/llvm-project/pull/95124#issuecomment-4267900795
## A Swift class is extended both by a Swift extension that adds an @objc
## protocol conformance and by an ObjC category. The Swift extension's
## category references the class through its Swift metadata symbol
## (_$s4ModA3FooCN) while the ObjC category references it through the aliased
## ObjC class symbol (_OBJC_CLASS_$__TtC4ModA3Foo). The two symbols get two
## separate categoryMap entries, so mergeCategoriesIntoBaseClass runs twice
## for the same class, and the second invocation re-parses the protocol list
## that the first one emitted, which always carries a trailing null pointer.
## The merger used to infer the expected protocol list layout from symbol
## names ("Swift" lists were assumed to have no trailing null) and crashed
## on the mismatch.

# RUN: rm -rf %t; split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o ModA.o ModA.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o ModB.o ModB.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o extras.o extras.s
# RUN: %lld -arch arm64 -dylib -o out.dylib ModA.o ModB.o extras.o -objc_category_merging -undefined dynamic_lookup
# RUN: llvm-objdump --objc-meta-data --macho out.dylib | FileCheck %s

## Both categories must be merged into the class, with the @objc protocol
## conformance present on both the class and the metaclass.
# CHECK:      Contents of (__DATA_CONST,__objc_classlist) section
# CHECK-NEXT: {{.*}} _OBJC_CLASS_$__TtC4ModA3Foo
# CHECK-NEXT:            isa {{.*}} _OBJC_METACLASS_$__TtC4ModA3Foo
# CHECK-NEXT:     superclass 0x0 _OBJC_CLASS_$_NSObject
# CHECK-NEXT:          cache 0x0
# CHECK-NEXT:         vtable 0x0
# CHECK-NEXT:           data {{.*}} (struct class_ro_t *) Swift class
# CHECK-NEXT:                     flags 0x80
# CHECK-NEXT:             instanceStart 8
# CHECK-NEXT:              instanceSize 8
# CHECK-NEXT:                  reserved 0x0
# CHECK-NEXT:                ivarLayout 0x0
# CHECK-NEXT:                      name {{.*}} _TtC4ModA3Foo
# CHECK-NEXT:               baseMethods {{.*}} (struct method_list_t *)
# CHECK-NEXT:                    entsize 12 (relative)
# CHECK-NEXT:                      count 3
# CHECK-NEXT:                       name {{.*}} cf
# CHECK-NEXT:                      types {{.*}} v16@0:8
# CHECK-NEXT:                        imp {{.*}} -[Foo(CExt) cf]
# CHECK-NEXT:                       name {{.*}} p
# CHECK-NEXT:                      types {{.*}} v16@0:8
# CHECK-NEXT:                        imp {{.*}} _$s4ModA3FooC0A1BE1pyyFTo
# CHECK-NEXT:                       name {{.*}} init
# CHECK-NEXT:                      types {{.*}} @16@0:8
# CHECK-NEXT:                        imp {{.*}} _$s4ModA3FooCACycfcTo
# CHECK-NEXT:             baseProtocols {{.*}}
# CHECK-NEXT:                      count 1
# CHECK-NEXT:                      list[0] {{.*}} (struct protocol_t *)
# CHECK-NEXT:                          isa 0x0
# CHECK-NEXT:                         name {{.*}} _TtP4ModA1P_

# CHECK:      Meta Class
# CHECK-NEXT:            isa 0x0
# CHECK-NEXT:     superclass 0x0 _OBJC_METACLASS_$_NSObject
# CHECK-NEXT:          cache 0x0
# CHECK-NEXT:         vtable 0x0
# CHECK-NEXT:           data {{.*}} (struct class_ro_t *)
# CHECK-NEXT:                     flags 0x81 RO_META
# CHECK-NEXT:             instanceStart 40
# CHECK-NEXT:              instanceSize 40
# CHECK-NEXT:                  reserved 0x0
# CHECK-NEXT:                ivarLayout 0x0
# CHECK-NEXT:                      name {{.*}} _TtC4ModA3Foo
# CHECK-NEXT:               baseMethods 0x0 (struct method_list_t *)
# CHECK-NEXT:             baseProtocols {{.*}}
# CHECK-NEXT:                      count 1
# CHECK-NEXT:                      list[0] {{.*}} (struct protocol_t *)
# CHECK-NEXT:                          isa 0x0
# CHECK-NEXT:                         name {{.*}} _TtP4ModA1P_

## All input categories must have been merged into the class and erased.
# CHECK-NOT: Contents of (__DATA_CONST,__objc_catlist) section

#--- ModA.s
;  ================== Generated from Swift: ==================
; import Foundation
;
; @objc public protocol P {
;     func p()
; }
;
; @objc open class Foo: NSObject {}
;  ===========================================================
; xcrun swiftc -target arm64-apple-macosx11.0 -module-name ModA \
;   -parse-as-library -emit-objc-header-path ModA.h -S ModA.swift -o ModA.s
; (assembly reduced to the class metadata)
	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 26, 2
	.p2align	2
_$s4ModA3FooCACycfcTo:
	ret

	.globl	_$s4ModA3FooCfD
	.p2align	2
_$s4ModA3FooCfD:
	ret

	.section	__TEXT,__objc_methname,cstring_literals
"L_selector_data(init)":
	.asciz	"init"

	.section	__DATA,__data
	.globl	_OBJC_METACLASS_$__TtC4ModA3Foo
	.p2align	3, 0x0
_OBJC_METACLASS_$__TtC4ModA3Foo:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__METACLASS_DATA__TtC4ModA3Foo

	.section	__TEXT,__cstring,cstring_literals
l_.str.13._TtC4ModA3Foo:
	.asciz	"_TtC4ModA3Foo"

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__METACLASS_DATA__TtC4ModA3Foo:
	.long	129
	.long	40
	.long	40
	.long	0
	.quad	0
	.quad	l_.str.13._TtC4ModA3Foo
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
__INSTANCE_METHODS__TtC4ModA3Foo:
	.long	24
	.long	1
	.quad	"L_selector_data(init)"
	.quad	"l_.str.7.@16@0:8"
	.quad	_$s4ModA3FooCACycfcTo

	.p2align	3, 0x0
__DATA__TtC4ModA3Foo:
	.long	128
	.long	8
	.long	8
	.long	0
	.quad	0
	.quad	l_.str.13._TtC4ModA3Foo
	.quad	__INSTANCE_METHODS__TtC4ModA3Foo
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.p2align	3, 0x0
_$s4ModA3FooCMf:
	.quad	0
	.quad	_$s4ModA3FooCfD
	.quad	_$sBOWV
	.quad	_OBJC_METACLASS_$__TtC4ModA3Foo
	.quad	_OBJC_CLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__DATA__TtC4ModA3Foo+2
	.long	0
	.long	0
	.long	8
	.short	7
	.short	0
	.long	104
	.long	24
	.quad	0
	.quad	0

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0
_objc_classes_$s4ModA3FooCN:
	.quad	_$s4ModA3FooCN

	.no_dead_strip	_OBJC_METACLASS_$__TtC4ModA3Foo
	.no_dead_strip	_$s4ModA3FooCN
	.no_dead_strip	_objc_classes_$s4ModA3FooCN
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	100796224

	.globl	_$s4ModA3FooCN
	.alt_entry	_$s4ModA3FooCN
.set _$s4ModA3FooCN, _$s4ModA3FooCMf+24
	.globl	_OBJC_CLASS_$__TtC4ModA3Foo
.set _OBJC_CLASS_$__TtC4ModA3Foo, _$s4ModA3FooCN
.subsections_via_symbols

#--- ModB.s
;  ================== Generated from Swift: ==================
; import Foundation
; import ModA
;
; extension Foo: P {
;     public func p() {}
; }
;  ===========================================================
; xcrun swiftc -target arm64-apple-macosx11.0 -module-name ModB \
;   -parse-as-library -I. -S ModB.swift -o ModB.s
; (assembly reduced to the category metadata)
	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 26, 2
	.p2align	2
_$s4ModA3FooC0A1BE1pyyFTo:
	ret

	.section	__TEXT,__cstring,cstring_literals
l_.str.4.ModB:
	.asciz	"ModB"

	.section	__TEXT,__objc_methname,cstring_literals
"L_selector_data(p)":
	.asciz	"p"

	.section	__TEXT,__cstring,cstring_literals
"l_.str.7.v16@0:8":
	.asciz	"v16@0:8"

	.section	__DATA,__objc_data
	.p2align	3, 0x0
__CATEGORY_INSTANCE_METHODS__TtC4ModA3Foo_$_ModB:
	.long	24
	.long	1
	.quad	"L_selector_data(p)"
	.quad	"l_.str.7.v16@0:8"
	.quad	_$s4ModA3FooC0A1BE1pyyFTo

	.section	__TEXT,__cstring,cstring_literals
l_.str.12._TtP4ModA1P_:
	.asciz	"_TtP4ModA1P_"

	.private_extern	__PROTOCOL__TtP4ModA1P_
	.section	__DATA,__objc_const
	.globl	__PROTOCOL__TtP4ModA1P_
	.weak_definition	__PROTOCOL__TtP4ModA1P_
	.p2align	3, 0x0
__PROTOCOL__TtP4ModA1P_:
	.quad	0
	.quad	l_.str.12._TtP4ModA1P_
	.quad	0
	.quad	__PROTOCOL_INSTANCE_METHODS__TtP4ModA1P_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	96
	.long	1
	.quad	__PROTOCOL_METHOD_TYPES__TtP4ModA1P_
	.quad	0
	.quad	0

	.private_extern	l_OBJC_LABEL_PROTOCOL_$__TtP4ModA1P_
	.section	__DATA,__objc_protolist,coalesced,no_dead_strip
	.globl	l_OBJC_LABEL_PROTOCOL_$__TtP4ModA1P_
	.weak_definition	l_OBJC_LABEL_PROTOCOL_$__TtP4ModA1P_
	.p2align	3, 0x0
l_OBJC_LABEL_PROTOCOL_$__TtP4ModA1P_:
	.quad	__PROTOCOL__TtP4ModA1P_

	.private_extern	__PROTOCOL_INSTANCE_METHODS__TtP4ModA1P_
	.section	__DATA,__objc_data
	.globl	__PROTOCOL_INSTANCE_METHODS__TtP4ModA1P_
	.weak_definition	__PROTOCOL_INSTANCE_METHODS__TtP4ModA1P_
	.p2align	3, 0x0
__PROTOCOL_INSTANCE_METHODS__TtP4ModA1P_:
	.long	24
	.long	1
	.quad	"L_selector_data(p)"
	.quad	"l_.str.7.v16@0:8"
	.quad	0

	.private_extern	__PROTOCOL_METHOD_TYPES__TtP4ModA1P_
	.section	__DATA,__objc_const
	.globl	__PROTOCOL_METHOD_TYPES__TtP4ModA1P_
	.weak_definition	__PROTOCOL_METHOD_TYPES__TtP4ModA1P_
	.p2align	3, 0x0
__PROTOCOL_METHOD_TYPES__TtP4ModA1P_:
	.quad	"l_.str.7.v16@0:8"

;; Note that the protocol list does not have a trailing null pointer, unlike
;; clang-generated (and category-merger-generated) protocol lists.
	.section	__DATA,__objc_const
	.p2align	3, 0x0
__CATEGORY_PROTOCOLS__TtC4ModA3Foo_$_ModB:
	.quad	1
	.quad	__PROTOCOL__TtP4ModA1P_

	.p2align	3, 0x0
__CATEGORY__TtC4ModA3Foo_$_ModB:
	.quad	l_.str.4.ModB
	.quad	_$s4ModA3FooCN
	.quad	__CATEGORY_INSTANCE_METHODS__TtC4ModA3Foo_$_ModB
	.quad	0
	.quad	__CATEGORY_PROTOCOLS__TtC4ModA3Foo_$_ModB
	.quad	0
	.quad	0
	.long	60
	.space	4

	.section	__DATA,__objc_catlist,regular,no_dead_strip
	.p2align	3, 0x0
_objc_categories:
	.quad	__CATEGORY__TtC4ModA3Foo_$_ModB

	.no_dead_strip	_$s4ModA3FooC0A1BE1pyyFTo
	.no_dead_strip	l_OBJC_LABEL_PROTOCOL_$__TtP4ModA1P_
	.no_dead_strip	_objc_categories
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	100796224
.subsections_via_symbols

#--- extras.s
;  ================== Generated from ObjC: ==================
; #import <Foundation/Foundation.h>
; #import "ModA.h"
;
; @interface Foo (CExt)
; - (void)cf;
; @end
;
; @implementation Foo (CExt)
; - (void)cf {}
; @end
;  ==========================================================
; xcrun clang -target arm64-apple-macosx11.0 -S extras.m -o extras.s
; (method body removed)
	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 26, 2
	.p2align	2                               ; -- Begin function -[Foo(CExt) cf]
"-[Foo(CExt) cf]":                      ; @"\01-[Foo(CExt) cf]"
	.cfi_startproc
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"CExt"

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:                  ; @OBJC_METH_VAR_NAME_
	.asciz	"cf"

	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:                  ; @OBJC_METH_VAR_TYPE_
	.asciz	"v16@0:8"

	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_INSTANCE_METHODS__TtC4ModA3Foo_$_CExt"
__OBJC_$_CATEGORY_INSTANCE_METHODS__TtC4ModA3Foo_$_CExt:
	.long	24                              ; 0x18
	.long	1                               ; 0x1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[Foo(CExt) cf]"

	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY__TtC4ModA3Foo_$_CExt"
__OBJC_$_CATEGORY__TtC4ModA3Foo_$_CExt:
	.quad	l_OBJC_CLASS_NAME_
	.quad	_OBJC_CLASS_$__TtC4ModA3Foo
	.quad	__OBJC_$_CATEGORY_INSTANCE_METHODS__TtC4ModA3Foo_$_CExt
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	64                              ; 0x40
	.space	4

	.section	__DATA,__objc_catlist,regular,no_dead_strip
	.p2align	3, 0x0                          ; @"OBJC_LABEL_CATEGORY_$"
l_OBJC_LABEL_CATEGORY_$:
	.quad	__OBJC_$_CATEGORY__TtC4ModA3Foo_$_CExt

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
