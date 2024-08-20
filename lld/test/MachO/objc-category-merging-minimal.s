# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t && cd %t

############ Test merging multiple categories into a single category ############
## Create a dylib with a fake base class to link against in when merging between categories
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o a64_fakedylib.o a64_fakedylib.s
# RUN: %lld -arch arm64 a64_fakedylib.o -o a64_fakedylib.dylib -dylib

## Create our main testing dylib - linking against the fake dylib above
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o merge_cat_minimal.o merge_cat_minimal.s
# RUN: %lld -arch arm64 -dylib -o merge_cat_minimal_no_merge.dylib a64_fakedylib.dylib merge_cat_minimal.o
# RUN: %lld -arch arm64 -dylib -o merge_cat_minimal_merge.dylib -objc_category_merging a64_fakedylib.dylib merge_cat_minimal.o

## Now verify that the flag caused category merging to happen appropriatelly
# RUN: llvm-objdump --objc-meta-data --macho merge_cat_minimal_no_merge.dylib | FileCheck %s --check-prefixes=NO_MERGE_CATS
# RUN: llvm-objdump --objc-meta-data --macho merge_cat_minimal_merge.dylib | FileCheck %s --check-prefixes=MERGE_CATS

############ Test merging multiple categories into the base class ############
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o merge_base_class_minimal.o merge_base_class_minimal.s
# RUN: %lld -arch arm64 -dylib -o merge_base_class_minimal_yes_merge.dylib -objc_category_merging merge_base_class_minimal.o merge_cat_minimal.o
# RUN: %lld -arch arm64 -dylib -o merge_base_class_minimal_no_merge.dylib merge_base_class_minimal.o merge_cat_minimal.o

# RUN: llvm-objdump --objc-meta-data --macho merge_base_class_minimal_no_merge.dylib  | FileCheck %s --check-prefixes=NO_MERGE_INTO_BASE
# RUN: llvm-objdump --objc-meta-data --macho merge_base_class_minimal_yes_merge.dylib | FileCheck %s --check-prefixes=YES_MERGE_INTO_BASE

############ Test merging swift category into the base class ############
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o MyBaseClassSwiftExtension.o MyBaseClassSwiftExtension.s
# RUN: %lld -no_objc_relative_method_lists -arch arm64 -dylib -o merge_base_class_swift_minimal_yes_merge.dylib -objc_category_merging MyBaseClassSwiftExtension.o merge_base_class_minimal.o
# RUN: llvm-objdump --objc-meta-data --macho merge_base_class_swift_minimal_yes_merge.dylib | FileCheck %s --check-prefixes=YES_MERGE_INTO_BASE_SWIFT

#### Check merge categories enabled ###
# Check that the original categories are not there
MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_Category01
MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_Category02

# Check that the merged cateogry is there, in the correct format
MERGE_CATS: __OBJC_$_CATEGORY_MyBaseClass(Category01|Category02)
MERGE_CATS-NEXT:   name {{.*}} Category01|Category02
MERGE_CATS:       instanceMethods
MERGE_CATS-NEXT:  entsize 12 (relative)
MERGE_CATS-NEXT:  count 2
MERGE_CATS-NEXT:   name {{.*}} cat01_InstanceMethod
MERGE_CATS-NEXT:  types {{.*}} v16@0:8
MERGE_CATS-NEXT:    imp {{.*}} -[MyBaseClass(Category01) cat01_InstanceMethod]
MERGE_CATS-NEXT:   name {{.*}} cat02_InstanceMethod
MERGE_CATS-NEXT:  types {{.*}} v16@0:8
MERGE_CATS-NEXT:    imp {{.*}} -[MyBaseClass(Category02) cat02_InstanceMethod]
MERGE_CATS-NEXT:         classMethods 0x0
MERGE_CATS-NEXT:            protocols 0x0
MERGE_CATS-NEXT:   instanceProperties 0x0

#### Check merge categories disabled ###
# Check that the merged category is not there
NO_MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass(Category01|Category02)

# Check that the original categories are there
NO_MERGE_CATS: __OBJC_$_CATEGORY_MyBaseClass_$_Category01
NO_MERGE_CATS: __OBJC_$_CATEGORY_MyBaseClass_$_Category02


#### Check merge cateogires into base class is disabled ####
NO_MERGE_INTO_BASE: __OBJC_$_CATEGORY_MyBaseClass_$_Category01
NO_MERGE_INTO_BASE: __OBJC_$_CATEGORY_MyBaseClass_$_Category02

#### Check merge cateogires into base class is enabled and categories are merged into base class ####
YES_MERGE_INTO_BASE-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_Category01
YES_MERGE_INTO_BASE-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_Category02

YES_MERGE_INTO_BASE: _OBJC_CLASS_$_MyBaseClass
YES_MERGE_INTO_BASE-NEXT: _OBJC_METACLASS_$_MyBaseClass
YES_MERGE_INTO_BASE: baseMethods
YES_MERGE_INTO_BASE-NEXT: entsize 12 (relative)
YES_MERGE_INTO_BASE-NEXT: count 3
YES_MERGE_INTO_BASE-NEXT: name {{.*}} cat01_InstanceMethod
YES_MERGE_INTO_BASE-NEXT: types {{.*}} v16@0:8
YES_MERGE_INTO_BASE-NEXT: imp {{.*}} -[MyBaseClass(Category01) cat01_InstanceMethod]
YES_MERGE_INTO_BASE-NEXT: name {{.*}} cat02_InstanceMethod
YES_MERGE_INTO_BASE-NEXT: types {{.*}} v16@0:8
YES_MERGE_INTO_BASE-NEXT: imp {{.*}} -[MyBaseClass(Category02) cat02_InstanceMethod]
YES_MERGE_INTO_BASE-NEXT: name {{.*}} baseInstanceMethod
YES_MERGE_INTO_BASE-NEXT: types {{.*}} v16@0:8
YES_MERGE_INTO_BASE-NEXT: imp {{.*}} -[MyBaseClass baseInstanceMethod]


#### Check merge swift category into base class ###
YES_MERGE_INTO_BASE_SWIFT: _OBJC_CLASS_$_MyBaseClass
YES_MERGE_INTO_BASE_SWIFT-NEXT: _OBJC_METACLASS_$_MyBaseClass
YES_MERGE_INTO_BASE_SWIFT: baseMethods
YES_MERGE_INTO_BASE_SWIFT-NEXT: entsize 24
YES_MERGE_INTO_BASE_SWIFT-NEXT: count 2
YES_MERGE_INTO_BASE_SWIFT-NEXT: name {{.*}} swiftMethod
YES_MERGE_INTO_BASE_SWIFT-NEXT: types {{.*}} v16@0:8
YES_MERGE_INTO_BASE_SWIFT-NEXT: imp _$sSo11MyBaseClassC0abC14SwiftExtensionE11swiftMethodyyFTo
YES_MERGE_INTO_BASE_SWIFT-NEXT: name {{.*}} baseInstanceMethod
YES_MERGE_INTO_BASE_SWIFT-NEXT: types {{.*}} v16@0:8
YES_MERGE_INTO_BASE_SWIFT-NEXT: imp -[MyBaseClass baseInstanceMethod]


#--- a64_fakedylib.s

    .section    __DATA,__objc_data
    .globl    _OBJC_CLASS_$_MyBaseClass
_OBJC_CLASS_$_MyBaseClass:
    .quad    0

#--- merge_cat_minimal.s

;  ================== Generated from ObjC: ==================
; __attribute__((objc_root_class))
; @interface MyBaseClass
; - (void)baseInstanceMethod;
; @end
;
; @interface MyBaseClass(Category01)
; - (void)cat01_InstanceMethod;
; @end
;
; @implementation MyBaseClass(Category01)
; - (void)cat01_InstanceMethod {}
; @end
;
; @interface MyBaseClass(Category02)
; - (void)cat02_InstanceMethod;
; @end
;
; @implementation MyBaseClass(Category02)
; - (void)cat02_InstanceMethod {}
; @end
;  ================== Generated from ObjC: ==================

	.section	__TEXT,__text,regular,pure_instructions
	.p2align	2                               ; -- Begin function -[MyBaseClass(Category01) cat01_InstanceMethod]
"-[MyBaseClass(Category01) cat01_InstanceMethod]": ; @"\01-[MyBaseClass(Category01) cat01_InstanceMethod]"
	.cfi_startproc
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[MyBaseClass(Category02) cat02_InstanceMethod]
"-[MyBaseClass(Category02) cat02_InstanceMethod]": ; @"\01-[MyBaseClass(Category02) cat02_InstanceMethod]"
	.cfi_startproc
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"Category01"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:                  ; @OBJC_METH_VAR_NAME_
	.asciz	"cat01_InstanceMethod"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:                  ; @OBJC_METH_VAR_TYPE_
	.asciz	"v16@0:8"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category01"
__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category01:
	.long	24                              ; 0x18
	.long	1                               ; 0x1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass(Category01) cat01_InstanceMethod]"
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_MyBaseClass_$_Category01"
__OBJC_$_CATEGORY_MyBaseClass_$_Category01:
	.quad	l_OBJC_CLASS_NAME_
	.quad	_OBJC_CLASS_$_MyBaseClass
	.quad	__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category01
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	64                              ; 0x40
	.space	4
	.section	__DATA,__objc_const
l_OBJC_CLASS_NAME_.1:                   ; @OBJC_CLASS_NAME_.1
	.asciz	"Category02"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.2:                ; @OBJC_METH_VAR_NAME_.2
	.asciz	"cat02_InstanceMethod"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category02"
__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category02:
	.long	24                              ; 0x18
	.long	1                               ; 0x1
	.quad	l_OBJC_METH_VAR_NAME_.2
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass(Category02) cat02_InstanceMethod]"
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_MyBaseClass_$_Category02"
__OBJC_$_CATEGORY_MyBaseClass_$_Category02:
	.quad	l_OBJC_CLASS_NAME_.1
	.quad	_OBJC_CLASS_$_MyBaseClass
	.quad	__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category02
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	64                              ; 0x40
	.space	4
	.section	__DATA,__objc_catlist,regular,no_dead_strip
	.p2align	3, 0x0                          ; @"OBJC_LABEL_CATEGORY_$"
l_OBJC_LABEL_CATEGORY_$:
	.quad	__OBJC_$_CATEGORY_MyBaseClass_$_Category01
	.quad	__OBJC_$_CATEGORY_MyBaseClass_$_Category02
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	96
.subsections_via_symbols

.addrsig
.addrsig_sym __OBJC_$_CATEGORY_MyBaseClass_$_Category01

#--- merge_base_class_minimal.s
; clang -c merge_base_class_minimal.mm -O3 -target arm64-apple-macos -arch arm64 -S -o merge_base_class_minimal.s
;  ================== Generated from ObjC: ==================
; __attribute__((objc_root_class))
; @interface MyBaseClass
; - (void)baseInstanceMethod;
; @end
;
; @implementation MyBaseClass
; - (void)baseInstanceMethod {}
; @end
;  ================== Generated from ObjC  ==================
	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0
	.p2align	2
"-[MyBaseClass baseInstanceMethod]":
	.cfi_startproc
; %bb.0:
	ret
	.cfi_endproc
	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_MyBaseClass
	.p2align	3, 0x0
_OBJC_CLASS_$_MyBaseClass:
	.quad	_OBJC_METACLASS_$_MyBaseClass
	.quad	0
	.quad	0
	.quad	0
	.quad	__OBJC_CLASS_RO_$_MyBaseClass
	.globl	_OBJC_METACLASS_$_MyBaseClass
	.p2align	3, 0x0
_OBJC_METACLASS_$_MyBaseClass:
	.quad	_OBJC_METACLASS_$_MyBaseClass
	.quad	_OBJC_CLASS_$_MyBaseClass
	.quad	0
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_MyBaseClass
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:
	.asciz	"MyBaseClass"
	.section	__DATA,__objc_const
	.p2align	3, 0x0
__OBJC_METACLASS_RO_$_MyBaseClass:
	.long	3
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
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:
	.asciz	"baseInstanceMethod"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:
	.asciz	"v16@0:8"
	.section	__DATA,__objc_const
	.p2align	3, 0x0
__OBJC_$_INSTANCE_METHODS_MyBaseClass:
	.long	24
	.long	1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass baseInstanceMethod]"
	.p2align	3, 0x0
__OBJC_CLASS_RO_$_MyBaseClass:
	.long	2
	.long	0
	.long	0
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_INSTANCE_METHODS_MyBaseClass
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_MyBaseClass
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64
.subsections_via_symbols


#--- MyBaseClassSwiftExtension.s
; xcrun -sdk macosx swiftc -emit-assembly MyBaseClassSwiftExtension.swift -import-objc-header YourProject-Bridging-Header.h -o MyBaseClassSwiftExtension.s
;  ================== Generated from Swift: ==================
; import Foundation
; extension MyBaseClass {
;     @objc func swiftMethod() {
;     }
; }
;  ================== Generated from Swift ===================
	.private_extern	_$sSo11MyBaseClassC0abC14SwiftExtensionE11swiftMethodyyF
	.globl	_$sSo11MyBaseClassC0abC14SwiftExtensionE11swiftMethodyyF
	.p2align	2
_$sSo11MyBaseClassC0abC14SwiftExtensionE11swiftMethodyyF:
	.cfi_startproc
	mov	w0, #0
	ret
	.cfi_endproc

	.p2align	2
_$sSo11MyBaseClassC0abC14SwiftExtensionE11swiftMethodyyFTo:
	.cfi_startproc
	mov	w0, #0
	ret
	.cfi_endproc

	.section	__TEXT,__cstring,cstring_literals
	.p2align	4, 0x0
l_.str.25.MyBaseClassSwiftExtension:
	.asciz	"MyBaseClassSwiftExtension"

	.section	__TEXT,__objc_methname,cstring_literals
"L_selector_data(swiftMethod)":
	.asciz	"swiftMethod"

	.section	__TEXT,__cstring,cstring_literals
"l_.str.7.v16@0:8":
	.asciz	"v16@0:8"

	.section	__DATA,__objc_data
	.p2align	3, 0x0
__CATEGORY_INSTANCE_METHODS_MyBaseClass_$_MyBaseClassSwiftExtension:
	.long	24
	.long	1
	.quad	"L_selector_data(swiftMethod)"
	.quad	"l_.str.7.v16@0:8"
	.quad	_$sSo11MyBaseClassC0abC14SwiftExtensionE11swiftMethodyyFTo

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__CATEGORY_MyBaseClass_$_MyBaseClassSwiftExtension:
	.quad	l_.str.25.MyBaseClassSwiftExtension
	.quad	_OBJC_CLASS_$_MyBaseClass
	.quad	__CATEGORY_INSTANCE_METHODS_MyBaseClass_$_MyBaseClassSwiftExtension
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.long	60
	.space	4

	.section	__DATA,__objc_catlist,regular,no_dead_strip
	.p2align	3, 0x0
_objc_categories:
	.quad	__CATEGORY_MyBaseClass_$_MyBaseClassSwiftExtension

	.no_dead_strip	_main
	.no_dead_strip	l_entry_point

.subsections_via_symbols
