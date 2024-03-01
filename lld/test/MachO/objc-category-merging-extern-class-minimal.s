# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t && cd %t

## Create a dylib with a fake base class to link against
# RUN: llvm-mc -filetype=obj -triple=arm64-ios-simulator -o a64_fakedylib.o a64_fakedylib.s
# RUN: ld64.lld a64_fakedylib.o -o a64_fakedylib.dylib -dylib -arch arm64 -platform_version ios-simulator 14.0 15.0

## Create our main testing dylib - linking against the fake dylib above
# RUN: llvm-mc -filetype=obj -triple=arm64-ios-simulator -o merge_cat_minimal.o merge_cat_minimal.s
# RUN: ld64.lld -dylib -o merge_cat_minimal_no_merge.dylib a64_fakedylib.dylib merge_cat_minimal.o -arch arm64 -platform_version ios-simulator 14.0 15.0
# RUN: ld64.lld -dylib -o merge_cat_minimal_merge.dylib -objc_category_merging a64_fakedylib.dylib merge_cat_minimal.o -arch arm64 -platform_version ios-simulator 14.0 15.0

## Now verify that the flag caused category merging to happen appropriatelly
# RUN: llvm-objdump --objc-meta-data --macho merge_cat_minimal_no_merge.dylib | FileCheck %s --check-prefixes=NO_MERGE_CATS
# RUN: llvm-objdump --objc-meta-data --macho merge_cat_minimal_merge.dylib | FileCheck %s --check-prefixes=MERGE_CATS

#### Check merge categories enabled ###
# Check that the original categories are not there
MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_Category01
MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_Category02

# Check that the merged cateogry is there, in the correct format
MERGE_CATS: __OBJC_$_CATEGORY_MyBaseClass_$_(Category01|Category02)
MERGE_CATS: instanceMethods
MERGE_CATS-NEXT: 24
MERGE_CATS-NEXT: 2
MERGE_CATS-NEXT:   name {{.*}} cat01_InstanceMethod
MERGE_CATS-NEXT:  types {{.*}} v16@0:8
MERGE_CATS-NEXT:    imp -[MyBaseClass(Category01) cat01_InstanceMethod]
MERGE_CATS-NEXT:   name {{.*}} cat02_InstanceMethod
MERGE_CATS-NEXT:  types {{.*}} v16@0:8
MERGE_CATS-NEXT:    imp -[MyBaseClass(Category02) cat02_InstanceMethod]
MERGE_CATS-NEXT:         classMethods 0x0
MERGE_CATS-NEXT:            protocols 0x0
MERGE_CATS-NEXT:   instanceProperties 0x0

#### Check merge categories disabled ###
# Check that the merged category is not there
NO_MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_(Category01|Category02)

# Check that the original categories are there
NO_MERGE_CATS: __OBJC_$_CATEGORY_MyBaseClass_$_Category01
NO_MERGE_CATS: __OBJC_$_CATEGORY_MyBaseClass_$_Category02



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
	.ios_version_min 7, 0
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
	.section	__TEXT,__objc_classname,cstring_literals
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
