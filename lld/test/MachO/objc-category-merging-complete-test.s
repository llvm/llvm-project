# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t && cd %t

############ Test merging multiple categories into a single category ############
## Create a dylib to link against(a64_file1.dylib) and merge categories in the main binary (file2_merge_a64.exe)
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o a64_file1.o a64_file1.s
# RUN: %lld -arch arm64 a64_file1.o -o a64_file1.dylib -dylib

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o a64_file2.o a64_file2.s
# RUN: %lld -no_objc_relative_method_lists -arch arm64 -o a64_file2_no_merge.exe a64_file1.dylib a64_file2.o
# RUN: %lld -no_objc_relative_method_lists -arch arm64 -o a64_file2_no_merge_v2.exe a64_file1.dylib a64_file2.o -no_objc_category_merging
# RUN: %lld -no_objc_relative_method_lists -arch arm64 -o a64_file2_no_merge_v3.exe a64_file1.dylib a64_file2.o -objc_category_merging -no_objc_category_merging
# RUN: %lld -no_objc_relative_method_lists -arch arm64 -o a64_file2_merge.exe -objc_category_merging a64_file1.dylib a64_file2.o

# RUN: llvm-objdump --objc-meta-data --macho a64_file2_no_merge.exe | FileCheck %s --check-prefixes=NO_MERGE_CATS
# RUN: llvm-objdump --objc-meta-data --macho a64_file2_no_merge_v2.exe | FileCheck %s --check-prefixes=NO_MERGE_CATS
# RUN: llvm-objdump --objc-meta-data --macho a64_file2_no_merge_v3.exe | FileCheck %s --check-prefixes=NO_MERGE_CATS
# RUN: llvm-objdump --objc-meta-data --macho a64_file2_merge.exe | FileCheck %s --check-prefixes=MERGE_CATS

############ Test merging multiple categories into the base class ############
# RUN: %lld -no_objc_relative_method_lists -arch arm64 -o a64_file2_merge_into_class.exe -objc_category_merging a64_file1.o a64_file2.o
# RUN: llvm-objdump --objc-meta-data --macho a64_file2_merge_into_class.exe | FileCheck %s --check-prefixes=MERGE_CATS_CLS


MERGE_CATS:     __OBJC_$_CATEGORY_MyBaseClass(Category02|Category03)
MERGE_CATS-NEXT:              name {{.*}} Category02|Category03
MERGE_CATS:           instanceMethods
MERGE_CATS-NEXT:           entsize 24
MERGE_CATS-NEXT:             count 4
MERGE_CATS-NEXT:              name {{.*}} class02InstanceMethod
MERGE_CATS-NEXT:             types {{.*}} v16@0:8
MERGE_CATS-NEXT:               imp -[MyBaseClass(Category02) class02InstanceMethod]
MERGE_CATS-NEXT:              name {{.*}} myProtocol02Method
MERGE_CATS-NEXT:             types {{.*}} v16@0:8
MERGE_CATS-NEXT:               imp -[MyBaseClass(Category02) myProtocol02Method]
MERGE_CATS-NEXT:              name {{.*}} class03InstanceMethod
MERGE_CATS-NEXT:             types {{.*}} v16@0:8
MERGE_CATS-NEXT:               imp -[MyBaseClass(Category03) class03InstanceMethod]
MERGE_CATS-NEXT:              name {{.*}} myProtocol03Method
MERGE_CATS-NEXT:             types {{.*}} v16@0:8
MERGE_CATS-NEXT:               imp -[MyBaseClass(Category03) myProtocol03Method]
MERGE_CATS-NEXT:      classMethods {{.*}}
MERGE_CATS-NEXT:           entsize 24
MERGE_CATS-NEXT:             count 4
MERGE_CATS-NEXT:              name {{.*}} class02ClassMethod
MERGE_CATS-NEXT:             types {{.*}} v16@0:8
MERGE_CATS-NEXT:               imp +[MyBaseClass(Category02) class02ClassMethod]
MERGE_CATS-NEXT:              name {{.*}} MyProtocol02Prop
MERGE_CATS-NEXT:             types {{.*}} i16@0:8
MERGE_CATS-NEXT:               imp +[MyBaseClass(Category02) MyProtocol02Prop]
MERGE_CATS-NEXT:              name {{.*}} class03ClassMethod
MERGE_CATS-NEXT:             types {{.*}} v16@0:8
MERGE_CATS-NEXT:               imp +[MyBaseClass(Category03) class03ClassMethod]
MERGE_CATS-NEXT:              name {{.*}} MyProtocol03Prop
MERGE_CATS-NEXT:             types {{.*}} i16@0:8
MERGE_CATS-NEXT:               imp +[MyBaseClass(Category03) MyProtocol03Prop]
MERGE_CATS-NEXT:         protocols
MERGE_CATS-NEXT:                      count 2
MERGE_CATS-NEXT:              list[0] {{.*}} (struct protocol_t *)
MERGE_CATS-NEXT:                  isa 0x0
MERGE_CATS-NEXT:                 name {{.*}} MyProtocol02
MERGE_CATS-NEXT:            protocols 0x0
MERGE_CATS-NEXT:          instanceMethods
MERGE_CATS-NEXT:               entsize 24
MERGE_CATS-NEXT:                 count 2
MERGE_CATS-NEXT:                  name {{.*}} myProtocol02Method
MERGE_CATS-NEXT:                 types {{.*}} v16@0:8
MERGE_CATS-NEXT:                   imp 0x0
MERGE_CATS-NEXT:                  name {{.*}} MyProtocol02Prop
MERGE_CATS-NEXT:                 types {{.*}} i16@0:8
MERGE_CATS-NEXT:                   imp 0x0
MERGE_CATS-NEXT:             classMethods
MERGE_CATS-NEXT:      optionalInstanceMethods 0x0
MERGE_CATS-NEXT:         optionalClassMethods 0x0
MERGE_CATS-NEXT:           instanceProperties {{.*}}
MERGE_CATS-NEXT:              list[1] {{.*}}
MERGE_CATS-NEXT:                  isa 0x0
MERGE_CATS-NEXT:                 name {{.*}} MyProtocol03
MERGE_CATS-NEXT:            protocols 0x0
MERGE_CATS-NEXT:          instanceMethods
MERGE_CATS-NEXT:               entsize 24
MERGE_CATS-NEXT:                 count 2
MERGE_CATS-NEXT:                  name {{.*}} myProtocol03Method
MERGE_CATS-NEXT:                 types {{.*}} v16@0:8
MERGE_CATS-NEXT:                   imp 0x0
MERGE_CATS-NEXT:                  name {{.*}} MyProtocol03Prop
MERGE_CATS-NEXT:                 types {{.*}} i16@0:8
MERGE_CATS-NEXT:                   imp 0x0
MERGE_CATS-NEXT:             classMethods 0x0
MERGE_CATS-NEXT:      optionalInstanceMethods 0x0
MERGE_CATS-NEXT:         optionalClassMethods 0x0
MERGE_CATS-NEXT:           instanceProperties {{.*}}
MERGE_CATS-NEXT:      instanceProperties
MERGE_CATS-NEXT:                    entsize 16
MERGE_CATS-NEXT:                      count 2
MERGE_CATS-NEXT:                 name {{.*}} MyProtocol02Prop
MERGE_CATS-NEXT:            attributes {{.*}} Ti,R,D
MERGE_CATS-NEXT:                 name {{.*}} MyProtocol03Prop
MERGE_CATS-NEXT:            attributes {{.*}} Ti,R,D
MERGE_CATS:        __OBJC_$_CATEGORY_MyBaseClass_$_Category04


NO_MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass(Category02|Category03)
NO_MERGE_CATS: __OBJC_$_CATEGORY_MyBaseClass_$_Category02
NO_MERGE_CATS: instanceMethods
NO_MERGE_CATS-NEXT: 24
NO_MERGE_CATS-NEXT: 2
NO_MERGE_CATS: classMethods
NO_MERGE_CATS-NEXT: 24
NO_MERGE_CATS-NEXT: 2


MERGE_CATS_CLS:        _OBJC_CLASS_$_MyBaseClass
MERGE_CATS_CLS-NEXT:            isa {{.*}} _OBJC_METACLASS_$_MyBaseClass
MERGE_CATS_CLS-NEXT:     superclass 0x0
MERGE_CATS_CLS-NEXT:          cache {{.*}} __objc_empty_cache
MERGE_CATS_CLS-NEXT:         vtable 0x0
MERGE_CATS_CLS-NEXT:           data {{.*}} (struct class_ro_t *)
MERGE_CATS_CLS-NEXT:                     flags 0x2 RO_ROOT
MERGE_CATS_CLS-NEXT:             instanceStart 0
MERGE_CATS_CLS-NEXT:              instanceSize 4
MERGE_CATS_CLS-NEXT:                  reserved 0x0
MERGE_CATS_CLS-NEXT:                ivarLayout 0x0
MERGE_CATS_CLS-NEXT:                      name {{.*}} MyBaseClass
MERGE_CATS_CLS-NEXT:               baseMethods {{.*}} (struct method_list_t *)
MERGE_CATS_CLS-NEXT:            entsize 24
MERGE_CATS_CLS-NEXT:              count 8
MERGE_CATS_CLS-NEXT:               name {{.*}} class02InstanceMethod
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp -[MyBaseClass(Category02) class02InstanceMethod]
MERGE_CATS_CLS-NEXT:               name {{.*}} myProtocol02Method
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp -[MyBaseClass(Category02) myProtocol02Method]
MERGE_CATS_CLS-NEXT:               name {{.*}} class03InstanceMethod
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp -[MyBaseClass(Category03) class03InstanceMethod]
MERGE_CATS_CLS-NEXT:               name {{.*}} myProtocol03Method
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp -[MyBaseClass(Category03) myProtocol03Method]
MERGE_CATS_CLS-NEXT:               name {{.*}} baseInstanceMethod
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp -[MyBaseClass baseInstanceMethod]
MERGE_CATS_CLS-NEXT:               name {{.*}} myProtocol01Method
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp -[MyBaseClass myProtocol01Method]
MERGE_CATS_CLS-NEXT:               name {{.*}} MyProtocol01Prop
MERGE_CATS_CLS-NEXT:              types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                imp -[MyBaseClass MyProtocol01Prop]
MERGE_CATS_CLS-NEXT:               name {{.*}} setMyProtocol01Prop:
MERGE_CATS_CLS-NEXT:              types {{.*}} v20@0:8i16
MERGE_CATS_CLS-NEXT:                imp -[MyBaseClass setMyProtocol01Prop:]
MERGE_CATS_CLS-NEXT:             baseProtocols {{.*}}
MERGE_CATS_CLS-NEXT:                       count 3
MERGE_CATS_CLS-NEXT:               list[0] {{.*}} (struct protocol_t *)
MERGE_CATS_CLS-NEXT:                   isa 0x0
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol02
MERGE_CATS_CLS-NEXT:             protocols 0x0
MERGE_CATS_CLS-NEXT:           instanceMethods {{.*}} (struct method_list_t *)
MERGE_CATS_CLS-NEXT:                entsize 24
MERGE_CATS_CLS-NEXT:                  count 2
MERGE_CATS_CLS-NEXT:                   name {{.*}} myProtocol02Method
MERGE_CATS_CLS-NEXT:                  types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:                   name {{.*}} MyProtocol02Prop
MERGE_CATS_CLS-NEXT:                  types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:              classMethods 0x0 (struct method_list_t *)
MERGE_CATS_CLS-NEXT:       optionalInstanceMethods 0x0
MERGE_CATS_CLS-NEXT:          optionalClassMethods 0x0
MERGE_CATS_CLS-NEXT:            instanceProperties {{.*}}
MERGE_CATS_CLS-NEXT:               list[1] {{.*}} (struct protocol_t *)
MERGE_CATS_CLS-NEXT:                   isa 0x0
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol03
MERGE_CATS_CLS-NEXT:             protocols 0x0
MERGE_CATS_CLS-NEXT:           instanceMethods {{.*}} (struct method_list_t *)
MERGE_CATS_CLS-NEXT:                entsize 24
MERGE_CATS_CLS-NEXT:                  count 2
MERGE_CATS_CLS-NEXT:                   name {{.*}} myProtocol03Method
MERGE_CATS_CLS-NEXT:                  types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:                   name {{.*}} MyProtocol03Prop
MERGE_CATS_CLS-NEXT:                  types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:              classMethods 0x0 (struct method_list_t *)
MERGE_CATS_CLS-NEXT:       optionalInstanceMethods 0x0
MERGE_CATS_CLS-NEXT:          optionalClassMethods 0x0
MERGE_CATS_CLS-NEXT:            instanceProperties {{.*}}
MERGE_CATS_CLS-NEXT:               list[2] {{.*}} (struct protocol_t *)
MERGE_CATS_CLS-NEXT:                   isa 0x0
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol01
MERGE_CATS_CLS-NEXT:             protocols 0x0
MERGE_CATS_CLS-NEXT:           instanceMethods {{.*}} (struct method_list_t *)
MERGE_CATS_CLS-NEXT:                entsize 24
MERGE_CATS_CLS-NEXT:                  count 3
MERGE_CATS_CLS-NEXT:                   name {{.*}} myProtocol01Method
MERGE_CATS_CLS-NEXT:                  types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:                   name {{.*}} MyProtocol01Prop
MERGE_CATS_CLS-NEXT:                  types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:                   name {{.*}} setMyProtocol01Prop:
MERGE_CATS_CLS-NEXT:                  types {{.*}} v20@0:8i16
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:              classMethods 0x0 (struct method_list_t *)
MERGE_CATS_CLS-NEXT:       optionalInstanceMethods 0x0
MERGE_CATS_CLS-NEXT:          optionalClassMethods 0x0
MERGE_CATS_CLS-NEXT:            instanceProperties {{.*}}
MERGE_CATS_CLS-NEXT:                     ivars {{.*}}
MERGE_CATS_CLS-NEXT:                     entsize 32
MERGE_CATS_CLS-NEXT:                       count 1
MERGE_CATS_CLS-NEXT:                offset {{.*}} 0
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol01Prop
MERGE_CATS_CLS-NEXT:                  type {{.*}} i
MERGE_CATS_CLS-NEXT:             alignment 2
MERGE_CATS_CLS-NEXT:                  size 4
MERGE_CATS_CLS-NEXT:            weakIvarLayout 0x0
MERGE_CATS_CLS-NEXT:            baseProperties {{.*}}
MERGE_CATS_CLS-NEXT:                     entsize 16
MERGE_CATS_CLS-NEXT:                       count 3
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol02Prop
MERGE_CATS_CLS-NEXT:             attributes {{.*}} Ti,R,D
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol03Prop
MERGE_CATS_CLS-NEXT:             attributes {{.*}} Ti,R,D
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol01Prop
MERGE_CATS_CLS-NEXT:             attributes {{.*}} Ti,N,VMyProtocol01Prop
MERGE_CATS_CLS-NEXT: Meta Class
MERGE_CATS_CLS-NEXT:            isa {{.*}} _OBJC_METACLASS_$_MyBaseClass
MERGE_CATS_CLS-NEXT:     superclass {{.*}} _OBJC_CLASS_$_MyBaseClass
MERGE_CATS_CLS-NEXT:          cache {{.*}} __objc_empty_cache
MERGE_CATS_CLS-NEXT:         vtable 0x0
MERGE_CATS_CLS-NEXT:           data {{.*}} (struct class_ro_t *)
MERGE_CATS_CLS-NEXT:                     flags 0x3 RO_META RO_ROOT
MERGE_CATS_CLS-NEXT:             instanceStart 40
MERGE_CATS_CLS-NEXT:              instanceSize 40
MERGE_CATS_CLS-NEXT:                  reserved 0x0
MERGE_CATS_CLS-NEXT:                ivarLayout 0x0
MERGE_CATS_CLS-NEXT:                      name {{.*}} MyBaseClass
MERGE_CATS_CLS-NEXT:               baseMethods {{.*}} (struct method_list_t *)
MERGE_CATS_CLS-NEXT:            entsize 24
MERGE_CATS_CLS-NEXT:              count 5
MERGE_CATS_CLS-NEXT:               name {{.*}} class02ClassMethod
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp +[MyBaseClass(Category02) class02ClassMethod]
MERGE_CATS_CLS-NEXT:               name {{.*}} MyProtocol02Prop
MERGE_CATS_CLS-NEXT:              types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                imp +[MyBaseClass(Category02) MyProtocol02Prop]
MERGE_CATS_CLS-NEXT:               name {{.*}} class03ClassMethod
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp +[MyBaseClass(Category03) class03ClassMethod]
MERGE_CATS_CLS-NEXT:               name {{.*}} MyProtocol03Prop
MERGE_CATS_CLS-NEXT:              types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                imp +[MyBaseClass(Category03) MyProtocol03Prop]
MERGE_CATS_CLS-NEXT:               name {{.*}} baseClassMethod
MERGE_CATS_CLS-NEXT:              types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                imp +[MyBaseClass baseClassMethod]
MERGE_CATS_CLS-NEXT:             baseProtocols {{.*}}
MERGE_CATS_CLS-NEXT:                       count 3
MERGE_CATS_CLS-NEXT:               list[0] {{.*}} (struct protocol_t *)
MERGE_CATS_CLS-NEXT:                   isa 0x0
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol02
MERGE_CATS_CLS-NEXT:             protocols 0x0
MERGE_CATS_CLS-NEXT:           instanceMethods {{.*}} (struct method_list_t *)
MERGE_CATS_CLS-NEXT:                entsize 24
MERGE_CATS_CLS-NEXT:                  count 2
MERGE_CATS_CLS-NEXT:                   name {{.*}} myProtocol02Method
MERGE_CATS_CLS-NEXT:                  types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:                   name {{.*}} MyProtocol02Prop
MERGE_CATS_CLS-NEXT:                  types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:              classMethods 0x0 (struct method_list_t *)
MERGE_CATS_CLS-NEXT:       optionalInstanceMethods 0x0
MERGE_CATS_CLS-NEXT:          optionalClassMethods 0x0
MERGE_CATS_CLS-NEXT:            instanceProperties {{.*}}
MERGE_CATS_CLS-NEXT:               list[1] {{.*}} (struct protocol_t *)
MERGE_CATS_CLS-NEXT:                   isa 0x0
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol03
MERGE_CATS_CLS-NEXT:             protocols 0x0
MERGE_CATS_CLS-NEXT:           instanceMethods {{.*}} (struct method_list_t *)
MERGE_CATS_CLS-NEXT:                entsize 24
MERGE_CATS_CLS-NEXT:                  count 2
MERGE_CATS_CLS-NEXT:                   name {{.*}} myProtocol03Method
MERGE_CATS_CLS-NEXT:                  types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:                   name {{.*}} MyProtocol03Prop
MERGE_CATS_CLS-NEXT:                  types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:              classMethods 0x0 (struct method_list_t *)
MERGE_CATS_CLS-NEXT:       optionalInstanceMethods 0x0
MERGE_CATS_CLS-NEXT:          optionalClassMethods 0x0
MERGE_CATS_CLS-NEXT:            instanceProperties {{.*}}
MERGE_CATS_CLS-NEXT:               list[2] {{.*}} (struct protocol_t *)
MERGE_CATS_CLS-NEXT:                   isa 0x0
MERGE_CATS_CLS-NEXT:                  name {{.*}} MyProtocol01
MERGE_CATS_CLS-NEXT:             protocols 0x0
MERGE_CATS_CLS-NEXT:           instanceMethods {{.*}} (struct method_list_t *)
MERGE_CATS_CLS-NEXT:                entsize 24
MERGE_CATS_CLS-NEXT:                  count 3
MERGE_CATS_CLS-NEXT:                   name {{.*}} myProtocol01Method
MERGE_CATS_CLS-NEXT:                  types {{.*}} v16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:                   name {{.*}} MyProtocol01Prop
MERGE_CATS_CLS-NEXT:                  types {{.*}} i16@0:8
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:                   name {{.*}} setMyProtocol01Prop:
MERGE_CATS_CLS-NEXT:                  types {{.*}} v20@0:8i16
MERGE_CATS_CLS-NEXT:                    imp 0x0
MERGE_CATS_CLS-NEXT:              classMethods 0x0 (struct method_list_t *)
MERGE_CATS_CLS-NEXT:       optionalInstanceMethods 0x0
MERGE_CATS_CLS-NEXT:          optionalClassMethods 0x0
MERGE_CATS_CLS-NEXT:            instanceProperties {{.*}}
MERGE_CATS_CLS-NEXT:                     ivars 0x0
MERGE_CATS_CLS-NEXT:            weakIvarLayout 0x0
MERGE_CATS_CLS-NEXT:            baseProperties 0x0
MERGE_CATS_CLS:        __OBJC_$_CATEGORY_MyBaseClass_$_Category04


#--- a64_file1.s

## @protocol MyProtocol01
## - (void)myProtocol01Method;
## @property (nonatomic) int MyProtocol01Prop;
## @end
##
## __attribute__((objc_root_class))
## @interface MyBaseClass<MyProtocol01>
## - (void)baseInstanceMethod;
## - (void)myProtocol01Method;
## + (void)baseClassMethod;
## @end
##
## @implementation MyBaseClass
## @synthesize MyProtocol01Prop;
## - (void)baseInstanceMethod {}
## - (void)myProtocol01Method {}
## + (void)baseClassMethod {}
## @end
##
## void *_objc_empty_cache;

	.section	__TEXT,__text,regular,pure_instructions
	.p2align	2                               ; -- Begin function -[MyBaseClass baseInstanceMethod]
"-[MyBaseClass baseInstanceMethod]":    ; @"\01-[MyBaseClass baseInstanceMethod]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[MyBaseClass myProtocol01Method]
"-[MyBaseClass myProtocol01Method]":    ; @"\01-[MyBaseClass myProtocol01Method]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function +[MyBaseClass baseClassMethod]
"+[MyBaseClass baseClassMethod]":       ; @"\01+[MyBaseClass baseClassMethod]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[MyBaseClass MyProtocol01Prop]
"-[MyBaseClass MyProtocol01Prop]":      ; @"\01-[MyBaseClass MyProtocol01Prop]"
	.cfi_startproc
; %bb.0:                                ; %entry
Lloh0:
	adrp	x8, _OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop@PAGE
Lloh1:
	ldrsw	x8, [x8, _OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop@PAGEOFF]
	ldr	w0, [x0, x8]
	ret
	.loh AdrpLdr	Lloh0, Lloh1
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[MyBaseClass setMyProtocol01Prop:]
"-[MyBaseClass setMyProtocol01Prop:]":  ; @"\01-[MyBaseClass setMyProtocol01Prop:]"
	.cfi_startproc
; %bb.0:                                ; %entry
Lloh2:
	adrp	x8, _OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop@PAGE
Lloh3:
	ldrsw	x8, [x8, _OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop@PAGEOFF]
	str	w2, [x0, x8]
	ret
	.loh AdrpLdr	Lloh2, Lloh3
	.cfi_endproc
                                        ; -- End function
	.private_extern	_OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop ; @"OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop"
	.section	__DATA,__objc_ivar
	.globl	_OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop
	.p2align	2, 0x0
_OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop:
	.long	0                               ; 0x0
	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_MyBaseClass       ; @"OBJC_CLASS_$_MyBaseClass"
	.p2align	3, 0x0
_OBJC_CLASS_$_MyBaseClass:
	.quad	_OBJC_METACLASS_$_MyBaseClass
	.quad	0
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_CLASS_RO_$_MyBaseClass
	.globl	_OBJC_METACLASS_$_MyBaseClass   ; @"OBJC_METACLASS_$_MyBaseClass"
	.p2align	3, 0x0
_OBJC_METACLASS_$_MyBaseClass:
	.quad	_OBJC_METACLASS_$_MyBaseClass
	.quad	_OBJC_CLASS_$_MyBaseClass
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_MyBaseClass
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"MyBaseClass"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:                  ; @OBJC_METH_VAR_NAME_
	.asciz	"baseClassMethod"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:                  ; @OBJC_METH_VAR_TYPE_
	.asciz	"v16@0:8"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_CLASS_METHODS_MyBaseClass"
__OBJC_$_CLASS_METHODS_MyBaseClass:
	.long	24                              ; 0x18
	.long	1                               ; 0x1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[MyBaseClass baseClassMethod]"
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_.1:                   ; @OBJC_CLASS_NAME_.1
	.asciz	"MyProtocol01"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.2:                ; @OBJC_METH_VAR_NAME_.2
	.asciz	"myProtocol01Method"
l_OBJC_METH_VAR_NAME_.3:                ; @OBJC_METH_VAR_NAME_.3
	.asciz	"MyProtocol01Prop"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_.4:                ; @OBJC_METH_VAR_TYPE_.4
	.asciz	"i16@0:8"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.5:                ; @OBJC_METH_VAR_NAME_.5
	.asciz	"setMyProtocol01Prop:"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_.6:                ; @OBJC_METH_VAR_TYPE_.6
	.asciz	"v20@0:8i16"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol01"
__OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol01:
	.long	24                              ; 0x18
	.long	3                               ; 0x3
	.quad	l_OBJC_METH_VAR_NAME_.2
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	0
	.quad	l_OBJC_METH_VAR_NAME_.3
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.quad	0
	.quad	l_OBJC_METH_VAR_NAME_.5
	.quad	l_OBJC_METH_VAR_TYPE_.6
	.quad	0
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_PROP_NAME_ATTR_:                 ; @OBJC_PROP_NAME_ATTR_
	.asciz	"MyProtocol01Prop"
l_OBJC_PROP_NAME_ATTR_.7:               ; @OBJC_PROP_NAME_ATTR_.7
	.asciz	"Ti,N"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_PROP_LIST_MyProtocol01"
__OBJC_$_PROP_LIST_MyProtocol01:
	.long	16                              ; 0x10
	.long	1                               ; 0x1
	.quad	l_OBJC_PROP_NAME_ATTR_
	.quad	l_OBJC_PROP_NAME_ATTR_.7
	.p2align	3, 0x0                          ; @"_OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol01"
__OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol01:
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.quad	l_OBJC_METH_VAR_TYPE_.6
	.private_extern	__OBJC_PROTOCOL_$_MyProtocol01 ; @"_OBJC_PROTOCOL_$_MyProtocol01"
	.section	__DATA,__data
	.globl	__OBJC_PROTOCOL_$_MyProtocol01
	.weak_definition	__OBJC_PROTOCOL_$_MyProtocol01
	.p2align	3, 0x0
__OBJC_PROTOCOL_$_MyProtocol01:
	.quad	0
	.quad	l_OBJC_CLASS_NAME_.1
	.quad	0
	.quad	__OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol01
	.quad	0
	.quad	0
	.quad	0
	.quad	__OBJC_$_PROP_LIST_MyProtocol01
	.long	96                              ; 0x60
	.long	0                               ; 0x0
	.quad	__OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol01
	.quad	0
	.quad	0
	.private_extern	__OBJC_LABEL_PROTOCOL_$_MyProtocol01 ; @"_OBJC_LABEL_PROTOCOL_$_MyProtocol01"
	.section	__DATA,__objc_protolist,coalesced,no_dead_strip
	.globl	__OBJC_LABEL_PROTOCOL_$_MyProtocol01
	.weak_definition	__OBJC_LABEL_PROTOCOL_$_MyProtocol01
	.p2align	3, 0x0
__OBJC_LABEL_PROTOCOL_$_MyProtocol01:
	.quad	__OBJC_PROTOCOL_$_MyProtocol01
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_CLASS_PROTOCOLS_$_MyBaseClass"
__OBJC_CLASS_PROTOCOLS_$_MyBaseClass:
	.quad	1                               ; 0x1
	.quad	__OBJC_PROTOCOL_$_MyProtocol01
	.quad	0
	.p2align	3, 0x0                          ; @"_OBJC_METACLASS_RO_$_MyBaseClass"
__OBJC_METACLASS_RO_$_MyBaseClass:
	.long	3                               ; 0x3
	.long	40                              ; 0x28
	.long	40                              ; 0x28
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_CLASS_METHODS_MyBaseClass
	.quad	__OBJC_CLASS_PROTOCOLS_$_MyBaseClass
	.quad	0
	.quad	0
	.quad	0
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.8:                ; @OBJC_METH_VAR_NAME_.8
	.asciz	"baseInstanceMethod"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_INSTANCE_METHODS_MyBaseClass"
__OBJC_$_INSTANCE_METHODS_MyBaseClass:
	.long	24                              ; 0x18
	.long	4                               ; 0x4
	.quad	l_OBJC_METH_VAR_NAME_.8
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass baseInstanceMethod]"
	.quad	l_OBJC_METH_VAR_NAME_.2
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass myProtocol01Method]"
	.quad	l_OBJC_METH_VAR_NAME_.3
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.quad	"-[MyBaseClass MyProtocol01Prop]"
	.quad	l_OBJC_METH_VAR_NAME_.5
	.quad	l_OBJC_METH_VAR_TYPE_.6
	.quad	"-[MyBaseClass setMyProtocol01Prop:]"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_.9:                ; @OBJC_METH_VAR_TYPE_.9
	.asciz	"i"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_INSTANCE_VARIABLES_MyBaseClass"
__OBJC_$_INSTANCE_VARIABLES_MyBaseClass:
	.long	32                              ; 0x20
	.long	1                               ; 0x1
	.quad	_OBJC_IVAR_$_MyBaseClass.MyProtocol01Prop
	.quad	l_OBJC_METH_VAR_NAME_.3
	.quad	l_OBJC_METH_VAR_TYPE_.9
	.long	2                               ; 0x2
	.long	4                               ; 0x4
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_PROP_NAME_ATTR_.10:              ; @OBJC_PROP_NAME_ATTR_.10
	.asciz	"Ti,N,VMyProtocol01Prop"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_PROP_LIST_MyBaseClass"
__OBJC_$_PROP_LIST_MyBaseClass:
	.long	16                              ; 0x10
	.long	1                               ; 0x1
	.quad	l_OBJC_PROP_NAME_ATTR_
	.quad	l_OBJC_PROP_NAME_ATTR_.10
	.p2align	3, 0x0                          ; @"_OBJC_CLASS_RO_$_MyBaseClass"
__OBJC_CLASS_RO_$_MyBaseClass:
	.long	2                               ; 0x2
	.long	0                               ; 0x0
	.long	4                               ; 0x4
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_INSTANCE_METHODS_MyBaseClass
	.quad	__OBJC_CLASS_PROTOCOLS_$_MyBaseClass
	.quad	__OBJC_$_INSTANCE_VARIABLES_MyBaseClass
	.quad	0
	.quad	__OBJC_$_PROP_LIST_MyBaseClass
	.globl	__objc_empty_cache              ; @_objc_empty_cache
.zerofill __DATA,__common,__objc_empty_cache,8,3
	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0                          ; @"OBJC_LABEL_CLASS_$"
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_MyBaseClass
	.no_dead_strip	__OBJC_LABEL_PROTOCOL_$_MyProtocol01
	.no_dead_strip	__OBJC_PROTOCOL_$_MyProtocol01
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	96
.subsections_via_symbols


#--- a64_file2.s

## @protocol MyProtocol01
## - (void)myProtocol01Method;
## @end
##
## @protocol MyProtocol02
## - (void)myProtocol02Method;
## @property(readonly) int MyProtocol02Prop;
## @end
##
## @protocol MyProtocol03
## - (void)myProtocol03Method;
## @property(readonly) int MyProtocol03Prop;
## @end
##
##
## __attribute__((objc_root_class))
## @interface MyBaseClass<MyProtocol01>
## - (void)baseInstanceMethod;
## - (void)myProtocol01Method;
## + (void)baseClassMethod;
## @end
##
##
##
## @interface MyBaseClass(Category02)<MyProtocol02>
## - (void)class02InstanceMethod;
## - (void)myProtocol02Method;
## + (void)class02ClassMethod;
## + (int)MyProtocol02Prop;
## @end
##
## @implementation MyBaseClass(Category02)
## - (void)class02InstanceMethod {}
## - (void)myProtocol02Method {}
## + (void)class02ClassMethod {}
## + (int)MyProtocol02Prop { return 0;}
## @dynamic MyProtocol02Prop;
## @end
##
## @interface MyBaseClass(Category03)<MyProtocol03>
## - (void)class03InstanceMethod;
## - (void)myProtocol03Method;
## + (void)class03ClassMethod;
## + (int)MyProtocol03Prop;
## @end
##
## @implementation MyBaseClass(Category03)
## - (void)class03InstanceMethod {}
## - (void)myProtocol03Method {}
## + (void)class03ClassMethod {}
## + (int)MyProtocol03Prop { return 0;}
## @dynamic MyProtocol03Prop;
## @end
##
## // This category shouldn't be merged
## @interface MyBaseClass(Category04)
## + (void)load;
## @end
##
## @implementation MyBaseClass(Category04)
## + (void)load {}
## @end
##
## int main() {
##     return 0;
## }


	.section	__TEXT,__text,regular,pure_instructions
	.p2align	2                               ; -- Begin function -[MyBaseClass(Category02) class02InstanceMethod]
"-[MyBaseClass(Category02) class02InstanceMethod]": ; @"\01-[MyBaseClass(Category02) class02InstanceMethod]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[MyBaseClass(Category02) myProtocol02Method]
"-[MyBaseClass(Category02) myProtocol02Method]": ; @"\01-[MyBaseClass(Category02) myProtocol02Method]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function +[MyBaseClass(Category02) class02ClassMethod]
"+[MyBaseClass(Category02) class02ClassMethod]": ; @"\01+[MyBaseClass(Category02) class02ClassMethod]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function +[MyBaseClass(Category02) MyProtocol02Prop]
"+[MyBaseClass(Category02) MyProtocol02Prop]": ; @"\01+[MyBaseClass(Category02) MyProtocol02Prop]"
	.cfi_startproc
; %bb.0:                                ; %entry
	b	_OUTLINED_FUNCTION_0
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[MyBaseClass(Category03) class03InstanceMethod]
"-[MyBaseClass(Category03) class03InstanceMethod]": ; @"\01-[MyBaseClass(Category03) class03InstanceMethod]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function -[MyBaseClass(Category03) myProtocol03Method]
"-[MyBaseClass(Category03) myProtocol03Method]": ; @"\01-[MyBaseClass(Category03) myProtocol03Method]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function +[MyBaseClass(Category03) class03ClassMethod]
"+[MyBaseClass(Category03) class03ClassMethod]": ; @"\01+[MyBaseClass(Category03) class03ClassMethod]"
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function +[MyBaseClass(Category03) MyProtocol03Prop]
"+[MyBaseClass(Category03) MyProtocol03Prop]": ; @"\01+[MyBaseClass(Category03) MyProtocol03Prop]"
	.cfi_startproc
; %bb.0:                                ; %entry
	b	_OUTLINED_FUNCTION_0
	.cfi_endproc
                                        ; -- End function
	.p2align	2
"+[MyBaseClass(Category04) load]":
	.cfi_startproc
; %bb.0:
	ret
	.cfi_endproc
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:                                ; %entry
	b	_OUTLINED_FUNCTION_0
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function OUTLINED_FUNCTION_0
_OUTLINED_FUNCTION_0:                   ; @OUTLINED_FUNCTION_0 Tail Call
	.cfi_startproc
; %bb.0:
	mov	w0, #0
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"Category02"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:                  ; @OBJC_METH_VAR_NAME_
	.asciz	"class02InstanceMethod"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:                  ; @OBJC_METH_VAR_TYPE_
	.asciz	"v16@0:8"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.1:                ; @OBJC_METH_VAR_NAME_.1
	.asciz	"myProtocol02Method"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category02"
__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category02:
	.long	24                              ; 0x18
	.long	2                               ; 0x2
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass(Category02) class02InstanceMethod]"
	.quad	l_OBJC_METH_VAR_NAME_.1
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass(Category02) myProtocol02Method]"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.2:                ; @OBJC_METH_VAR_NAME_.2
	.asciz	"class02ClassMethod"
l_OBJC_METH_VAR_NAME_.3:                ; @OBJC_METH_VAR_NAME_.3
	.asciz	"MyProtocol02Prop"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_.4:                ; @OBJC_METH_VAR_TYPE_.4
	.asciz	"i16@0:8"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_CLASS_METHODS_MyBaseClass_$_Category02"
__OBJC_$_CATEGORY_CLASS_METHODS_MyBaseClass_$_Category02:
	.long	24                              ; 0x18
	.long	2                               ; 0x2
	.quad	l_OBJC_METH_VAR_NAME_.2
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[MyBaseClass(Category02) class02ClassMethod]"
	.quad	l_OBJC_METH_VAR_NAME_.3
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.quad	"+[MyBaseClass(Category02) MyProtocol02Prop]"
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_.5:                   ; @OBJC_CLASS_NAME_.5
	.asciz	"MyProtocol02"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol02"
__OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol02:
	.long	24                              ; 0x18
	.long	2                               ; 0x2
	.quad	l_OBJC_METH_VAR_NAME_.1
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	0
	.quad	l_OBJC_METH_VAR_NAME_.3
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.quad	0
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_PROP_NAME_ATTR_:                 ; @OBJC_PROP_NAME_ATTR_
	.asciz	"MyProtocol02Prop"
l_OBJC_PROP_NAME_ATTR_.6:               ; @OBJC_PROP_NAME_ATTR_.6
	.asciz	"Ti,R"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_PROP_LIST_MyProtocol02"
__OBJC_$_PROP_LIST_MyProtocol02:
	.long	16                              ; 0x10
	.long	1                               ; 0x1
	.quad	l_OBJC_PROP_NAME_ATTR_
	.quad	l_OBJC_PROP_NAME_ATTR_.6
	.p2align	3, 0x0                          ; @"_OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol02"
__OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol02:
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.private_extern	__OBJC_PROTOCOL_$_MyProtocol02 ; @"_OBJC_PROTOCOL_$_MyProtocol02"
	.section	__DATA,__data
	.globl	__OBJC_PROTOCOL_$_MyProtocol02
	.weak_definition	__OBJC_PROTOCOL_$_MyProtocol02
	.p2align	3, 0x0
__OBJC_PROTOCOL_$_MyProtocol02:
	.quad	0
	.quad	l_OBJC_CLASS_NAME_.5
	.quad	0
	.quad	__OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol02
	.quad	0
	.quad	0
	.quad	0
	.quad	__OBJC_$_PROP_LIST_MyProtocol02
	.long	96                              ; 0x60
	.long	0                               ; 0x0
	.quad	__OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol02
	.quad	0
	.quad	0
	.private_extern	__OBJC_LABEL_PROTOCOL_$_MyProtocol02 ; @"_OBJC_LABEL_PROTOCOL_$_MyProtocol02"
	.section	__DATA,__objc_protolist,coalesced,no_dead_strip
	.globl	__OBJC_LABEL_PROTOCOL_$_MyProtocol02
	.weak_definition	__OBJC_LABEL_PROTOCOL_$_MyProtocol02
	.p2align	3, 0x0
__OBJC_LABEL_PROTOCOL_$_MyProtocol02:
	.quad	__OBJC_PROTOCOL_$_MyProtocol02
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_Category02"
__OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_Category02:
	.quad	1                               ; 0x1
	.quad	__OBJC_PROTOCOL_$_MyProtocol02
	.quad	0
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_PROP_NAME_ATTR_.7:               ; @OBJC_PROP_NAME_ATTR_.7
	.asciz	"Ti,R,D"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_PROP_LIST_MyBaseClass_$_Category02"
__OBJC_$_PROP_LIST_MyBaseClass_$_Category02:
	.long	16                              ; 0x10
	.long	1                               ; 0x1
	.quad	l_OBJC_PROP_NAME_ATTR_
	.quad	l_OBJC_PROP_NAME_ATTR_.7
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_MyBaseClass_$_Category02"
__OBJC_$_CATEGORY_MyBaseClass_$_Category02:
	.quad	l_OBJC_CLASS_NAME_
	.quad	_OBJC_CLASS_$_MyBaseClass
	.quad	__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category02
	.quad	__OBJC_$_CATEGORY_CLASS_METHODS_MyBaseClass_$_Category02
	.quad	__OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_Category02
	.quad	__OBJC_$_PROP_LIST_MyBaseClass_$_Category02
	.quad	0
	.long	64                              ; 0x40
	.space	4
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_.8:                   ; @OBJC_CLASS_NAME_.8
	.asciz	"Category03"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.9:                ; @OBJC_METH_VAR_NAME_.9
	.asciz	"class03InstanceMethod"
l_OBJC_METH_VAR_NAME_.10:               ; @OBJC_METH_VAR_NAME_.10
	.asciz	"myProtocol03Method"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category03"
__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category03:
	.long	24                              ; 0x18
	.long	2                               ; 0x2
	.quad	l_OBJC_METH_VAR_NAME_.9
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass(Category03) class03InstanceMethod]"
	.quad	l_OBJC_METH_VAR_NAME_.10
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyBaseClass(Category03) myProtocol03Method]"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.11:               ; @OBJC_METH_VAR_NAME_.11
	.asciz	"class03ClassMethod"
l_OBJC_METH_VAR_NAME_.12:               ; @OBJC_METH_VAR_NAME_.12
	.asciz	"MyProtocol03Prop"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_CLASS_METHODS_MyBaseClass_$_Category03"
__OBJC_$_CATEGORY_CLASS_METHODS_MyBaseClass_$_Category03:
	.long	24                              ; 0x18
	.long	2                               ; 0x2
	.quad	l_OBJC_METH_VAR_NAME_.11
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[MyBaseClass(Category03) class03ClassMethod]"
	.quad	l_OBJC_METH_VAR_NAME_.12
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.quad	"+[MyBaseClass(Category03) MyProtocol03Prop]"
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_.13:                  ; @OBJC_CLASS_NAME_.13
	.asciz	"MyProtocol03"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol03"
__OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol03:
	.long	24                              ; 0x18
	.long	2                               ; 0x2
	.quad	l_OBJC_METH_VAR_NAME_.10
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	0
	.quad	l_OBJC_METH_VAR_NAME_.12
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.quad	0
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_PROP_NAME_ATTR_.14:              ; @OBJC_PROP_NAME_ATTR_.14
	.asciz	"MyProtocol03Prop"
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_$_PROP_LIST_MyProtocol03"
__OBJC_$_PROP_LIST_MyProtocol03:
	.long	16                              ; 0x10
	.long	1                               ; 0x1
	.quad	l_OBJC_PROP_NAME_ATTR_.14
	.quad	l_OBJC_PROP_NAME_ATTR_.6
	.p2align	3, 0x0                          ; @"_OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol03"
__OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol03:
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.private_extern	__OBJC_PROTOCOL_$_MyProtocol03 ; @"_OBJC_PROTOCOL_$_MyProtocol03"
	.section	__DATA,__data
	.globl	__OBJC_PROTOCOL_$_MyProtocol03
	.weak_definition	__OBJC_PROTOCOL_$_MyProtocol03
	.p2align	3, 0x0
__OBJC_PROTOCOL_$_MyProtocol03:
	.quad	0
	.quad	l_OBJC_CLASS_NAME_.13
	.quad	0
	.quad	__OBJC_$_PROTOCOL_INSTANCE_METHODS_MyProtocol03
	.quad	0
	.quad	0
	.quad	0
	.quad	__OBJC_$_PROP_LIST_MyProtocol03
	.long	96                              ; 0x60
	.long	0                               ; 0x0
	.quad	__OBJC_$_PROTOCOL_METHOD_TYPES_MyProtocol03
	.quad	0
	.quad	0
	.private_extern	__OBJC_LABEL_PROTOCOL_$_MyProtocol03 ; @"_OBJC_LABEL_PROTOCOL_$_MyProtocol03"
	.section	__DATA,__objc_protolist,coalesced,no_dead_strip
	.globl	__OBJC_LABEL_PROTOCOL_$_MyProtocol03
	.weak_definition	__OBJC_LABEL_PROTOCOL_$_MyProtocol03
	.p2align	3, 0x0
__OBJC_LABEL_PROTOCOL_$_MyProtocol03:
	.quad	__OBJC_PROTOCOL_$_MyProtocol03
	.section	__DATA,__objc_const
	.p2align	3, 0x0                          ; @"_OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_Category03"
__OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_Category03:
	.quad	1                               ; 0x1
	.quad	__OBJC_PROTOCOL_$_MyProtocol03
	.quad	0
	.p2align	3, 0x0                          ; @"_OBJC_$_PROP_LIST_MyBaseClass_$_Category03"
__OBJC_$_PROP_LIST_MyBaseClass_$_Category03:
	.long	16                              ; 0x10
	.long	1                               ; 0x1
	.quad	l_OBJC_PROP_NAME_ATTR_.14
	.quad	l_OBJC_PROP_NAME_ATTR_.7
	.p2align	3, 0x0                          ; @"_OBJC_$_CATEGORY_MyBaseClass_$_Category03"
__OBJC_$_CATEGORY_MyBaseClass_$_Category03:
	.quad	l_OBJC_CLASS_NAME_.8
	.quad	_OBJC_CLASS_$_MyBaseClass
	.quad	__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category03
	.quad	__OBJC_$_CATEGORY_CLASS_METHODS_MyBaseClass_$_Category03
	.quad	__OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_Category03
	.quad	__OBJC_$_PROP_LIST_MyBaseClass_$_Category03
	.quad	0
	.long	64                              ; 0x40
	.space	4
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_.15:
	.asciz	"Category04"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.16:
	.asciz	"load"
	.section	__DATA,__objc_const
	.p2align	3, 0x0
__OBJC_$_CATEGORY_CLASS_METHODS_MyBaseClass_$_Category04:
	.long	24
	.long	1
	.quad	l_OBJC_METH_VAR_NAME_.16
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[MyBaseClass(Category04) load]"
	.p2align	3, 0x0
__OBJC_$_CATEGORY_MyBaseClass_$_Category04:
	.quad	l_OBJC_CLASS_NAME_.15
	.quad	_OBJC_CLASS_$_MyBaseClass
	.quad	0
	.quad	__OBJC_$_CATEGORY_CLASS_METHODS_MyBaseClass_$_Category04
	.quad	0
	.quad	0
	.quad	0
	.long	64
	.space	4
	.section	__DATA,__objc_catlist,regular,no_dead_strip
	.p2align	3, 0x0                          ; @"OBJC_LABEL_CATEGORY_$"
l_OBJC_LABEL_CATEGORY_$:
	.quad	__OBJC_$_CATEGORY_MyBaseClass_$_Category02
	.quad	__OBJC_$_CATEGORY_MyBaseClass_$_Category03
	.quad	__OBJC_$_CATEGORY_MyBaseClass_$_Category04
	.section	__DATA,__objc_nlcatlist,regular,no_dead_strip
	.p2align	3, 0x0
l_OBJC_LABEL_NONLAZY_CATEGORY_$:
	.quad	__OBJC_$_CATEGORY_MyBaseClass_$_Category04

	.no_dead_strip	__OBJC_LABEL_PROTOCOL_$_MyProtocol02
	.no_dead_strip	__OBJC_LABEL_PROTOCOL_$_MyProtocol03
	.no_dead_strip	__OBJC_PROTOCOL_$_MyProtocol02
	.no_dead_strip	__OBJC_PROTOCOL_$_MyProtocol03
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	96
.subsections_via_symbols
