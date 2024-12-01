; REQUIRES: aarch64

; Here we test that if we defined a protocol MyTestProtocol and also a category MyTestProtocol
; then when merging the category into the base class (and deleting the category), we don't
; delete the 'MyTestProtocol' name

; RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %T/erase-objc-name.o %s
; RUN: %lld -no_objc_relative_method_lists -arch arm64 -dylib -o %T/erase-objc-name.dylib %T/erase-objc-name.o -objc_category_merging
; RUN: llvm-objdump --objc-meta-data --macho %T/erase-objc-name.dylib | FileCheck %s --check-prefixes=MERGE_CATS

; === Check merge categories enabled ===
; Check that the original categories are not there
; MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_Category01
; MERGE_CATS-NOT: __OBJC_$_CATEGORY_MyBaseClass_$_Category02

; Check that we get the expected output - most importantly that the protocol is named `MyTestProtocol`
; MERGE_CATS:        Contents of (__DATA_CONST,__objc_classlist) section
; MERGE_CATS-NEXT:   _OBJC_CLASS_$_MyBaseClass
; MERGE_CATS-NEXT:            isa {{.*}} _OBJC_METACLASS_$_MyBaseClass
; MERGE_CATS-NEXT:     superclass {{.*}}
; MERGE_CATS-NEXT:          cache {{.*}}
; MERGE_CATS-NEXT:         vtable {{.*}}
; MERGE_CATS-NEXT:           data {{.*}} (struct class_ro_t *)
; MERGE_CATS-NEXT:                     flags {{.*}} RO_ROOT
; MERGE_CATS-NEXT:             instanceStart 0
; MERGE_CATS-NEXT:              instanceSize 0
; MERGE_CATS-NEXT:                  reserved {{.*}}
; MERGE_CATS-NEXT:                ivarLayout {{.*}}
; MERGE_CATS-NEXT:                      name {{.*}} MyBaseClass
; MERGE_CATS-NEXT:               baseMethods {{.*}} (struct method_list_t *)
; MERGE_CATS-NEXT:             entsize 24
; MERGE_CATS-NEXT:               count 2
; MERGE_CATS-NEXT:                name {{.*}} getValue
; MERGE_CATS-NEXT:               types {{.*}} i16@0:8
; MERGE_CATS-NEXT:                 imp -[MyBaseClass(MyTestProtocol) getValue]
; MERGE_CATS-NEXT:                name {{.*}} baseInstanceMethod
; MERGE_CATS-NEXT:               types {{.*}} v16@0:8
; MERGE_CATS-NEXT:                 imp -[MyBaseClass baseInstanceMethod]
; MERGE_CATS-NEXT:             baseProtocols {{.*}}
; MERGE_CATS-NEXT:                       count 1
; MERGE_CATS-NEXT:                list[0] {{.*}} (struct protocol_t *)
; MERGE_CATS-NEXT:                       isa {{.*}}
; MERGE_CATS-NEXT:                      name {{.*}} MyTestProtocol
; MERGE_CATS-NEXT:                 protocols {{.*}}
; MERGE_CATS-NEXT:            instanceMethods {{.*}} (struct method_list_t *)
; MERGE_CATS-NEXT:                    entsize 24
; MERGE_CATS-NEXT:                      count 1
; MERGE_CATS-NEXT:                       name {{.*}} getValue
; MERGE_CATS-NEXT:                      types {{.*}} i16@0:8
; MERGE_CATS-NEXT:                        imp {{.*}}
; MERGE_CATS-NEXT:               classMethods {{.*}} (struct method_list_t *)
; MERGE_CATS-NEXT:     optionalInstanceMethods {{.*}}
; MERGE_CATS-NEXT:        optionalClassMethods {{.*}}
; MERGE_CATS-NEXT:          instanceProperties {{.*}}
; MERGE_CATS-NEXT:                     ivars {{.*}}
; MERGE_CATS-NEXT:            weakIvarLayout {{.*}}
; MERGE_CATS-NEXT:            baseProperties {{.*}}
; MERGE_CATS-NEXT: Meta Class
; MERGE_CATS-NEXT:            isa {{.*}} _OBJC_METACLASS_$_MyBaseClass
; MERGE_CATS-NEXT:     superclass {{.*}} _OBJC_CLASS_$_MyBaseClass
; MERGE_CATS-NEXT:          cache {{.*}}
; MERGE_CATS-NEXT:         vtable {{.*}}
; MERGE_CATS-NEXT:           data {{.*}} (struct class_ro_t *)
; MERGE_CATS-NEXT:                     flags {{.*}} RO_META RO_ROOT
; MERGE_CATS-NEXT:             instanceStart 40
; MERGE_CATS-NEXT:              instanceSize 40
; MERGE_CATS-NEXT:                  reserved {{.*}}
; MERGE_CATS-NEXT:                ivarLayout {{.*}}
; MERGE_CATS-NEXT:                      name {{.*}} MyBaseClass
; MERGE_CATS-NEXT:               baseMethods {{.*}} (struct method_list_t *)
; MERGE_CATS-NEXT:             baseProtocols {{.*}}
; MERGE_CATS-NEXT:                       count 1
; MERGE_CATS-NEXT:                list[0] {{.*}} (struct protocol_t *)
; MERGE_CATS-NEXT:                       isa {{.*}}
; MERGE_CATS-NEXT:                      name {{.*}} MyTestProtocol
; MERGE_CATS-NEXT:                 protocols {{.*}}
; MERGE_CATS-NEXT:            instanceMethods {{.*}} (struct method_list_t *)
; MERGE_CATS-NEXT:                    entsize 24
; MERGE_CATS-NEXT:                      count 1
; MERGE_CATS-NEXT:                       name {{.*}} getValue
; MERGE_CATS-NEXT:                      types {{.*}} i16@0:8
; MERGE_CATS-NEXT:                        imp {{.*}}
; MERGE_CATS-NEXT:               classMethods {{.*}} (struct method_list_t *)
; MERGE_CATS-NEXT:     optionalInstanceMethods {{.*}}
; MERGE_CATS-NEXT:        optionalClassMethods {{.*}}
; MERGE_CATS-NEXT:          instanceProperties {{.*}}
; MERGE_CATS-NEXT:                     ivars {{.*}}
; MERGE_CATS-NEXT:            weakIvarLayout {{.*}}
; MERGE_CATS-NEXT:            baseProperties {{.*}}
; MERGE_CATS-NEXT: Contents of (__DATA_CONST,__objc_protolist) section
; MERGE_CATS-NEXT: {{.*}} {{.*}} __OBJC_PROTOCOL_$_MyTestProtocol
; MERGE_CATS-NEXT: Contents of (__DATA_CONST,__objc_imageinfo) section
; MERGE_CATS-NEXT:   version 0
; MERGE_CATS-NEXT:     flags {{.*}} OBJC_IMAGE_HAS_CATEGORY_CLASS_PROPERTIES


; ================== repro.sh ====================
; # Write the Objective-C code to a file
; cat << EOF > MyClass.m
; @protocol MyTestProtocol
; - (int)getValue;
; @end
;
; __attribute__((objc_root_class))
; @interface MyBaseClass
; - (void)baseInstanceMethod;
; @end
;
; @implementation MyBaseClass
; - (void)baseInstanceMethod {}
; @end
;
; @interface MyBaseClass (MyTestProtocol) <MyTestProtocol>
; @end
;
; @implementation MyBaseClass (MyTestProtocol)
;
; - (int)getValue {
;     return 0x30;
; }
;
; @end
; EOF
;
; # Compile the Objective-C file to assembly
; xcrun clang -S -arch arm64 MyClass.m -o MyClass.s
; ==============================================


       .section      __TEXT,__text,regular,pure_instructions
       .p2align      2                               ; -- Begin function -[MyBaseClass baseInstanceMethod]
"-[MyBaseClass baseInstanceMethod]":    ; @"\01-[MyBaseClass baseInstanceMethod]"
       .cfi_startproc
; %bb.0:
       sub    sp, sp, #16
       .cfi_def_cfa_offset 16
       str    x0, [sp, #8]
       str    x1, [sp]
       add    sp, sp, #16
       ret
       .cfi_endproc
                                        ; -- End function
       .p2align      2                               ; -- Begin function -[MyBaseClass(MyTestProtocol) getValue]
"-[MyBaseClass(MyTestProtocol) getValue]": ; @"\01-[MyBaseClass(MyTestProtocol) getValue]"
       .cfi_startproc
; %bb.0:
       sub    sp, sp, #16
       .cfi_def_cfa_offset 16
       str    x0, [sp, #8]
       str    x1, [sp]
       mov    w0, #48                         ; =0x30
       add    sp, sp, #16
       ret
       .cfi_endproc
                                        ; -- End function
       .section      __DATA,__objc_data
       .globl _OBJC_CLASS_$_MyBaseClass       ; @"OBJC_CLASS_$_MyBaseClass"
       .p2align      3, 0x0
_OBJC_CLASS_$_MyBaseClass:
       .quad  _OBJC_METACLASS_$_MyBaseClass
       .quad  0
       .quad  __objc_empty_cache
       .quad  0
       .quad  __OBJC_CLASS_RO_$_MyBaseClass
       .globl _OBJC_METACLASS_$_MyBaseClass   ; @"OBJC_METACLASS_$_MyBaseClass"
       .p2align      3, 0x0
_OBJC_METACLASS_$_MyBaseClass:
       .quad  _OBJC_METACLASS_$_MyBaseClass
       .quad  _OBJC_CLASS_$_MyBaseClass
       .quad  __objc_empty_cache
       .quad  0
       .quad  __OBJC_METACLASS_RO_$_MyBaseClass
       .section      __TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
       .asciz "MyBaseClass"
       .section      __DATA,__objc_const
       .p2align      3, 0x0                          ; @"_OBJC_METACLASS_RO_$_MyBaseClass"
__OBJC_METACLASS_RO_$_MyBaseClass:
       .long  131                             ; 0x83
       .long  40                              ; 0x28
       .long  40                              ; 0x28
       .space 4
       .quad  0
       .quad  l_OBJC_CLASS_NAME_
       .quad  0
       .quad  0
       .quad  0
       .quad  0
       .quad  0
       .section      __TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:                  ; @OBJC_METH_VAR_NAME_
       .asciz "baseInstanceMethod"
       .section      __TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:                  ; @OBJC_METH_VAR_TYPE_
       .asciz "v16@0:8"
       .section      __DATA,__objc_const
       .p2align      3, 0x0                          ; @"_OBJC_$_INSTANCE_METHODS_MyBaseClass"
__OBJC_$_INSTANCE_METHODS_MyBaseClass:
       .long  24                              ; 0x18
       .long  1                               ; 0x1
       .quad  l_OBJC_METH_VAR_NAME_
       .quad  l_OBJC_METH_VAR_TYPE_
       .quad  "-[MyBaseClass baseInstanceMethod]"
       .p2align      3, 0x0                          ; @"_OBJC_CLASS_RO_$_MyBaseClass"
__OBJC_CLASS_RO_$_MyBaseClass:
       .long  130                             ; 0x82
       .long  0                               ; 0x0
       .long  0                               ; 0x0
       .space 4
       .quad  0
       .quad  l_OBJC_CLASS_NAME_
       .quad  __OBJC_$_INSTANCE_METHODS_MyBaseClass
       .quad  0
       .quad  0
       .quad  0
       .quad  0
       .section      __TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_.1:                   ; @OBJC_CLASS_NAME_.1
       .asciz "MyTestProtocol"
       .section      __TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.2:                ; @OBJC_METH_VAR_NAME_.2
       .asciz "getValue"
       .section      __TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_.3:                ; @OBJC_METH_VAR_TYPE_.3
       .asciz "i16@0:8"
       .section      __DATA,__objc_const
       .p2align      3, 0x0                          ; @"_OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_MyTestProtocol"
__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_MyTestProtocol:
       .long  24                              ; 0x18
       .long  1                               ; 0x1
       .quad  l_OBJC_METH_VAR_NAME_.2
       .quad  l_OBJC_METH_VAR_TYPE_.3
       .quad  "-[MyBaseClass(MyTestProtocol) getValue]"
       .p2align      3, 0x0                          ; @"_OBJC_$_PROTOCOL_INSTANCE_METHODS_MyTestProtocol"
__OBJC_$_PROTOCOL_INSTANCE_METHODS_MyTestProtocol:
       .long  24                              ; 0x18
       .long  1                               ; 0x1
       .quad  l_OBJC_METH_VAR_NAME_.2
       .quad  l_OBJC_METH_VAR_TYPE_.3
       .quad  0
       .p2align      3, 0x0                          ; @"_OBJC_$_PROTOCOL_METHOD_TYPES_MyTestProtocol"
__OBJC_$_PROTOCOL_METHOD_TYPES_MyTestProtocol:
       .quad  l_OBJC_METH_VAR_TYPE_.3
       .private_extern      __OBJC_PROTOCOL_$_MyTestProtocol ; @"_OBJC_PROTOCOL_$_MyTestProtocol"
       .section      __DATA,__data
       .globl __OBJC_PROTOCOL_$_MyTestProtocol
       .weak_definition     __OBJC_PROTOCOL_$_MyTestProtocol
       .p2align      3, 0x0
__OBJC_PROTOCOL_$_MyTestProtocol:
       .quad  0
       .quad  l_OBJC_CLASS_NAME_.1
       .quad  0
       .quad  __OBJC_$_PROTOCOL_INSTANCE_METHODS_MyTestProtocol
       .quad  0
       .quad  0
       .quad  0
       .quad  0
       .long  96                              ; 0x60
       .long  0                               ; 0x0
       .quad  __OBJC_$_PROTOCOL_METHOD_TYPES_MyTestProtocol
       .quad  0
       .quad  0
       .private_extern      __OBJC_LABEL_PROTOCOL_$_MyTestProtocol ; @"_OBJC_LABEL_PROTOCOL_$_MyTestProtocol"
       .section      __DATA,__objc_protolist,coalesced,no_dead_strip
       .globl __OBJC_LABEL_PROTOCOL_$_MyTestProtocol
       .weak_definition     __OBJC_LABEL_PROTOCOL_$_MyTestProtocol
       .p2align      3, 0x0
__OBJC_LABEL_PROTOCOL_$_MyTestProtocol:
       .quad  __OBJC_PROTOCOL_$_MyTestProtocol
       .section      __DATA,__objc_const
       .p2align      3, 0x0                          ; @"_OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_MyTestProtocol"
__OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_MyTestProtocol:
       .quad  1                               ; 0x1
       .quad  __OBJC_PROTOCOL_$_MyTestProtocol
       .quad  0
       .p2align      3, 0x0                          ; @"_OBJC_$_CATEGORY_MyBaseClass_$_MyTestProtocol"
__OBJC_$_CATEGORY_MyBaseClass_$_MyTestProtocol:
       .quad  l_OBJC_CLASS_NAME_.1
       .quad  _OBJC_CLASS_$_MyBaseClass
       .quad  __OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_MyTestProtocol
       .quad  0
       .quad  __OBJC_CATEGORY_PROTOCOLS_$_MyBaseClass_$_MyTestProtocol
       .quad  0
       .quad  0
       .long  64                              ; 0x40
       .space 4
       .section      __DATA,__objc_classlist,regular,no_dead_strip
       .p2align      3, 0x0                          ; @"OBJC_LABEL_CLASS_$"
l_OBJC_LABEL_CLASS_$:
       .quad  _OBJC_CLASS_$_MyBaseClass
       .section      __DATA,__objc_catlist,regular,no_dead_strip
       .p2align      3, 0x0                          ; @"OBJC_LABEL_CATEGORY_$"
l_OBJC_LABEL_CATEGORY_$:
       .quad  __OBJC_$_CATEGORY_MyBaseClass_$_MyTestProtocol
       .no_dead_strip       __OBJC_PROTOCOL_$_MyTestProtocol
       .no_dead_strip       __OBJC_LABEL_PROTOCOL_$_MyTestProtocol
       .section      __DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
       .long  0
       .long  64

__objc_empty_cache:
_$sBOWV:
  .quad 0

.subsections_via_symbols
