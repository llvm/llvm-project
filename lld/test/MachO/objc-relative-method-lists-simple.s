# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t && cd %t

## Compile a64_rel_dylib.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o a64_rel_dylib.o a64_simple_class.s

## Test arm64 + relative method lists
# RUN: ld64.lld a64_rel_dylib.o -o a64_rel_dylib.dylib -map a64_rel_dylib.map -dylib -arch arm64 -platform_version macos 11.0.0 11.0.0 -objc_relative_method_lists
# RUN: llvm-objdump --macho --objc-meta-data a64_rel_dylib.dylib  | FileCheck %s --check-prefix=CHK_REL

## Test arm64 + traditional method lists (no relative offsets)
# RUN: ld64.lld a64_rel_dylib.o -o a64_rel_dylib.dylib -map a64_rel_dylib.map -dylib -arch arm64 -platform_version macos 11.0.0 11.0.0 -no_objc_relative_method_lists
# RUN: llvm-objdump --macho --objc-meta-data a64_rel_dylib.dylib  | FileCheck %s --check-prefix=CHK_NO_REL

CHK_NO_REL-NOT: (relative)

CHK_REL:       Contents of (__DATA_CONST,__objc_classlist) section
CHK_REL-NEXT:  _OBJC_CLASS_$_MyClass
CHK_REL:       baseMethods
CHK_REL-NEXT:  entsize 24 (relative)
CHK_REL-NEXT:  count 3
CHK_REL-NEXT:   name 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) instance_method_00
CHK_REL-NEXT:  types 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) v16@0:8
CHK_REL-NEXT:    imp 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) -[MyClass instance_method_00]
CHK_REL-NEXT:   name 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) instance_method_01
CHK_REL-NEXT:  types 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) v16@0:8
CHK_REL-NEXT:    imp 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) -[MyClass instance_method_01]
CHK_REL-NEXT:   name 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) instance_method_02
CHK_REL-NEXT:  types 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) v16@0:8
CHK_REL-NEXT:    imp 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}}) -[MyClass instance_method_02]

CHK_REL:       Meta Class
CHK_REL-NEXT:  isa 0x{{[0-9a-f]*}} _OBJC_METACLASS_$_MyClass
CHK_REL:       baseMethods 0x694 (struct method_list_t *)
CHK_REL-NEXT:  entsize 24 (relative)
CHK_REL-NEXT:  count 3
CHK_REL-NEXT:   name 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  class_method_00
CHK_REL-NEXT:  types 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  v16@0:8
CHK_REL-NEXT:    imp 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  +[MyClass class_method_00]
CHK_REL-NEXT:   name 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  class_method_01
CHK_REL-NEXT:  types 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  v16@0:8
CHK_REL-NEXT:    imp 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  +[MyClass class_method_01]
CHK_REL-NEXT:   name 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  class_method_02
CHK_REL-NEXT:  types 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  v16@0:8
CHK_REL-NEXT:    imp 0x{{[0-9a-f]*}} (0x{{[0-9a-f]*}})  +[MyClass class_method_02]



######################## Generate a64_simple_class.s #########################
# clang -c simple_class.mm -s -o a64_simple_class.s -target arm64-apple-macos -arch arm64 -Oz

########################       simple_class.mm       ########################
#  __attribute__((objc_root_class))
#  @interface MyClass
#  - (void)instance_method_00;
#  - (void)instance_method_01;
#  - (void)instance_method_02;
#  + (void)class_method_00;
#  + (void)class_method_01;
#  + (void)class_method_02;
#  @end
#
#  @implementation MyClass
#  - (void)instance_method_00 {}
#  - (void)instance_method_01 {}
#  - (void)instance_method_02 {}
#  + (void)class_method_00 {}
#  + (void)class_method_01 {}
#  + (void)class_method_02 {}
#  @end
#
#  void *_objc_empty_cache;
#  void *_objc_empty_vtable;
#

#--- objc-macros.s
.macro .objc_selector_def name
	.p2align	2
"\name":
	.cfi_startproc
	ret
	.cfi_endproc
.endm

#--- a64_simple_class.s
.include "objc-macros.s"

.section	__TEXT,__text,regular,pure_instructions
.build_version macos, 11, 0

.objc_selector_def "-[MyClass instance_method_00]"
.objc_selector_def "-[MyClass instance_method_01]"
.objc_selector_def "-[MyClass instance_method_02]"

.objc_selector_def "+[MyClass class_method_00]"
.objc_selector_def "+[MyClass class_method_01]"
.objc_selector_def "+[MyClass class_method_02]"

.globl	__objc_empty_vtable
.zerofill __DATA,__common,__objc_empty_vtable,8,3
.section	__DATA,__objc_data
.globl	_OBJC_CLASS_$_MyClass
.p2align	3, 0x0

_OBJC_CLASS_$_MyClass:
	.quad	_OBJC_METACLASS_$_MyClass
	.quad	0
	.quad	__objc_empty_cache
	.quad	__objc_empty_vtable
	.quad	__OBJC_CLASS_RO_$_MyClass
	.globl	_OBJC_METACLASS_$_MyClass
	.p2align	3, 0x0

_OBJC_METACLASS_$_MyClass:
	.quad	_OBJC_METACLASS_$_MyClass
	.quad	_OBJC_CLASS_$_MyClass
	.quad	__objc_empty_cache
	.quad	__objc_empty_vtable
	.quad	__OBJC_METACLASS_RO_$_MyClass

	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:
	.asciz	"MyClass"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:
	.asciz	"class_method_00"
	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:
	.asciz	"v16@0:8"
	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.1:
	.asciz	"class_method_01"
l_OBJC_METH_VAR_NAME_.2:
	.asciz	"class_method_02"
	.section	__DATA,__objc_const
	.p2align	3, 0x0
__OBJC_$_CLASS_METHODS_MyClass:
	.long	24
	.long	3
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[MyClass class_method_00]"
	.quad	l_OBJC_METH_VAR_NAME_.1
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[MyClass class_method_01]"
	.quad	l_OBJC_METH_VAR_NAME_.2
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"+[MyClass class_method_02]"
	.p2align	3, 0x0

__OBJC_METACLASS_RO_$_MyClass:
	.long	3
	.long	40
	.long	40
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_CLASS_METHODS_MyClass
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_.3:
	.asciz	"instance_method_00"
l_OBJC_METH_VAR_NAME_.4:
	.asciz	"instance_method_01"
l_OBJC_METH_VAR_NAME_.5:
	.asciz	"instance_method_02"

	.section	__DATA,__objc_const
	.p2align	3, 0x0
__OBJC_$_INSTANCE_METHODS_MyClass:
	.long	24
	.long	3
	.quad	l_OBJC_METH_VAR_NAME_.3
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyClass instance_method_00]"
	.quad	l_OBJC_METH_VAR_NAME_.4
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyClass instance_method_01]"
	.quad	l_OBJC_METH_VAR_NAME_.5
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[MyClass instance_method_02]"
	.p2align	3, 0x0

__OBJC_CLASS_RO_$_MyClass:
	.long	2
	.long	0
	.long	0
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_INSTANCE_METHODS_MyClass
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.globl	__objc_empty_cache

.zerofill __DATA,__common,__objc_empty_cache,8,3
	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3, 0x0
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_MyClass
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64
.subsections_via_symbols
