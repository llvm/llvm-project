# REQUIRES: aarch64
## Category merging does not preserve the signing schemas of the pointers in
## arm64e category data, so an arm64e link warns about -objc_category_merging
## and keeps the categories as they are.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/base.o %t/base.s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib \
# RUN:   -install_name @executable_path/libbase.dylib %t/base.o -o %t/libbase.dylib
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/cats.o %t/cats.s
# RUN: %no-fatal-warnings-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -dylib -objc_category_merging %t/libbase.dylib %t/cats.o \
# RUN:   -o %t/cats.dylib 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-objdump --macho --section-headers --syms %t/cats.dylib > %t/out
# RUN: llvm-objdump --macho --dyld-info %t/cats.dylib >> %t/out
# RUN: FileCheck %s < %t/out

# WARN: warning: -objc_category_merging is not supported on arm64e, ignoring

## Both categories are still listed, and each keeps its signed method list
## pointer (key DA, diversity 0xC310, address diversity).
# CHECK:      __objc_catlist 00000010
# CHECK:      [[#%x,CAT1:]] l O __DATA,__objc_const __OBJC_$_CATEGORY_MyBaseClass_$_Category01
# CHECK:      [[#%x,CAT2:]] l O __DATA,__objc_const __OBJC_$_CATEGORY_MyBaseClass_$_Category02
# CHECK-LABEL: dyld information:
# CHECK:      0x[[#%X,CAT1+16]] 0x{{[0-9A-F]+}} rebase 0xC310 true DA
# CHECK:      0x[[#%X,CAT2+16]] 0x{{[0-9A-F]+}} rebase 0xC310 true DA

#--- base.s
.section __DATA,__objc_data
.globl _OBJC_CLASS_$_MyBaseClass
_OBJC_CLASS_$_MyBaseClass:
  .quad 0

#--- cats.s
.p2align 2
"-[MyBaseClass(Category01) cat01_InstanceMethod]":
  ret
"-[MyBaseClass(Category02) cat02_InstanceMethod]":
  ret

.section __TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:
  .asciz "Category01"
l_OBJC_CLASS_NAME_.1:
  .asciz "Category02"

.section __TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:
  .asciz "cat01_InstanceMethod"
l_OBJC_METH_VAR_NAME_.2:
  .asciz "cat02_InstanceMethod"

.section __TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:
  .asciz "v16@0:8"

.section __DATA,__objc_const
.p2align 3
__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category01:
  .long 24
  .long 1
  .quad l_OBJC_METH_VAR_NAME_
  .quad l_OBJC_METH_VAR_TYPE_
  .quad "-[MyBaseClass(Category01) cat01_InstanceMethod]"@AUTH(ia,0,addr)
__OBJC_$_CATEGORY_MyBaseClass_$_Category01:
  .quad l_OBJC_CLASS_NAME_
  .quad _OBJC_CLASS_$_MyBaseClass
  .quad __OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category01@AUTH(da,0xc310,addr)
  .quad 0
  .quad 0
  .quad 0
  .quad 0
  .long 64
  .space 4
__OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category02:
  .long 24
  .long 1
  .quad l_OBJC_METH_VAR_NAME_.2
  .quad l_OBJC_METH_VAR_TYPE_
  .quad "-[MyBaseClass(Category02) cat02_InstanceMethod]"@AUTH(ia,0,addr)
__OBJC_$_CATEGORY_MyBaseClass_$_Category02:
  .quad l_OBJC_CLASS_NAME_.1
  .quad _OBJC_CLASS_$_MyBaseClass
  .quad __OBJC_$_CATEGORY_INSTANCE_METHODS_MyBaseClass_$_Category02@AUTH(da,0xc310,addr)
  .quad 0
  .quad 0
  .quad 0
  .quad 0
  .long 64
  .space 4

.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
l_OBJC_LABEL_CATEGORY_$:
  .quad __OBJC_$_CATEGORY_MyBaseClass_$_Category01
  .quad __OBJC_$_CATEGORY_MyBaseClass_$_Category02

.section __DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
  .long 0
  .long 96

.subsections_via_symbols
