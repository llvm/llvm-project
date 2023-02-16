# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11.0 -I %t %t/cat1.s -o %t/cat1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11.0 -I %t %t/cat2.s -o %t/cat2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11.0 -I %t %t/klass.s -o %t/klass.o
# RUN: %lld -dylib -lobjc %t/klass.o -o %t/libklass.dylib

# RUN: %no-fatal-warnings-lld -dylib -lobjc %t/klass.o %t/cat1.o %t/cat2.o -o \
# RUN:   /dev/null 2>&1 | FileCheck %s --check-prefixes=CATCLS,CATCAT
# RUN: %no-fatal-warnings-lld -dylib -lobjc %t/libklass.dylib %t/cat1.o \
# RUN:   %t/cat2.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=CATCAT

# CATCLS:      warning: method '+s1' has conflicting definitions:
# CATCLS-NEXT: >>> defined in category Cat1 from {{.*}}cat1.o
# CATCLS-NEXT: >>> defined in class Foo from {{.*}}klass.o

# CATCLS:      warning: method '-m1' has conflicting definitions:
# CATCLS-NEXT: >>> defined in category Cat1 from {{.*}}cat1.o
# CATCLS-NEXT: >>> defined in class Foo from {{.*}}klass.o

# CATCAT:      warning: method '+s2' has conflicting definitions:
# CATCAT-NEXT: >>> defined in category Cat2 from {{.*}}cat2.o
# CATCAT-NEXT: >>> defined in category Cat1 from {{.*}}cat1.o

# CATCAT:      warning: method '-m2' has conflicting definitions:
# CATCAT-NEXT: >>> defined in category Cat2 from {{.*}}cat2.o
# CATCAT-NEXT: >>> defined in category Cat1 from {{.*}}cat1.o

#--- cat1.s

.include "objc-macros.s"

## @interface Foo(Cat1)
## -(void) m1;
## -(void) m2;
## +(void) s1;
## +(void) s2;
## @end
## 
## @implementation Foo(Cat1)
## -(void) m1 {}
## -(void) m2 {}
## +(void) s1 {}
## +(void) s2 {}
## @end

.section __DATA,__objc_catlist,regular,no_dead_strip
  .quad __OBJC_$_CATEGORY_Foo_$_Cat1

.section __DATA,__objc_const
__OBJC_$_CATEGORY_Foo_$_Cat1:
  .objc_classname "Cat1"
  .quad _OBJC_CLASS_$_Foo
  .quad __OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Cat1
  .quad __OBJC_$_CATEGORY_CLASS_METHODS_Foo_$_Cat1
  .quad 0
  .quad 0
  .quad 0
  .long 64
  .space 4

__OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Cat1:
  .long 24 # size of method entry
  .long 2 # number of methods
  .empty_objc_method "m1", "v16@0:8", "-[Foo(Cat1) m1]"
  .empty_objc_method "m2", "v16@0:8", "-[Foo(Cat2) m2]"

__OBJC_$_CATEGORY_CLASS_METHODS_Foo_$_Cat1:
  .long 24
  .long 2
  .empty_objc_method "s1", "v16@0:8", "+[Foo(Cat1) s1]"
  .empty_objc_method "s2", "v16@0:8", "+[Foo(Cat1) s2]"

.section __DATA,__objc_imageinfo,regular,no_dead_strip
  .long 0
  .long 64

.subsections_via_symbols

#--- cat2.s

.include "objc-macros.s"

## @interface Foo(Cat2)
## -(void) m2;
## +(void) s2;
## @end
## 
## @implementation Foo(Cat2)
## -(void) m2 {}
## +(void) s2 {}
## @end

.section __DATA,__objc_catlist,regular,no_dead_strip
  .quad __OBJC_$_CATEGORY_Foo_$_Cat2

.section __DATA,__objc_const
__OBJC_$_CATEGORY_Foo_$_Cat2:
  .objc_classname "Cat2"
  .quad _OBJC_CLASS_$_Foo
  .quad __OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Cat2
  .quad __OBJC_$_CATEGORY_CLASS_METHODS_Foo_$_Cat2
  .quad 0
  .quad 0
  .quad 0
  .long 64
  .space 4

__OBJC_$_CATEGORY_INSTANCE_METHODS_Foo_$_Cat2:
  .long 24
  .long 1
  .empty_objc_method "m2", "v16@0:8", "-[Foo(Cat2) m2]"

__OBJC_$_CATEGORY_CLASS_METHODS_Foo_$_Cat2:
  .long 24
  .long 1
  .empty_objc_method "s2", "v16@0:8", "+[Foo(Cat2) m2]"

.section __DATA,__objc_imageinfo,regular,no_dead_strip
  .long 0
  .long 64

.subsections_via_symbols

#--- klass.s

.include "objc-macros.s"

## @interface Foo
## -(void) m1;
## +(void) s1;
## @end
##
## @implementation Foo
## -(void) m1 {}
## +(void) s1 {}
## @end

.globl _OBJC_CLASS_$_Foo, _OBJC_METACLASS_$_Foo

.section __DATA,__objc_data
_OBJC_CLASS_$_Foo:
  .quad _OBJC_METACLASS_$_Foo
  .quad 0
  .quad __objc_empty_cache
  .quad 0
  .quad __OBJC_CLASS_RO_$_Foo

_OBJC_METACLASS_$_Foo:
  .quad _OBJC_METACLASS_$_Foo
  .quad _OBJC_CLASS_$_Foo
  .quad __objc_empty_cache
  .quad 0
  .quad __OBJC_METACLASS_RO_$_Foo

.section __DATA,__objc_const
__OBJC_METACLASS_RO_$_Foo:
  .long 3
  .long 40
  .long 40
  .space 4
  .quad 0
  .objc_classname "Foo"
  .quad __OBJC_$_CLASS_METHODS_Foo
  .quad 0
  .quad 0
  .quad 0
  .quad 0

__OBJC_CLASS_RO_$_Foo:
  .long 2
  .long 0
  .long 0
  .space 4
  .quad 0
  .objc_classname "Foo"
  .quad __OBJC_$_INSTANCE_METHODS_Foo
  .quad 0
  .quad 0
  .quad 0
  .quad 0

__OBJC_$_CLASS_METHODS_Foo:
  .long 24
  .long 1
  .empty_objc_method "s1", "v16@0:8", "+[Foo s1]"

__OBJC_$_INSTANCE_METHODS_Foo:
  .long 24
  .long 1
  .empty_objc_method "m1", "v16@0:8", "-[Foo m1]"

.section __DATA,__objc_classlist,regular,no_dead_strip
  .quad _OBJC_CLASS_$_Foo

.section __DATA,__objc_imageinfo,regular,no_dead_strip
  .long 0
  .long 64

.subsections_via_symbols

#--- objc-macros.s

# Macros for taking some of the boilerplate out of defining objc structs.

# NOTE: \@ below is a variable that gets auto-incremented by the assembler on
# each macro invocation. It serves as a mechanism for generating unique symbol
# names for each macro call.

.macro .objc_classname name

  .section __TEXT,__objc_classname,cstring_literals
  L_OBJC_CLASS_NAME_.\@:
    .asciz "\name"

  .section __DATA,__objc_const
  .quad L_OBJC_CLASS_NAME_.\@

.endm

# struct method_t {
#   const char *name;
#   const char *type;
#   void *impl;
# }
.macro .objc_method name, type, impl

  .section __TEXT,__objc_methname,cstring_literals
  L_OBJC_METH_VAR_NAME_.\@:
    .asciz "\name"

  .section __TEXT,__objc_methtype,cstring_literals
  L_OBJC_METH_VAR_TYPE_.\@:
    .asciz "\type"

  .section __DATA,__objc_const
    .quad L_OBJC_METH_VAR_NAME_.\@
    .quad L_OBJC_METH_VAR_TYPE_.\@
    .quad "\impl"

.endm

# Generate a method_t with a basic impl that just contains `ret`.
.macro .empty_objc_method name, type, impl

  .text
  "\impl":
    ret

  .objc_method "\name", "\type", "\impl"

.endm
