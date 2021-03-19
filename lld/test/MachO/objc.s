# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/has-objc-symbol.s -o %t/has-objc-symbol.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/has-objc-category.s -o %t/has-objc-category.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/has-swift.s -o %t/has-swift.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/no-objc.s -o %t/no-objc.o
# RUN: llvm-ar rcs %t/libHasSomeObjC.a %t/has-objc-symbol.o %t/has-objc-category.o %t/has-swift.o %t/no-objc.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -lSystem %t/test.o -o %t/test \
# RUN:   -L%t -lHasSomeObjC -ObjC
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s --check-prefix=OBJC

# OBJC:       Sections:
# OBJC-NEXT:  Idx Name           Size   VMA  Type
# OBJC-NEXT:    0 __text         {{.*}}      TEXT
# OBJC-NEXT:    1 __swift        {{.*}}      DATA
# OBJC-NEXT:    2 __objc_catlist {{.*}}      DATA
# OBJC-EMPTY:
# OBJC-NEXT:  SYMBOL TABLE:
# OBJC-NEXT:  g     F __TEXT,__text _main
# OBJC-NEXT:  g     F __TEXT,__text _OBJC_CLASS_$_MyObject

# RUN: %lld -lSystem %t/test.o -o %t/test \
# RUN:   -L%t -lHasSomeObjC
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s --check-prefix=NO-OBJC

# NO-OBJC:       Sections:
# NO-OBJC-NEXT:  Idx Name           Size   VMA  Type
# NO-OBJC-NEXT:    0 __text         {{.*}}      TEXT
# NO-OBJC-EMPTY:
# NO-OBJC-NEXT:  SYMBOL TABLE:
# NO-OBJC-NEXT:  g     F __TEXT,__text _main
# NO_OBJC-NEXT:  g *ABS* __mh_execute_header

#--- has-objc-symbol.s
.globl _OBJC_CLASS_$_MyObject
_OBJC_CLASS_$_MyObject:

#--- has-objc-category.s
.section	__DATA,__objc_catlist
.quad 0x1234

#--- has-swift.s
.section	__TEXT,__swift
.quad 0x1234

#--- no-objc.s
## This archive member should not be pulled in since it does not contain any
## ObjC-related data.
.globl _foo
.section __DATA,foo

foo:
  .quad 0x1234

#--- test.s
.globl _main
_main:
  ret
