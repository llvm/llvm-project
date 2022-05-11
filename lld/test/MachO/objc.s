# REQUIRES: x86, arm
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/has-objc-symbol.s -o %t/has-objc-symbol.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/has-objc-category.s -o %t/has-objc-category.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/has-objc-symbol-and-category.s -o %t/has-objc-symbol-and-category.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/has-swift.s -o %t/has-swift.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/has-swift-proto.s -o %t/has-swift-proto.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/no-objc.s -o %t/no-objc.o
## Make sure we don't mis-parse a 32-bit file as 64-bit
# RUN: llvm-mc -filetype=obj -triple=armv7-apple-watchos %t/no-objc.s -o %t/wrong-arch.o
# RUN: llvm-ar rcs %t/libHasSomeObjC.a %t/no-objc.o %t/has-objc-symbol.o %t/has-objc-category.o %t/has-swift.o %t/has-swift-proto.o %t/wrong-arch.o
# RUN: llvm-ar rcs %t/libHasSomeObjC2.a %t/no-objc.o %t/has-objc-symbol-and-category.o %t/has-swift.o %t/has-swift-proto.o %t/wrong-arch.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -lSystem %t/test.o -o %t/test -L%t -lHasSomeObjC -ObjC
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s --check-prefix=OBJC

# RUN: %lld -lSystem %t/test.o -o %t/test -L%t -lHasSomeObjC2 -ObjC
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s --check-prefix=OBJC

# RUN: %lld -lSystem %t/test.o -o %t/test --start-lib %t/no-objc.o %t/has-objc-symbol.o %t/has-objc-category.o %t/has-swift.o %t/has-swift-proto.o %t/wrong-arch.o --end-lib -ObjC
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s --check-prefix=OBJC

# OBJC:       Sections:
# OBJC-NEXT:  Idx Name            Size   VMA  Type
# OBJC-NEXT:    0 __text          {{.*}}      TEXT
# OBJC-NEXT:    1 __swift         {{.*}}      DATA
# OBJC-NEXT:    2 __swift5_fieldmd{{.*}}      DATA
# OBJC-NEXT:    3 __objc_catlist  {{.*}}      DATA
# OBJC-NEXT:    4 has_objc_symbol {{.*}}      DATA
# OBJC-EMPTY:
# OBJC-NEXT:  SYMBOL TABLE:
# OBJC-DAG:   g     F __TEXT,__text _main
# OBJC-DAG:   g     F __TEXT,__text _OBJC_CLASS_$_MyObject
# OBJC-DAG:   g     O __TEXT,__swift5_fieldmd $s7somelib4Blah_pMF

# RUN: %lld -lSystem %t/test.o -o %t/test -L%t -lHasSomeObjC
# RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s --check-prefix=NO-OBJC

# NO-OBJC:       Sections:
# NO-OBJC-NEXT:  Idx Name           Size   VMA  Type
# NO-OBJC-NEXT:    0 __text         {{.*}}      TEXT
# NO-OBJC-EMPTY:
# NO-OBJC-NEXT:  SYMBOL TABLE:
# NO-OBJC-NEXT:  g     F __TEXT,__text _main
# NO-OBJC-NEXT:  g     F __TEXT,__text __mh_execute_header
# NO-OBJC-NEXT:          *UND* dyld_stub_binder
# NO-OBJC-EMPTY:

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/refs-dup.s -o %t/refs-dup.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/refs-objc.s -o %t/refs-objc.o

## Check that -ObjC causes has-objc-symbol.o to be loaded first, prior to symbol
## resolution. This matches ld64's behavior.
# RUN: %lld -dylib %t/refs-dup.o %t/refs-objc.o -o %t/refs-dup -L%t -lHasSomeObjC -ObjC
# RUN: llvm-objdump --macho --syms %t/refs-dup | FileCheck %s --check-prefix=DUP-FROM-OBJC
# DUP-FROM-OBJC: g     O __DATA,has_objc_symbol _has_dup

## Without -ObjC, no-objc.o gets loaded first during symbol resolution, causing
## a duplicate symbol error.
# RUN: not %lld -dylib %t/refs-dup.o %t/refs-objc.o -o %t/refs-dup -L%t \
# RUN:   -lHasSomeObjC 2>&1 | FileCheck %s --check-prefix=DUP-ERROR
# DUP-ERROR: error: duplicate symbol: _has_dup

## TODO: Load has-objc-symbol.o prior to symbol resolution to match the archive behavior.
# RUN: not %lld -dylib %t/refs-dup.o %t/refs-objc.o -o %t/refs-dup --start-lib %t/no-objc.o \
# RUN:   %t/has-objc-symbol.o %t/has-objc-category.o %t/has-swift.o %t/wrong-arch.o --end-lib \
# RUN:   -ObjC  --check-prefix=DUP-FROM-OBJC

#--- has-objc-symbol.s
.globl _OBJC_CLASS_$_MyObject, _has_dup
_OBJC_CLASS_$_MyObject:

.section __DATA,has_objc_symbol
_has_dup:

#--- has-objc-category.s
.section __DATA,__objc_catlist
.quad 0x1234

#--- has-objc-symbol-and-category.s
## Make sure we load this archive member exactly once (i.e. no duplicate symbol
## error).
.globl _OBJC_CLASS_$_MyObject, _has_dup
_OBJC_CLASS_$_MyObject:

.section __DATA,has_objc_symbol
_has_dup:

.section __DATA,__objc_catlist
.quad 0x1234

#--- has-swift.s
.section __TEXT,__swift
.quad 0x1234

#--- has-swift-proto.s
.section __TEXT,__swift5_fieldmd
.globl $s7somelib4Blah_pMF
$s7somelib4Blah_pMF:

#--- no-objc.s
## This archive member should not be pulled in by -ObjC since it does not
## contain any ObjC-related data.
.globl _has_dup
.section __DATA,foo
.section __DATA,bar
.section __DATA,no_objc
_has_dup:

#--- test.s
.globl _main
_main:
  ret

#--- refs-dup.s
.data
.quad _has_dup

#--- refs-objc.s
.data
.quad _OBJC_CLASS_$_MyObject
