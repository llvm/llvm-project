# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: %lld %t.o -lSystem -o %t.out
# RUN: llvm-readobj --syms %t.out | FileCheck %s

## ld64 has a "TEMP work around until <rdar://problem/7702923> goes in"
## that promotes PrivateExtern ReferencedDynamically symbols in dylibs to
## normal Externs. lld does not do this.
# RUN: %lld -dylib %t.o -o %t.dylib
# RUN: llvm-readobj --syms %t.dylib | FileCheck %s

# CHECK:         Name: ___crashreporter_info__
# CHECK-NEXT:    PrivateExtern
# CHECK-NEXT:    Type: Section (0xE)
# CHECK-NEXT:    Section: __common
# CHECK-NEXT:    RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:    Flags [ (0x10)
# CHECK-NEXT:      ReferencedDynamically (0x10)
# CHECK-NEXT:    ]

## Reduced from lib/Support/PrettyStackTrace.cpp
.section __TEXT,__text,regular,pure_instructions

.globl _main
_main:
  ret

## .private_extern maches what PrettyStackTrace.cpp does, but it makes
## the ReferencedDynamically pointless: https://reviews.llvm.org/D27683#2763729
.private_extern ___crashreporter_info__
.globl ___crashreporter_info__
.desc ___crashreporter_info__,16
.zerofill __DATA,__common,___crashreporter_info__,8,3
.subsections_via_symbols
