# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/bar.s -o %t/bar.o

## "_llvm." symbols are not special. LLD would produce duplicate symbol errors
## if they were not within the LLVM segment.

## 1/ Test that LLD does not produce duplicate symbols errors when linking global symbols
##    with the same name under the LLVM segment.
# RUN: %lld -dylib %t/foo.o %t/bar.o -o %t/libDuplicate.dylib

## 2/ Test that all sections within an LLVM segment are dropped.
# RUN: llvm-objdump --section-headers %t/libDuplicate.dylib | FileCheck %s

# CHECK-LABEL: Sections:
# CHECK-NEXT:  Idx  Name    Size      VMA            Type
# CHECK-NEXT:  0    __text  00000000  {{[0-9a-f]+}}  TEXT

## 3/ Test that linking global symbol that is not under the LLVM segment produces duplicate
##    symbols
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin --defsym TEXT=0 %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin --defsym TEXT=0 %t/bar.s -o %t/bar.o
# RUN: not %lld -dylib %t/foo.o %t/bar.o -o %t/libDuplicate.dylib 2>&1 | FileCheck %s --check-prefix=DUP

# DUP: error: duplicate symbol: _llvm.foo

#--- foo.s
.globl _llvm.foo
.ifdef TEXT
  .section __TEXT,__cstring
.else
  .section __LLVM,__bitcode
.endif
  _llvm.foo:
    .asciz "test"

#--- bar.s
.globl _llvm.foo
.ifdef TEXT
  .section __TEXT,__cstring
.else
  .section __LLVM,__bitcode
.endif
  _llvm.foo:
    .asciz "test"
