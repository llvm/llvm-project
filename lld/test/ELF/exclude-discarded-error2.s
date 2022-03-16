# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.section .foo,"aG",@progbits,zz,comdat; .weak foo; foo:' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# CHECK: NOTYPE WEAK DEFAULT UND foo

.globl _start
_start:
  jmp foo

.section .foo,"aG",@progbits,zz,comdat
