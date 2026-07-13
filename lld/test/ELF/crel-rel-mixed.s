# REQUIRES: arm
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=armv7a -crel a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=armv7a b.s -o b.o
# RUN: not ld.lld -r a.o b.o 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: b.o:(.rel.text): REL cannot be converted to CREL

#--- a.s
.global _start, foo
_start:
  bl foo
  bl .text.foo

.section .text.foo,"ax"
foo:
  nop

#--- b.s
.globl fb
fb:
  bl fb
