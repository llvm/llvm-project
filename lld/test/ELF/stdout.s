# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o - > %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK: 0000000000201000 _start:
# CHECK: 201000: 90 nop

.globl _start
_start:
  nop
