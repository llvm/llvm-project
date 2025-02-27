# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
## Test that we only write to "-" once.
# RUN: env LLD_IN_TEST=2 ld.lld %t.o -o - > %t1
# RUN: llvm-objdump -d %t1 | FileCheck %s

# CHECK: nop

# RUN: ld.lld %t.o -o %t2
# RUN: diff %t1 %t2

.globl _start
_start:
  nop
