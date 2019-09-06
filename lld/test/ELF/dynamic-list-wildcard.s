# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "{ foo1*; };" > %t.list
# RUN: ld.lld -pie --dynamic-list %t.list %t -o %t
# RUN: llvm-nm -D %t | FileCheck %s

# CHECK:      foo1
# CHECK-NEXT: foo11
# CHECK-NOT:  {{.}}

.globl _start, foo1, foo11, foo2
foo1:
foo11:
foo2:
_start:
