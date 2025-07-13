# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o %t2.o
# RUN: ld.lld -shared -soname=t2 %t2.o -o %t2.so
# RUN: ld.lld -gc-sections -export-dynamic %t.o %t2.so -o %t
# RUN: llvm-readobj --dyn-symbols %t | FileCheck %s

# CHECK: Name: bar
# CHECK: Name: foo

.comm bar,4,4
.comm foo,4,4

.text
.globl _start
_start:
 .quad foo
