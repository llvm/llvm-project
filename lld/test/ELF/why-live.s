# REQUIRES: x86

# RUN: llvm-mc -n -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null --gc-sections --why-live=a | FileCheck %s

# CHECK: live symbol: a
# CHECK: >>> alive because of {{.*}}.o:(._start)
# CHECK: >>> alive because of _start

.globl _start
.section ._start,"ax",@progbits
_start:
jmp a

.globl a
.section .a,"ax",@progbits
a:
jmp a

