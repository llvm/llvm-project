## Check that group members are retained or discarded as a unit, and
## sections whose names are C identifiers aren't considered roots if
## they're members of a group.

# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o --gc-sections -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# RUN: echo ".global __start___data; __start___data:" > %t2.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t2.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld %t.o --gc-sections -o %t2 %t2.so
# RUN: llvm-readelf -s %t2 | FileCheck %s

# CHECK:     [[#%x,ADDR:]] {{.*}} __start___data
# CHECK:     [[#ADDR + 8]] {{.*}} __stop___data
# CHECK:     _start
# CHECK:     f
# CHECK-NOT: g

.weak __start___data
.weak __stop___data

.section .text,"ax",@progbits
.global _start
_start:
  .quad __start___data - .
  .quad __stop___data - .
  call f

.section __data,"axG",@progbits,f
.quad 0

.section .text.f,"axG",@progbits,f
.global f
f:
  nop

.section __data,"axG",@progbits,g
.quad 0

.section .text.g,"axG",@progbits,g
.global g
g:
  nop
