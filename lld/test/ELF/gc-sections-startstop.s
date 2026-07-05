## Check that group members are retained or discarded as a unit.

# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o --gc-sections -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# RUN: echo ".global __start___data; __start___data:" > %t2.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t2.s -o %t2.o
# RUN: ld.lld -shared %t2.o --soname=t2 -o %t2.so
# RUN: ld.lld %t.o --gc-sections -o %t2 %t2.so
# RUN: llvm-readelf -s %t2 | FileCheck %s

## The referenced __data is retained.
# CHECK:     [[#%x,ADDR:]] {{.*}} __start___data
# CHECK:     [[#ADDR + 8]] {{.*}} __stop___data
## __libc_atexit is retained even if there is no reference, as a workaround for
## glibc<2.34 (BZ #27492).
# CHECK:     [[#%x,ADDR:]] {{.*}} __start___libc_atexit
# CHECK:     [[#ADDR + 8]] {{.*}} __stop___libc_atexit
# CHECK:     _start
# CHECK:     f
# CHECK-NOT: g

## If -z nostart-stop-gc, sections whose names are C identifiers are retained by
## __start_/__stop_ references.
# RUN: ld.lld %t.o %t2.so --gc-sections -z nostart-stop-gc -o %t3
# RUN: llvm-readelf -s %t3 | FileCheck %s --check-prefix=NOGC
# NOGC:     [[#%x,ADDR:]] {{.*}} __start___data
# NOGC:     [[#ADDR + 16]] {{.*}} __stop___data

.weak __start___data
.weak __stop___data
.weak __start___libc_atexit
.weak __stop___libc_atexit

.section .text,"ax",@progbits
.global _start
_start:
  .quad __start___data - .
  .quad __stop___data - .
  .quad __start___libc_atexit - .
  .quad __stop___libc_atexit - .
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

.section __libc_atexit,"a",@progbits
.quad 0
