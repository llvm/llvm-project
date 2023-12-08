## Test that we don't create a .ARM.exidx for the main partition.
## Previously we were doing so, which is unnecessary and led to a crash.

# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %p/Inputs/shared.s -o %t1.o
# RUN: ld.lld -shared %t1.o -o %t.so
# RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o

# RUN: ld.lld -shared --gc-sections --dynamic-linker foo %t.o %t.so -o %t
# RUN: llvm-readelf --section-headers %t | FileCheck %s

# CHECK: .ARM.exidx
# CHECK-NOT: .ARM.exidx

.section .llvm_sympart,"",%llvm_sympart
.asciz "part1"
.4byte p1

.section .text.p1,"ax",%progbits
.globl p1
p1:
.fnstart
bx lr
.cantunwind
.fnend
