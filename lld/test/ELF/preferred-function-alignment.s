// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
// RUN: ld.lld --preferred-function-alignment=32 -o %t %t.o
// RUN: llvm-nm -n %t | FileCheck %s

// CHECK: 0000000000201120 t f1
.section .text.f1,"ax",@progbits
f1:
ret

// CHECK: 0000000000201140 t f2
.section .text.f2,"ax",@progbits
f2:
ret

