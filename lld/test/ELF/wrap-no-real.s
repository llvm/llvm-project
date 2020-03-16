// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/wrap-no-real.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/wrap-no-real2.s -o %t3.o
// RUN: ld.lld -o %t3.so -shared %t3.o

// RUN: ld.lld -o %t %t1.o %t2.o -wrap foo
// RUN: llvm-objdump -d --print-imm-hex %t | FileCheck %s

// RUN: ld.lld -o %t %t1.o %t2.o %t3.so -wrap foo
// RUN: llvm-objdump -d --print-imm-hex %t | FileCheck %s

// CHECK: <_start>:
// CHECK-NEXT: movl $0x11010, %edx
// CHECK-NEXT: movl $0x11010, %edx
// CHECK-NEXT: movl $0x11000, %edx

// RUN: llvm-objdump -t %t | FileCheck --check-prefix=SYM %s


// SYM:      {{.*}}           l .dynamic 0000000000000000 .hidden _DYNAMIC
// SYM-NEXT: 0000000000011000 g *ABS*    0000000000000000 __real_foo
// SYM-NEXT: 0000000000011010 g *ABS*    0000000000000000 __wrap_foo
// SYM-NEXT: {{.*}}           g .text    0000000000000000 _start
// SYM-NEXT: 0000000000011000 g *ABS*    0000000000000000 foo

.global _start
_start:
  movl $foo, %edx
  movl $__wrap_foo, %edx
  movl $__real_foo, %edx
