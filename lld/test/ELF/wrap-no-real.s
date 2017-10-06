// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/wrap-no-real.s -o %t2.o

// RUN: ld.lld -o %t %t1.o %t2.o -wrap foo
// RUN: llvm-objdump -d -print-imm-hex %t | FileCheck %s

// CHECK: _start:
// CHECK-NEXT: movl $0x11010, %edx
// CHECK-NEXT: movl $0x11010, %edx
// CHECK-NEXT: movl $0x11000, %edx

// RUN: llvm-readobj -t -s %t | FileCheck -check-prefix=SYM %s
// SYM-NOT:  Name: __real_foo
// SYM:      Name: foo
// SYM-NEXT: Value: 0x11000
// SYM-NOT:  Name: __real_foo
// SYM:      Name: __wrap_foo
// SYM-NEXT: Value: 0x11010
// SYM-NOT:  Name: __real_foo

.global _start
_start:
  movl $foo, %edx
  movl $__wrap_foo, %edx
  movl $__real_foo, %edx
