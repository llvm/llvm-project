// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/wrap.s -o %t2

// RUN: ld.lld -o %t3 %t %t2 -wrap foo -wrap nosuchsym
// RUN: llvm-objdump -d -print-imm-hex %t3 | FileCheck %s
// RUN: ld.lld -o %t3 %t %t2 --wrap foo -wrap=nosuchsym
// RUN: llvm-objdump -d -print-imm-hex %t3 | FileCheck %s

// CHECK: _start:
// CHECK-NEXT: movl $0x11010, %edx
// CHECK-NEXT: movl $0x11010, %edx
// CHECK-NEXT: movl $0x11000, %edx

// RUN: llvm-readobj -t %t3 | FileCheck -check-prefix=SYM %s
// SYM:      Name: foo
// SYM-NEXT: Value: 0x11000
// SYM:      Name: __wrap_foo
// SYM-NEXT: Value: 0x11010
// SYM:      Name: __real_foo
// SYM-NEXT: Value: 0x11020

.global _start
_start:
  movl $foo, %edx
  movl $__wrap_foo, %edx
  movl $__real_foo, %edx
