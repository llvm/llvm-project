// REQUIRES: ppc

// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/ppc64-func.s -o %t3.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t.o %t2.so %t3.o -o %t
// RUN: llvm-objdump -d %t | FileCheck %s

// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/ppc64-func.s -o %t3.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t.o %t2.so %t3.o -o %t
// RUN: llvm-objdump -d %t | FileCheck %s

    .text
    .abiversion 2
.global bar_local
bar_local:
  li 3, 2
  blr

# Calling external function foo in a shared object needs a nop.
# Calling local function bar_local doe snot need a nop.
// CHECK: Disassembly of section .text:
.global _start
_start:
  bl foo
  nop
  bl bar_local

// CHECK: _start:
// CHECK: 10010008:       {{.*}}     bl .+72
// CHECK-NOT: 1001000c:       {{.*}}     nop
// CHECK: 1001000c:       {{.*}}     ld 2, 24(1)
// CHECK: 10010010:       {{.*}}     bl .+67108848
// CHECK-NOT: 10010014:       {{.*}}     nop
// CHECK-NOT: 10010014:       {{.*}}     ld 2, 24(1)

# Calling a function in another object file which will have same
# TOC base does not need a nop. If nop present, do not rewrite to
# a toc restore
.global diff_object
_diff_object:
  bl foo_not_shared
  bl foo_not_shared
  nop

// CHECK: _diff_object:
// CHECK-NEXT: 10010014:       {{.*}}     bl .+28
// CHECK-NEXT: 10010018:       {{.*}}     bl .+24
// CHECK-NEXT: 1001001c:       {{.*}}     nop

# Branching to a local function does not need a nop
.global noretbranch
noretbranch:
  b bar_local
// CHECK: noretbranch:
// CHECK: 10010020:       {{.*}}     b .+67108832
// CHECK-NOT: 10010024:       {{.*}}     nop
// CHECK-NOT: 10010024:       {{.*}}     ld 2, 24(1)

// This should come last to check the end-of-buffer condition.
.global last
last:
  bl foo
  nop
// CHECK: last:
// CHECK: 10010024:       {{.*}}     bl .+44
// CHECK-NEXT: 10010028:       {{.*}}     ld 2, 24(1)

// CHECK: Disassembly of section .plt:
// CHECK: .plt:
// CHECK-NEXT: 10010050:       {{.*}}     std 2, 24(1)
// CHECK-NEXT: 10010054:       {{.*}}     addis 12, 2, 4098
// CHECK-NEXT: 10010058:       {{.*}}     ld 12, -32752(12)
// CHECK-NEXT: 1001005c:       {{.*}}     mtctr 12
