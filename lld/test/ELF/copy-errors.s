// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/protected-shared.s -o %t2.o
// RUN: ld.lld %t2.o -o %t2.so -shared
// RUN: not ld.lld %t.o %t2.so -o %t 2>&1 | FileCheck %s

.global _start
_start:


call bar
// CHECK: cannot preempt symbol bar

call zed
// CHECK: symbol zed is missing type
