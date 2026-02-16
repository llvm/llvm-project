// RUN: %clang_cc1 %s -triple x86_64 -O2 \
// RUN:  -mllvm -regalloc=pbqp \
// RUN:  -mllvm --print-changed -S |& FileCheck %s
// CHECK: IR Dump After PBQP Register Allocator (regallocpbqp) on foo

extern int foo(int a, int b) {
  return a + b;
}