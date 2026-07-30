// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

void foo(int *P) {
  // CHECK: llvm.prefetch
  __builtin_prefetch(P);
  __builtin_prefetch(P, 1);
}
