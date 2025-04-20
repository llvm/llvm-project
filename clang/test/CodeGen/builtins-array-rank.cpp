// RUN: %clang_cc1 -triple aarch64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

unsigned long array_rank_binary_operator(void) {
  // CHECK: ret i64 3
  return __array_rank(int[10]) | 2;
}
