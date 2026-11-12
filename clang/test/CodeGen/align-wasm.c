// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm %s -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -emit-llvm %s -o - \
// RUN:   | FileCheck %s

void test1_f(void *);

void test1_g(void) {
  float x[4];
  test1_f(x);
}
// CHECK: @test1_g
// CHECK: alloca [4 x float], align 16
