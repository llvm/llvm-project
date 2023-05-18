// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// PR5599

void test1_f(void *);

void test1_g(void) {
  float x[4];
  test1_f(x);
}
// CHECK: @test1_g
// CHECK: alloca [4 x float], align 16
