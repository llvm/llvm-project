// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu       -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc      -emit-llvm %s -o - | FileCheck --check-prefix=MSVC %s
// PR5599

char arr[16];

void test1_f(void *);

void test1_g(void) {
  float x[4];
  test1_f(x);
}
// CHECK: @arr = {{.*}} align 16
// CHECK: @test1_g
// CHECK: alloca [4 x float], align 16

// The "large array" alignment increase does not apply on windows-msvc.
// MSVC: @arr = {{.*}} align 8
// MSVC: @test1_g
// MSVC: alloca [4 x float], align 4
