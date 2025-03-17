// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s

int a[10];
// CHECK: @a = external dso_local global [10 x i32]

int aa[10][5];
// CHECK: @aa = external dso_local global [10 x [5 x i32]]

extern int b[10];
// CHECK: @b = external dso_local global [10 x i32]

extern int bb[10][5];
// CHECK: @bb = external dso_local global [10 x [5 x i32]]

void f() {
  int l[10];
}
// CHECK: define void @f()
// CHECK-NEXT: alloca [10 x i32], i64 1, align 16

void f2(int p[10]) {}
// CHECK: define void @f2(ptr {{%.*}})
// CHECK-NEXT: alloca ptr, i64 1, align 8

void f3(int pp[10][5]) {}
// CHECK: define void @f3(ptr {{%.*}})
// CHECK-NEXT: alloca ptr, i64 1, align 8
