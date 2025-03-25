// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s

int a[10];
// CHECK: @a = external dso_local global [10 x i32]

int aa[10][5];
// CHECK: @aa = external dso_local global [10 x [5 x i32]]

extern int b[10];
// CHECK: @b = external dso_local global [10 x i32]

extern int bb[10][5];
// CHECK: @bb = external dso_local global [10 x [5 x i32]]

int c[10] = {};
// CHECK: @c = dso_local global [10 x i32] zeroinitializer

int d[3] = {1, 2, 3};
// CHECK: @d = dso_local global [3 x i32] [i32 1, i32 2, i32 3]

int dd[3][2] = {{1, 2}, {3, 4}, {5, 6}};
// CHECK: @dd = dso_local global [3 x [2 x i32]] [
// CHECK: [2 x i32] [i32 1, i32 2], [2 x i32]
// CHECK: [i32 3, i32 4], [2 x i32] [i32 5, i32 6]]

int e[10] = {1, 2};
// CHECK: @e = dso_local global [10 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0]

int f[5] = {1, 2};
// CHECK: @f = dso_local global [5 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0]

void func() {
  int l[10];
}
// CHECK: define void @func()
// CHECK-NEXT: alloca [10 x i32], i64 1, align 16

void func2(int p[10]) {}
// CHECK: define void @func2(ptr {{%.*}})
// CHECK-NEXT: alloca ptr, i64 1, align 8

void func3(int pp[10][5]) {}
// CHECK: define void @func3(ptr {{%.*}})
// CHECK-NEXT: alloca ptr, i64 1, align 8
