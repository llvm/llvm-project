// RUN: %clang_cc1 -triple lanai-unknown-unknown -mregparm 4 %s -emit-llvm -o - | FileCheck %s

void f1(int a, int b, int c, int d,
        int e, int f, int g, int h);

void f2(int a, int b) __attribute((regparm(0)));

void f0() {
// CHECK: call void @f1(i32 inreg noundef 1, i32 inreg noundef 2, i32 inreg noundef 3, i32 inreg noundef 4,
// CHECK: i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8)
  f1(1, 2, 3, 4, 5, 6, 7, 8);
// CHECK: call void @f2(i32 noundef 1, i32 noundef 2)
  f2(1, 2);
}

// CHECK: declare void @f1(i32 inreg noundef, i32 inreg noundef, i32 inreg noundef, i32 inreg noundef,
// CHECK: i32 noundef, i32 noundef, i32 noundef, i32 noundef)
// CHECK: declare void @f2(i32 noundef, i32 noundef)
