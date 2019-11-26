// RUN: %clang_cc1 -triple i386-unknown-unknown -O1 -emit-llvm -o - %s | FileCheck %s
// CHECK-LABEL: define i32 @f0()
// CHECK:   ret i32 0
// CHECK-LABEL: define i32 @f1()
// CHECK:   ret i32 0
// CHECK-LABEL: define i32 @f2()
// CHECK:   ret i32 0
// <rdar://problem/6113085>

struct s0 {
  int x, y;
};

int f0() {
  struct s0 x = {0};
  return x.y;
}

int f1() {
  struct s0 x[2] = { {0} };
  return x[1].x;
}

int f2() {
  int x[2] = { 0 };
  return x[1];
}

