// RUN: %clang_cc1 -triple arc-unknown-unknown %s -emit-llvm -o - \
// RUN:   | FileCheck %s

// 64-bit fields need only be 32-bit aligned for arc.

typedef struct {
  int aa;
  double bb;
} s1;

// CHECK: define{{.*}} i32 @f1
// CHECK: ret i32 12
int f1() {
  return sizeof(s1);
}

typedef struct {
  int aa;
  long long bb;
} s2;
// CHECK: define{{.*}} i32 @f2
// CHECK: ret i32 12
int f2() {
  return sizeof(s2);
}

