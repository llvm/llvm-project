// RUN: %clang_cc1 -std=c11 -emit-llvm -o - -triple x86_64-unknown-linux %s | FileCheck %s

// Tests for issue #207785.

#define C(x) __attribute__((cleanup(x)))
void f1(double *x);
void f2(double *x);
void f3(double *x);

// CHECK-LABEL: define {{.*}} void @g1()
void g1() {
  // CHECK: call void @f3
  // CHECK-NOT: call void @f2
  // CHECK-NOT: call void @f1
  C(f1) C(f2) C(f3) double x;
}

// CHECK-LABEL: define {{.*}} void @g2()
void g2() {
  // CHECK: call void @f1
  C(f1) C(f1) double x;
}

// CHECK-LABEL: define {{.*}} void @g3()
void g3() {
  // CHECK: call void @f1
  // CHECK-NOT: call void @f2
  C(f2) C(f1) double x;
}
