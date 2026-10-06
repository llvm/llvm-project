// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-unknown-linux %s | FileCheck %s

// Tests for issue #207785.

void f1(double *x);
void f2(double *x);
void f3(double *x);

// CHECK-LABEL: define {{.*}} void @g1()
void g1() {
  // CHECK: call void @f3
  // CHECK-NOT: call void @f2
  // CHECK-NOT: call void @f1
  __attribute__((cleanup(f1)))
  __attribute__((cleanup(f2)))
  __attribute__((cleanup(f3)))
  double x;
}

// CHECK-LABEL: define {{.*}} void @g2()
void g2() {
  // CHECK: call void @f1
  __attribute__((cleanup(f1)))
  __attribute__((cleanup(f1)))
  double x;
}

// CHECK-LABEL: define {{.*}} void @g3()
void g3() {
  // CHECK: call void @f1
  // CHECK-NOT: call void @f2
  __attribute__((cleanup(f2)))
  __attribute__((cleanup(f1)))
  double x;
}

// CHECK-LABEL: define {{.*}} void @g4()
void g4() {
  // CHECK: call void @f1
  // CHECK-NOT: call void @f2
  [[gnu::cleanup(f2)]]
  [[gnu::cleanup(f1)]]
  double x;
}

// CHECK-LABEL: define {{.*}} void @g5()
void g5() {
  // CHECK: call void @f1
  // CHECK-NOT: call void @f2
  [[gnu::cleanup(f2)]]
  __attribute__((cleanup(f1)))
  double x;
}
