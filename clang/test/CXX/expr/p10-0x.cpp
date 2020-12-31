// RUN: %clang_cc1 -emit-llvm -triple x86_64-pc-linux-gnu %s -o - -std=c++11 | FileCheck %s

volatile int g1;
struct S {
  volatile int a;
} g2;

volatile int& refcall();

// CHECK: define{{.*}} void @_Z2f1PViPV1S
void f1(volatile int *x, volatile S* s) {
  // We should perform the load in these cases.
  // CHECK: load volatile i32, i32*
  (*x);
  // CHECK: load volatile i32, i32*
  __extension__ g1;
  // CHECK: load volatile i32, i32*
  s->a;
  // CHECK: load volatile i32, i32*
  g2.a;
  // CHECK: load volatile i32, i32*
  s->*(&S::a);
  // CHECK: load volatile i32, i32*
  // CHECK: load volatile i32, i32*
  x[0], 1 ? x[0] : *x;

  // CHECK: load volatile i32, i32*
  // CHECK: load volatile i32, i32*
  // CHECK: load volatile i32, i32*
  *x ?: *x;

  // CHECK: load volatile i32, i32*
  ({ *x; });

  // CHECK-NOT: load volatile
  // CHECK: ret
}

// CHECK: define{{.*}} void @_Z2f2PVi
// CHECK-NOT: load volatile
// CHECK: ret
void f2(volatile int *x) {
  // We shouldn't perform the load in these cases.
  refcall();
  1 ? refcall() : *x;
}

// CHECK: define{{.*}} void @_Z2f3v()
// CHECK-NOT: load
// CHECK-NOT: memcpy

void f3(void) {
  volatile char a[10];
  a;
}
