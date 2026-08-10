// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++11 -emit-llvm -o - %s | FileCheck %s

__attribute__((internal_linkage)) void f() {}
// CHECK-DAG: define internal void @_ZL1fv

class A {
public:
  static int y __attribute__((internal_linkage));
  static int y2 [[clang::internal_linkage]];
// CHECK-DAG: @_ZN1A1yE = internal global
// CHECK-DAG: @_ZN1A2y2E = internal global
  void f1() __attribute__((internal_linkage));
// CHECK-DAG: define internal void @_ZN1A2f1Ev
  void f2() __attribute__((internal_linkage)) {}
// CHECK-DAG: define internal void @_ZN1A2f2Ev
  static void f4() __attribute__((internal_linkage)) {}
// CHECK-DAG: define internal void @_ZN1A2f4Ev
  A() __attribute__((internal_linkage)) {}
// CHECK-DAG: define internal void @_ZN1AC1Ev
// CHECK-DAG: define internal void @_ZN1AC2Ev
  ~A() __attribute__((internal_linkage)) {}
// CHECK-DAG: define internal void @_ZN1AD1Ev
// CHECK-DAG: define internal void @_ZN1AD2Ev
};

int A::y;
int A::y2;

void A::f1() {
}

// Forward declaration w/o an attribute.
class B;

// Internal_linkage on a class affects all its members.
class __attribute__((internal_linkage)) B {
public:
  B() {}
  // CHECK-DAG: define internal void @_ZNL1BC1Ev
  // CHECK-DAG: define internal void @_ZNL1BC2Ev
  ~B() {}
  // CHECK-DAG: define internal void @_ZNL1BD1Ev
  // CHECK-DAG: define internal void @_ZNL1BD2Ev
  void f() {};
  // CHECK-DAG: define internal void @_ZNL1B1fEv
  static int x;
  // CHECK-DAG: @_ZNL1B1xE = internal global
};

int B::x;

// Forward declaration with the attribute.
class __attribute__((internal_linkage)) C;
class C {
public:
  static int x;
  // CHECK-DAG: @_ZNL1C1xE = internal global
};

int C::x;

__attribute__((internal_linkage)) void g();
void g() {}
// CHECK-DAG: define internal void @_ZL1gv()

void use() {
  A a;
  a.f1();
  a.f2();
  A::f4();
  f();
  int &Y = A::y;
  int &Y2 = A::y2;
  B b;
  b.f();
  int &XX2 = B::x;
  g();
  int &XX3 = C::x;
}
