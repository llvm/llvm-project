// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// PR5484
namespace PR5484 {
struct A { };
extern A a;

void f(const A & = a);

void g() {
  f();
}
}

struct A1 {
 A1();
 ~A1();
};

struct A2 {
 A2();
 ~A2();
};

struct B {
 B(const A1& = A1(), const A2& = A2());
};

// CHECK-LABEL: define{{.*}} void @_Z2f1v()
void f1() {

 // CHECK: call void @_ZN2A1C1Ev(
 // CHECK: call void @_ZN2A2C1Ev(
 // CHECK: call void @_ZN1BC1ERK2A1RK2A2(
 // CHECK: call void @_ZN2A2D1Ev
 // CHECK: call void @_ZN2A1D1Ev
 B bs[2];
}

struct C {
 B bs[2];
 C();
};

// CHECK-LABEL: define{{.*}} void @_ZN1CC2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN2A1C1Ev(
// CHECK: call void @_ZN2A2C1Ev(
// CHECK: call void @_ZN1BC1ERK2A1RK2A2(
// CHECK: call void @_ZN2A2D1Ev
// CHECK: call void @_ZN2A1D1Ev

// CHECK-LABEL: define{{.*}} void @_ZN1CC1Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN1CC2Ev(
C::C() { }

// CHECK-LABEL: define{{.*}} void @_Z2f3v()
void f3() {
 // CHECK: call void @_ZN2A1C1Ev(
 // CHECK: call void @_ZN2A2C1Ev(
 // CHECK: call void @_ZN1BC1ERK2A1RK2A2(
 // CHECK: call void @_ZN2A2D1Ev
 // CHECK: call void @_ZN2A1D1Ev
 B *bs = new B[2];
 delete bs;
}

void f4() {
  void g4(int a, int b = 7);
  {
    void g4(int a, int b = 5);
  }
  void g4(int a = 5, int b);

  // CHECK: call void @_Z2g4ii(i32 noundef 5, i32 noundef 7)
  g4();
}
