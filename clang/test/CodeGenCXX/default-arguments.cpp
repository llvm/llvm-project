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

namespace GH113324 {
struct S1 {
  friend void f1(S1, int = 42) {}
};

template <bool, class> using __enable_if_t = int;
template <int v> struct S2 {
  static const int value = v;
};
struct S3 {
  template <__enable_if_t<S2<true>::value, int> = 0> S3(const char *);
};
struct S4 {
  template <typename a, typename b> friend void f2(int, S4, a, b, S3 = "") {}
};

void test() {
  S1 s1;
  f1(s1);

  S4 s4;
  f2(0, s4, [] {}, [] {});
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
