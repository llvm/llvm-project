// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

struct T {
  T();
  ~T();
};

void f(const T& t = T());

class X { // ...
public:
        X();
        X(const X&, const T& t = T());
};

// CHECK-LABEL: define{{.*}} void @_Z1gv()
void g() {
  // CHECK:      call void @_ZN1TC1Ev([[T:%.*]]* {{[^,]*}} [[AGG1:%.*]])
  // CHECK-NEXT: call void @_Z1fRK1T([[T]]* noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[AGG1]])
  // CHECK-NEXT: call void @_ZN1TD1Ev([[T]]* {{[^,]*}} [[AGG1]])
  f();

  // CHECK-NEXT: call void @_ZN1TC1Ev([[T:%.*]]* {{[^,]*}} [[AGG2:%.*]])
  // CHECK-NEXT: call void @_Z1fRK1T([[T]]* noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[AGG2]])
  // CHECK-NEXT: call void @_ZN1TD1Ev([[T]]* {{[^,]*}} [[AGG2]])
  f();

  // CHECK-NEXT: call void @_ZN1XC1Ev(
  X a;

  // CHECK-NEXT: call void @_ZN1TC1Ev(
  // CHECK-NEXT: call void @_ZN1XC1ERKS_RK1T(
  // CHECK-NEXT: call void @_ZN1TD1Ev(
  X b(a);

  // CHECK-NEXT: call void @_ZN1TC1Ev(
  // CHECK-NEXT: call void @_ZN1XC1ERKS_RK1T(
  // CHECK-NEXT: call void @_ZN1TD1Ev(
  X c = a;
}


class obj{ int a; float b; double d; };
// CHECK-LABEL: define{{.*}} void @_Z1hv()
void h() {
  // CHECK: call void @llvm.memset.p0i8.i64(
  obj o = obj();
}

// PR7028 - mostly this shouldn't crash
namespace test1 {
  struct A { A(); };
  struct B { B(); ~B(); };

  struct C {
    C(const B &file = B());
  };
  C::C(const B &file) {}

  struct D {
    C c;
    A a;

    // CHECK-LABEL: define linkonce_odr void @_ZN5test11DC2Ev(%"struct.test1::D"* {{[^,]*}} %this) unnamed_addr
    // CHECK:      call void @_ZN5test11BC1Ev(
    // CHECK-NEXT: call void @_ZN5test11CC1ERKNS_1BE(
    // CHECK-NEXT: call void @_ZN5test11BD1Ev(
    // CHECK:      call void @_ZN5test11AC1Ev(
    D() : c(), a() {}
  };

  D d;
}

namespace test2 {
  // CHECK:  define linkonce_odr void @_ZN5test21AIiED2Ev(
  template <typename T> struct A { A() {} ~A() {} };
  template <typename> void f(const A<int> & = {}) {}
  void g() { f<int>(); }
}
