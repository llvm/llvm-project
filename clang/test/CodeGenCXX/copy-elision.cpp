// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown-gnu -emit-llvm -O1 -fexperimental-new-pass-manager -o - %s | FileCheck %s

template <class T> T test() {
  return T();
}

struct A {
  A();
  A(A &);
  A(int);
  operator int();
};

// FIXME: There should be copy elision here.
// CHECK-LABEL: define{{.*}} void @_Z4testI1AET_v
// CHECK:       call void @_ZN1AC1Ev
// CHECK-NEXT:  call noundef i32 @_ZN1AcviEv
// CHECK-NEXT:  call void @_ZN1AC1Ei
// CHECK-NEXT:  call void @llvm.lifetime.end
template A test<A>();

struct BSub {};
struct B : BSub {
  B();
  B(B &);
  B(const BSub &);
};

// FIXME: There should be copy elision here.
// CHECK-LABEL: define{{.*}} void @_Z4testI1BET_v
// CHECK:       call void @_ZN1BC1Ev
// CHECK:       call void @_ZN1BC1ERK4BSub
// CHECK-NEXT:  call void @llvm.lifetime.end
template B test<B>();
