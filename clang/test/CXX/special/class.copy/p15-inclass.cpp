// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

namespace PR11418 {
  struct NonPOD {
    NonPOD();
    NonPOD(const NonPOD &);
    NonPOD(NonPOD &&);
  };

  struct X {
    NonPOD np;
    int a = 17;
  };

  void check_copy(X x) {
    X x2(x);
  }

  void check_move(X x) {
    X x3(static_cast<X&&>(x));
  }

  // CHECK: define linkonce_odr void @_ZN7PR114181XC2ERKS0_
  // CHECK-NOT: 17
  // CHECK: call void @_ZN7PR114186NonPODC1ERKS0_
  // CHECK-NOT: 17
  // CHECK: load i32, i32*
  // CHECK-NOT: 17
  // CHECK: store i32
  // CHECK-NOT: 17
  // CHECK: ret

  // CHECK: define linkonce_odr void @_ZN7PR114181XC2EOS0_
  // CHECK-NOT: 17
  // CHECK: call void @_ZN7PR114186NonPODC1EOS0_
  // CHECK-NOT: 17
  // CHECK: load i32, i32*
  // CHECK-NOT: 17
  // CHECK: store i32
  // CHECK-NOT: 17
  // CHECK: ret
}
