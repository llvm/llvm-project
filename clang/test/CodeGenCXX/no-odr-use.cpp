// RUN: %clang_cc1 -std=c++11 -emit-llvm -o - -triple x86_64-linux-gnu %s | FileCheck %s --check-prefixes=CHECK,CHECK-CXX11
// RUN: %clang_cc1 -std=c++2a -emit-llvm -o - -triple x86_64-linux-gnu %s | FileCheck %s --check-prefixes=CHECK,CHECK-CXX2A

// CHECK-DAG: @__const._Z1fi.a = private unnamed_addr constant {{.*}} { i32 1, [2 x i32] [i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6] }
// CHECK-CXX11-DAG: @_ZN7PR422765State1mE.const = private unnamed_addr constant [2 x { i64, i64 }] [{ {{.*}} @_ZN7PR422765State2f1Ev {{.*}}, i64 0 }, { {{.*}} @_ZN7PR422765State2f2Ev {{.*}}, i64 0 }]
// CHECK-CXX2A-DAG: @_ZN7PR422765State1mE = linkonce_odr constant [2 x { i64, i64 }] [{ {{.*}} @_ZN7PR422765State2f1Ev {{.*}}, i64 0 }, { {{.*}} @_ZN7PR422765State2f2Ev {{.*}}, i64 0 }], comdat

struct A { int x, y[2]; int arr[3]; };
// CHECK-LABEL: define{{.*}} i32 @_Z1fi(
int f(int i) {
  // CHECK: call void {{.*}}memcpy{{.*}}({{.*}}, {{.*}} @__const._Z1fi.a
  constexpr A a = {1, 2, 3, 4, 5, 6};

  // CHECK-LABEL: define {{.*}}@"_ZZ1fiENK3$_0clEiM1Ai"(
  return [] (int n, int A::*p) {
    // CHECK: br i1
    return (n >= 0
      // CHECK: getelementptr inbounds [3 x i32], ptr getelementptr inbounds ({{.*}} @__const._Z1fi.a, i32 0, i32 2), i64 0, i64 %
      ? a.arr[n]
      // CHECK: br i1
      : (n == -1
        // CHECK: getelementptr inbounds i8, ptr @__const._Z1fi.a, i64 %
        // CHECK: load i32
        ? a.*p
        // CHECK: getelementptr inbounds [2 x i32], ptr getelementptr inbounds ({{.*}} @__const._Z1fi.a, i32 0, i32 1), i64 0, i64 %
        // CHECK: load i32
        : a.y[2 - n]));
  }(i, &A::x);
}

namespace PR42276 {
  class State {
    void syncDirtyObjects();
    void f1(), f2();
    using l = void (State::*)();
    static constexpr l m[]{&State::f1, &State::f2};
  };
  // CHECK-LABEL: define{{.*}} void @_ZN7PR422765State16syncDirtyObjectsEv(
  void State::syncDirtyObjects() {
    for (int i = 0; i < sizeof(m) / sizeof(m[0]); ++i)
      // CHECK-CXX11: getelementptr inbounds [2 x { i64, i64 }], ptr @_ZN7PR422765State1mE.const, i64 0, i64 %
      // CHECK-CXX2A: getelementptr inbounds [2 x { i64, i64 }], ptr @_ZN7PR422765State1mE, i64 0, i64 %
      (this->*m[i])();
  }
}
