// RUN: %clang_cc1 -std=c++1z %s -triple x86_64-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - | FileCheck %s

namespace Constant {
  struct A {
    int n;
    char k;
    ~A();
  };

  struct B {
    char k2;
  };

  struct C : B {};

  struct D : A, C {};

  C c1 = {};
  C c2 = {1};
  // CHECK: @_ZN8Constant2c1E ={{.*}} global { i8 } zeroinitializer, align 1
  // CHECK: @_ZN8Constant2c2E ={{.*}} global { i8 } { i8 1 }, align 1

  // Test packing bases into tail padding.
  D d1 = {};
  D d2 = {1, 2, 3};
  D d3 = {1};
  // CHECK: @_ZN8Constant2d1E ={{.*}} global { i32, i8, i8 } zeroinitializer, align 4
  // CHECK: @_ZN8Constant2d2E ={{.*}} global { i32, i8, i8 } { i32 1, i8 2, i8 3 }, align 4
  // CHECK: @_ZN8Constant2d3E ={{.*}} global { i32, i8, i8 } { i32 1, i8 0, i8 0 }, align 4

  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN8Constant1DD1Ev, {{.*}} @_ZN8Constant2d1E

  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN8Constant1DD1Ev, {{.*}} @_ZN8Constant2d2E

  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN8Constant1DD1Ev, {{.*}} @_ZN8Constant2d3E
}

namespace Dynamic {
  struct A {
    A();
    A(int);
    A(const char*, unsigned);
    ~A();
    void *p;
  };

  struct B {
    ~B();
    int n = 5;
  };

  struct C {
    C(bool = true);
  };

  int f(), g(), h(), i();
  struct D : A, B, C {
    int n = f();
  };

  D d1 = {};
  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: call void @_ZN7Dynamic1AC2Ev({{.*}} @_ZN7Dynamic2d1E
  // CHECK: store i32 5, {{.*}}ptr getelementptr inbounds {{.*}} @_ZN7Dynamic2d1E{{.*}}, i64 8
  // CHECK: invoke void @_ZN7Dynamic1CC2Eb({{.*}} @_ZN7Dynamic2d1E{{.*}}, i1 noundef zeroext true)
  // CHECK:   unwind label %[[UNWIND:.*]]
  // CHECK: invoke noundef i32 @_ZN7Dynamic1fEv()
  // CHECK:   unwind label %[[UNWIND:.*]]
  // CHECK: store i32 {{.*}}, ptr getelementptr {{.*}} @_ZN7Dynamic2d1E, i32 0, i32 2
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN7Dynamic1DD1Ev, {{.*}} @_ZN7Dynamic2d1E
  // CHECK: ret
  //
  //   UNWIND:
  // CHECK: call void @_ZN7Dynamic1BD1Ev(ptr {{.*}}getelementptr inbounds {{.*}} @_ZN7Dynamic2d1E{{.*}}, i64 8
  // CHECK: call void @_ZN7Dynamic1AD1Ev({{.*}} @_ZN7Dynamic2d1E

  D d2 = {1, 2, false};
  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: call void @_ZN7Dynamic1AC1Ei({{.*}} @_ZN7Dynamic2d2E{{.*}}, i32 noundef 1)
  // CHECK: store i32 2, {{.*}}ptr getelementptr inbounds {{.*}}@_ZN7Dynamic2d2E{{.*}}, i64 8
  // CHECK: invoke void @_ZN7Dynamic1CC1Eb({{.*}} @_ZN7Dynamic2d2E{{.*}}, i1 noundef zeroext false)
  // CHECK: invoke noundef i32 @_ZN7Dynamic1fEv()
  // CHECK: store i32 {{.*}}, ptr getelementptr {{.*}} @_ZN7Dynamic2d2E, i32 0, i32 2
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN7Dynamic1DD1Ev, {{.*}} @_ZN7Dynamic2d2E
  // CHECK: ret void

  D d3 = {g(), h(), {}, i()};
  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: %[[G_CALL:.*]] = call noundef i32 @_ZN7Dynamic1gEv()
  // CHECK: call void @_ZN7Dynamic1AC1Ei({{.*}} @_ZN7Dynamic2d3E{{.*}}, i32 noundef %[[G_CALL]])
  // CHECK: %[[H_CALL:.*]] = invoke noundef i32 @_ZN7Dynamic1hEv()
  // CHECK:   unwind label %[[DESTROY_A_LPAD:.*]]
  // CHECK: store i32 %[[H_CALL]], {{.*}}ptr getelementptr inbounds {{.*}} @_ZN7Dynamic2d3E{{.*}}, i64 8
  // CHECK: invoke void @_ZN7Dynamic1CC2Eb({{.*}} @_ZN7Dynamic2d3E{{.*}}, i1 noundef zeroext true)
  // CHECK:   unwind label %[[DESTROY_AB_LPAD:.*]]
  // CHECK: %[[I_CALL:.*]] = invoke noundef i32 @_ZN7Dynamic1iEv()
  // CHECK:   unwind label %[[DESTROY_AB_LPAD:.*]]
  // CHECK: store i32 %[[I_CALL]], ptr getelementptr {{.*}} @_ZN7Dynamic2d3E, i32 0, i32 2
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN7Dynamic1DD1Ev, {{.*}} @_ZN7Dynamic2d3E
  // CHECK: ret
  //
  //   DESTROY_A_LPAD:
  // CHECK: br label %[[A_CLEANUP:.*]]
  //
  //   DESTROY_B_LPAD:
  // CHECK: call void @_ZN7Dynamic1BD1Ev(ptr {{.*}}getelementptr inbounds {{.*}} @_ZN7Dynamic2d3E{{.*}}, i64 8
  // CHECK: br label %[[A_CLEANUP:.*]]
  //
  //   A_CLEANUP:
  // CHECK: call void @_ZN7Dynamic1AD1Ev({{.*}} @_ZN7Dynamic2d3E
}

namespace Instantiated1 {
  struct A { A(); };
  struct B : A { using A::A; };
  template<int> B v({});
  template B v<0>;
  // CHECK-LABEL: define {{.*}}global_var_init{{.*}} comdat($_ZN13Instantiated11vILi0EEE) {
  // CHECK: call void @_ZN13Instantiated11BC1Ev(ptr {{[^,]*}} @_ZN13Instantiated11vILi0EEE)
}

namespace Instantiated2 {
  struct A { A(); };
  struct B : A {};
  template<int> B v({});
  template B v<0>;
  // CHECK-LABEL: define {{.*}}global_var_init{{.*}} comdat($_ZN13Instantiated21vILi0EEE) {
  // CHECK: call void @_ZN13Instantiated21AC2Ev(
}
