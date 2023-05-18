// RUN: %clang_cc1 -std=c++1y -triple x86_64-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s

struct S {
  S();
  ~S();
};

struct T {
  T() noexcept;
  ~T();
  int n;
};

// CHECK-LABEL: define{{.*}} void @_Z1fv(
void f() {
  // CHECK: call void @_ZN1SC1Ev(
  // CHECK: invoke void @__cxa_throw
  //
  // Ensure we call the lambda destructor here, and do not call the destructor
  // for the capture.
  // CHECK: landingpad
  // CHECK-NOT: _ZN1SD
  // CHECK: call void @"_ZZ1fvEN3$_0D1Ev"(
  // CHECK-NOT: _ZN1SD
  // CHECK: resume
  [s = S()] {}, throw 0;

  // CHECK: }
}

// CHECK-LABEL: define{{.*}} void @_Z1gv(
void g() {
  // CHECK: call void @_ZN1SC1Ev(
  // CHECK: invoke void @__cxa_throw
  //
  // Ensure we call the lambda destructor here, and do not call the destructor
  // for the capture.
  // CHECK: landingpad
  // CHECK-NOT: @"_ZZ1gvEN3$_0D1Ev"(
  // CHECK: call void @_ZN1SD1Ev(
  // CHECK-NOT: @"_ZZ1gvEN3$_0D1Ev"(
  // CHECK: resume
  [s = S(), t = (throw 0, 1)] {};

  // CHECK: }
}

void x() noexcept;
void y() noexcept;

// CHECK-LABEL: define{{.*}} void @_Z1hbb(
void h(bool b1, bool b2) {
  // CHECK: {{.*}} = alloca i1,
  // CHECK: %[[S_ISACTIVE:.*]] = alloca i1,
  // CHECK: {{.*}} = alloca i1,

  // lambda init: s and t, branch on b1
  // CHECK: call void @_ZN1SC1Ev(
  // CHECK: store i1 true, ptr %[[S_ISACTIVE]], align 1
  // CHECK: call void @_ZN1TC1Ev(
  // CHECK: br i1

  // throw 1
  // CHECK: invoke void @__cxa_throw

  // completion of lambda init, branch on b2
  // CHECK: store i32 42,
  // CHECK: store i1 false, ptr %[[S_ISACTIVE]], align 1
  // CHECK: br i1

  // throw 2
  // CHECK: invoke void @__cxa_throw

  // end of full-expression
  // CHECK: call void @_Z1xv(
  // CHECK: call void @"_ZZ1hbbEN3$_0D1Ev"(
  // CHECK: call void @_ZN1TD1Ev(
  // CHECK: call void @_Z1yv(
  // CHECK: ret void

  // cleanups for throw 1
  // CHECK: landingpad
  // CHECK-NOT: @"_ZZ1hbbEN3$_0D1Ev"(
  // CHECK: br

  // cleanups for throw 2
  // CHECK: landingpad
  // CHECK: call void @"_ZZ1hbbEN3$_0D1Ev"(
  // CHECK: br

  // common cleanup code
  // CHECK: call void @_ZN1TD1Ev(
  // CHECK: load i1, ptr %[[S_ISACTIVE]],
  // CHECK: br i1

  // CHECK: call void @_ZN1SD1Ev(
  // CHECK: br

  // CHECK: resume
  [s = S(), t = T().n, u = (b1 ? throw 1 : 42)] {}, (b2 ? throw 2 : 0), x();
  y();

  // CHECK: }
}
