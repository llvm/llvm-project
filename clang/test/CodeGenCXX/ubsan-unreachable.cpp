// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=unreachable | FileCheck %s

extern void __attribute__((noreturn)) abort();

// CHECK-LABEL: define void @_Z14calls_noreturnv
void calls_noreturn() {
  abort();

  // Check that there are no attributes on the call site.
  // CHECK-NOT: call void @_Z5abortv{{.*}}#

  // CHECK: __ubsan_handle_builtin_unreachable
  // CHECK: unreachable
}

struct A {
  // CHECK: declare void @_Z5abortv{{.*}} [[ABORT_ATTR:#[0-9]+]]

  // CHECK-LABEL: define linkonce_odr void @_ZN1A5call1Ev
  void call1() {
    // CHECK-NOT: call void @_ZN1A16does_not_return2Ev{{.*}}#
    does_not_return2();

    // CHECK: __ubsan_handle_builtin_unreachable
    // CHECK: unreachable
  }

  // Test static members.
  static void __attribute__((noreturn)) does_not_return1() {
    // CHECK-NOT: call void @_Z5abortv{{.*}}#
    abort();
  }

  // CHECK-LABEL: define linkonce_odr void @_ZN1A5call2Ev
  void call2() {
    // CHECK-NOT: call void @_ZN1A16does_not_return1Ev{{.*}}#
    does_not_return1();

    // CHECK: __ubsan_handle_builtin_unreachable
    // CHECK: unreachable
  }

  // Test calls through pointers to non-static member functions.
  typedef void __attribute__((noreturn)) (A::*MemFn)();

  // CHECK-LABEL: define linkonce_odr void @_ZN1A5call3Ev
  void call3() {
    MemFn MF = &A::does_not_return2;
    (this->*MF)();

    // CHECK-NOT: call void %{{.*}}#
    // CHECK: __ubsan_handle_builtin_unreachable
    // CHECK: unreachable
  }

  // Test regular members.
  // CHECK-LABEL: define linkonce_odr void @_ZN1A16does_not_return2Ev({{.*}})
  // CHECK-SAME: [[DOES_NOT_RETURN_ATTR:#[0-9]+]]
  void __attribute__((noreturn)) does_not_return2() {
    // CHECK-NOT: call void @_Z5abortv(){{.*}}#
    abort();

    // CHECK: call void @__ubsan_handle_builtin_unreachable
    // CHECK: unreachable

    // CHECK: call void @__ubsan_handle_builtin_unreachable
    // CHECK: unreachable
  }
};

// CHECK: define linkonce_odr void @_ZN1A16does_not_return1Ev() [[DOES_NOT_RETURN_ATTR]]

void force_irgen() {
  A a;
  a.call1();
  a.call2();
  a.call3();
}

// CHECK-NOT: [[ABORT_ATTR]] = {{[^}]+}}noreturn
// CHECK-NOT: [[DOES_NOT_RETURN_ATTR]] = {{[^}]+}}noreturn
