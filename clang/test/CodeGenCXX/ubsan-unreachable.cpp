// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=unreachable | FileCheck %s

extern void __attribute__((noreturn)) abort();

// CHECK-LABEL: define void @_Z14calls_noreturnv()
void calls_noreturn() {
  // CHECK: call void @_Z5abortv() [[CALL_SITE_ATTR:#[0-9]+]]
  abort();

  // CHECK: __ubsan_handle_builtin_unreachable
  // CHECK: unreachable
}

struct A {
  // CHECK: declare void @_Z5abortv() [[EXTERN_FN_ATTR:#[0-9]+]]

  // CHECK-LABEL: define linkonce_odr void @_ZN1A5call1Ev
  void call1() {
    // CHECK: call void @_ZN1A16does_not_return2Ev({{.*}}) [[CALL_SITE_ATTR]]
    does_not_return2();

    // CHECK: __ubsan_handle_builtin_unreachable
    // CHECK: unreachable
  }

  // Test static members. Checks are below after `struct A` scope ends.
  static void __attribute__((noreturn)) does_not_return1() {
    abort();
  }

  // CHECK-LABEL: define linkonce_odr void @_ZN1A5call2Ev
  void call2() {
    // CHECK: call void @_ZN1A16does_not_return1Ev() [[CALL_SITE_ATTR]]
    does_not_return1();

    // CHECK: __ubsan_handle_builtin_unreachable
    // CHECK: unreachable
  }

  // Test calls through pointers to non-static member functions.
  typedef void __attribute__((noreturn)) (A::*MemFn)();

  // CHECK-LABEL: define linkonce_odr void @_ZN1A5call3Ev
  void call3() {
    MemFn MF = &A::does_not_return2;
    // CHECK: call void %{{[0-9]+\(.*}}) [[CALL_SITE_ATTR]]
    (this->*MF)();

    // CHECK: __ubsan_handle_builtin_unreachable
    // CHECK: unreachable
  }

  // Test regular members.
  // CHECK-LABEL: define linkonce_odr void @_ZN1A16does_not_return2Ev({{.*}})
  // CHECK-SAME: [[USER_FN_ATTR:#[0-9]+]]
  void __attribute__((noreturn)) does_not_return2() {
    // CHECK: call void @_Z5abortv() [[CALL_SITE_ATTR]]
    abort();

    // CHECK: call void @__ubsan_handle_builtin_unreachable
    // CHECK: unreachable

    // CHECK: call void @__ubsan_handle_builtin_unreachable
    // CHECK: unreachable
  }
};

// CHECK-LABEL: define linkonce_odr void @_ZN1A16does_not_return1Ev()
// CHECK-SAME: [[USER_FN_ATTR]]
// CHECK: call void @_Z5abortv() [[CALL_SITE_ATTR]]

void force_irgen() {
  A a;
  a.call1();
  a.call2();
  a.call3();
}

// 1) 'noreturn' should be removed from functions and call sites
// 2) 'expect_noreturn' added to call sites
// CHECK-LABEL: attributes
// CHECK: [[USER_FN_ATTR]] = { {{.*[^noreturn].*}} }
// CHECK: [[EXTERN_FN_ATTR]] = { {{.*[^noreturn].*}} }
// CHECK: [[CALL_SITE_ATTR]] = { expect_noreturn }
