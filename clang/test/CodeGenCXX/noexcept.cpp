// RUN: %clang_cc1 -no-opaque-pointers %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions -std=c++11 | FileCheck %s

// rdar://11904428
//   Ensure that we call __cxa_begin_catch before calling
//   std::terminate in a noexcept function.
namespace test0 {
  void foo();

  struct A {
    A();
    ~A();
  };

  void test() noexcept {
    A a;
    foo();
  }
}

// CHECK-LABEL:    define{{.*}} void @_ZN5test04testEv()
//   This goes to the terminate lpad.
// CHECK:      invoke void @_ZN5test01AC1Ev(
// CHECK-NEXT:   unwind label %[[TERMINATE_LPAD:.*]]
//   This also goes to the terminate lpad (no cleanups!).
// CHECK:      invoke void @_ZN5test03fooEv()
// CHECK-NEXT:   unwind label %[[TERMINATE_LPAD]]
//   Destructors don't throw by default in C++11.
// CHECK:      call void @_ZN5test01AD1Ev(
//   Cleanup lpad.
// CHECK: [[TERMINATE_LPAD]]:
// CHECK-NEXT: [[T0:%.*]] = landingpad
// CHECK-NEXT:   catch i8* null
// CHECK-NEXT: [[T1:%.*]] = extractvalue { i8*, i32 } [[T0]], 0
// CHECK-NEXT: call void @__clang_call_terminate(i8* [[T1]])
// CHECK-NEXT: unreachable

// CHECK-LABEL:  define linkonce_odr hidden void @__clang_call_terminate(
// CHECK:      call i8* @__cxa_begin_catch(
// CHECK-NEXT: call void @_ZSt9terminatev()
// CHECK-NEXT: unreachable
