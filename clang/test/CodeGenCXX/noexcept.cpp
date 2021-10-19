// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions -std=c++11 | FileCheck %s

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
// CHECK:      [[EXN:%.*]] = alloca i8*
//   This goes to the terminate lpad.
// CHECK:      invoke void @_ZN5test01AC1Ev(
//   This goes to the cleanup-and-then-terminate lpad.
// CHECK:      invoke void @_ZN5test03fooEv()
//   Destructors don't throw by default in C++11.
// CHECK:      call void @_ZN5test01AD1Ev(
//   Cleanup lpad.
// CHECK:      [[T0:%.*]] = landingpad
// CHECK-NEXT:   catch i8* null
// CHECK-NEXT: [[T1:%.*]] = extractvalue { i8*, i32 } [[T0]], 0
// CHECK-NEXT: store i8* [[T1]], i8** [[EXN]]
//   (Calling this destructor is not technically required.)
// CHECK:      call void @_ZN5test01AD1Ev(
// CHECK-NEXT: br label
//   The terminate landing pad jumps in here for some reason.
// CHECK:      [[T0:%.*]] = landingpad
// CHECK-NEXT:   catch i8* null
// CHECK-NEXT: [[T1:%.*]] = extractvalue { i8*, i32 } [[T0]], 0
// CHECK-NEXT: call void @__clang_call_terminate(i8* [[T1]])
// CHECK-NEXT: unreachable
//   The terminate handler chained to by the cleanup lpad.
// CHECK:      [[T0:%.*]] = load i8*, i8** [[EXN]]
// CHECK-NEXT: call void @__clang_call_terminate(i8* [[T0]])
// CHECK-NEXT: unreachable

// CHECK-LABEL:  define linkonce_odr hidden void @__clang_call_terminate(
// CHECK:      call i8* @__cxa_begin_catch(
// CHECK-NEXT: call void @_ZSt9terminatev()
// CHECK-NEXT: unreachable
