// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc18.0.0 -fcoroutines-ts -emit-llvm %s -o - -std=c++14 -disable-llvm-passes | FileCheck %s
struct no_suspend {
  bool await_ready() { return true; }
  template <typename F> void await_suspend(F) {}
  void await_resume() {}
};

struct A {
  no_suspend operator co_await() { return {}; }
};

struct B {};

no_suspend operator co_await(B const&) { return {}; }

// CHECK-LABEL: f(
extern "C" void f() {
  A a;
  B b;
  // CHECK: call void @"??__LA@@QEAA?AUno_suspend@@XZ"(
  a.operator co_await();
  // CHECK-NEXT: call i8 @"??__L@YA?AUno_suspend@@AEBUB@@@Z"(
  operator co_await(b);
}

