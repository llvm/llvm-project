// Test the behavior of http://wg21.link/P0664, a proposal to catch any
// exceptions thrown after the initial suspend point of a coroutine by
// executing the handler specified by the promise type's 'unhandled_exception'
// member function.
//
// RUN: %clang_cc1 -std=c++20 \
// RUN:   -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -fexceptions -fcxx-exceptions -disable-llvm-passes \
// RUN:   | FileCheck %s

#include "Inputs/coroutine.h"

struct throwing_awaitable {
  bool await_ready() { return true; }
  void await_suspend(std::coroutine_handle<>) {}
  void await_resume() { throw 42; }
};

struct throwing_task {
  struct promise_type {
    auto get_return_object() { return throwing_task{}; }
    auto initial_suspend() { return throwing_awaitable{}; }
    auto final_suspend() noexcept { return std::suspend_never{}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

// CHECK-LABEL: define{{.*}} void @_Z1fv()
throwing_task f() {
  // A variable RESUMETHREW is used to keep track of whether the body
  // of 'await_resume' threw an exception. Exceptions thrown in
  // 'await_resume' are unwound to RESUMELPAD.
  // CHECK: init.ready:
  // CHECK-NEXT: store i1 true, i1* %[[RESUMETHREW:.+]], align 1
  // CHECK-NEXT: invoke void @_ZN18throwing_awaitable12await_resumeEv
  // CHECK-NEXT: to label %[[RESUMECONT:.+]] unwind label %[[RESUMELPAD:.+]]

  // If 'await_resume' does not throw an exception, 'false' is stored in
  // variable RESUMETHREW.
  // CHECK: [[RESUMECONT]]:
  // CHECK-NEXT: store i1 false, i1* %[[RESUMETHREW]]
  // CHECK-NEXT: br label %[[RESUMETRYCONT:.+]]

  // 'unhandled_exception' is called for the exception thrown in
  // 'await_resume'. The variable RESUMETHREW is never set to false,
  // and a jump is made to RESUMETRYCONT.
  // CHECK: [[RESUMELPAD]]:
  // CHECK: br label %[[RESUMECATCH:.+]]
  // CHECK: [[RESUMECATCH]]:
  // CHECK: invoke void @_ZN13throwing_task12promise_type19unhandled_exceptionEv
  // CHECK-NEXT: to label %[[RESUMEENDCATCH:.+]] unwind label
  // CHECK: [[RESUMEENDCATCH]]:
  // CHECK-NEXT: invoke void @__cxa_end_catch()
  // CHECK-NEXT: to label %[[RESUMEENDCATCHCONT:.+]] unwind label
  // CHECK: [[RESUMEENDCATCHCONT]]:
  // CHECK-NEXT: br label %[[RESUMETRYCONT]]
  // CHECK: [[RESUMETRYCONT]]:
  // CHECK-NEXT: br label %[[CLEANUP:.+]]
  // CHECK: [[CLEANUP]]:
  // CHECK: switch i32 %{{.+}}, label %{{.+}} [
  // CHECK-NEXT: i32 0, label %[[CLEANUPCONT:.+]]
  // CHECK-NEXT: ]

  // The variable RESUMETHREW is loaded and if true, then 'await_resume'
  // threw an exception and the coroutine body is skipped, and the final
  // suspend is executed immediately. Otherwise, the coroutine body is
  // executed, and then the final suspend.
  // CHECK: [[CLEANUPCONT]]:
  // CHECK-NEXT: %[[RESUMETHREWLOAD:.+]] = load i1, i1* %[[RESUMETHREW]]
  // CHECK-NEXT: br i1 %[[RESUMETHREWLOAD]], label %[[RESUMEDCONT:.+]], label %[[RESUMEDBODY:.+]]

  // CHECK: [[RESUMEDBODY]]:
  // CHECK: invoke void @_ZN13throwing_task12promise_type11return_voidEv
  // CHECK-NEXT: to label %[[REDUMEDBODYCONT:.+]] unwind label
  // CHECK: [[REDUMEDBODYCONT]]:
  // CHECK-NEXT: br label %[[COROFINAL:.+]]

  // CHECK: [[RESUMEDCONT]]:
  // CHECK-NEXT: br label %[[COROFINAL]]

  // CHECK: [[COROFINAL]]:
  // CHECK: call void @_ZN13throwing_task12promise_type13final_suspendEv
  co_return;
}

struct noexcept_awaitable {
  bool await_ready() { return true; }
  void await_suspend(std::coroutine_handle<>) {}
  void await_resume() noexcept {}
};

struct noexcept_task {
  struct promise_type {
    auto get_return_object() { return noexcept_task{}; }
    auto initial_suspend() { return noexcept_awaitable{}; }
    auto final_suspend() noexcept { return std::suspend_never{}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

// CHECK-LABEL: define{{.*}} void @_Z1gv()
noexcept_task g() {
  // If the await_resume function is marked as noexcept, none of the additional
  // conditions that are present in f() above are added to the IR.
  // This means that no i1 are stored before or after calling await_resume:
  // CHECK: init.ready:
  // CHECK-NEXT: call void @_ZN18noexcept_awaitable12await_resumeEv
  // CHECK-NOT: store i1 false, i1*
  co_return;
}
