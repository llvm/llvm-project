// Verify that coroutine promise and allocated memory are freed up on exception.
// RUN: %clang_cc1 -std=c++1z -fcoroutines-ts -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s

namespace std::experimental {
template <typename... T> struct coroutine_traits;

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) { return {}; }
};
template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) { return {}; }
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) {}
};
}

struct suspend_always {
  bool await_ready();
  void await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};

template <> struct std::experimental::coroutine_traits<void> {
  struct promise_type {
    void get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend();
    void return_void();
    promise_type();
    ~promise_type();
    void unhandled_exception();
  };
};

struct Cleanup { ~Cleanup(); };
void may_throw();

// CHECK: define void @_Z1fv(
void f() {
  // CHECK: call i8* @_Znwm(i64

  // If promise constructor throws, check that we free the memory.

  // CHECK: invoke void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_typeC1Ev(
  // CHECK-NEXT: to label %{{.+}} unwind label %[[DeallocPad:.+]]

  Cleanup cleanup;
  may_throw();

  // if may_throw throws, check that we destroy the promise and free the memory.

  // CHECK: invoke void @_Z9may_throwv(
  // CHECK-NEXT: to label %{{.+}} unwind label %[[PromDtorPad:.+]]

  // CHECK: [[DeallocPad]]:
  // CHECK-NEXT: landingpad
  // CHECK-NEXT:   cleanup
  // CHECK: br label %[[Dealloc:.+]]

  // CHECK: [[PromDtorPad]]:
  // CHECK-NEXT: landingpad
  // CHECK-NEXT:   cleanup
  // CHECK: call void @_ZN7CleanupD1Ev(%struct.Cleanup*
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_typeD1Ev(
  // CHECK: br label %[[Dealloc]]

  // CHECK: [[Dealloc]]:
  // CHECK:   %[[Mem:.+]] = call i8* @llvm.coro.free(
  // CHECK:   call void @_ZdlPv(i8* %[[Mem]])

  co_return;
}
