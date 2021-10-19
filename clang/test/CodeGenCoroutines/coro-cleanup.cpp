// Verify that coroutine promise and allocated memory are freed up on exception.
// RUN: %clang_cc1 -std=c++1z -fcoroutines-ts -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s

namespace std::experimental {
template <typename... T> struct coroutine_traits;

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
};
template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) noexcept;
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};
}

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

template <> struct std::experimental::coroutine_traits<void> {
  struct promise_type {
    void get_return_object() noexcept;
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    promise_type();
    ~promise_type();
    void unhandled_exception() noexcept;
  };
};

struct Cleanup { ~Cleanup(); };
void may_throw();

// CHECK-LABEL: define{{.*}} void @_Z1fv(
void f() {
  // CHECK: call noalias nonnull i8* @_Znwm(i64

  // If promise constructor throws, check that we free the memory.

  // CHECK: invoke void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_typeC1Ev(
  // CHECK-NEXT: to label %{{.+}} unwind label %[[DeallocPad:.+]]

  // CHECK: [[DeallocPad]]:
  // CHECK-NEXT: landingpad
  // CHECK-NEXT:   cleanup
  // CHECK: br label %[[Dealloc:.+]]

  Cleanup cleanup;
  may_throw();

  // if may_throw throws, check that we destroy the promise and free the memory.

  // CHECK: invoke void @_Z9may_throwv(
  // CHECK-NEXT: to label %{{.+}} unwind label %[[CatchPad:.+]]

  // CHECK: [[CatchPad]]:
  // CHECK-NEXT:  landingpad
  // CHECK-NEXT:       catch i8* null
  // CHECK:  call void @_ZN7CleanupD1Ev(
  // CHECK:  br label %[[Catch:.+]]

  // CHECK: [[Catch]]:
  // CHECK:    call i8* @__cxa_begin_catch(
  // CHECK:    call void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_type19unhandled_exceptionEv(
  // CHECK:    invoke void @__cxa_end_catch()
  // CHECK-NEXT:    to label %[[Cont:.+]] unwind

  // CHECK: [[Cont]]:
  // CHECK-NEXT: br label %[[Cont2:.+]]
  // CHECK: [[Cont2]]:
  // CHECK-NEXT: br label %[[Cleanup:.+]]

  // CHECK: [[Cleanup]]:
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_typeD1Ev(
  // CHECK: %[[Mem0:.+]] = call i8* @llvm.coro.free(
  // CHECK: call void @_ZdlPv(i8* %[[Mem0]]

  // CHECK: [[Dealloc]]:
  // CHECK:   %[[Mem:.+]] = call i8* @llvm.coro.free(
  // CHECK:   call void @_ZdlPv(i8* %[[Mem]])

  co_return;
}

// CHECK-LABEL: define{{.*}} void @_Z1gv(
void g() {
  for (;;)
    co_await suspend_always{};
  // Since this is the endless loop there should be no fallthrough handler (call to 'return_void').
  // CHECK-NOT: call void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_type11return_voidEv
}
