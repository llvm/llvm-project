// Verify that coroutine promise and allocated memory are freed up on exception.
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s --check-prefixes=CHECK,THROWEND
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexceptions -fcxx-exceptions -fassume-nothrow-exception-dtor -disable-llvm-passes | FileCheck %s --check-prefixes=CHECK,NOTHROWEND

namespace std {
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
} // namespace std

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

template <> struct std::coroutine_traits<void> {
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
  // CHECK: call noalias noundef nonnull ptr @_Znwm(i64

  // If promise constructor throws, check that we free the memory.

  // CHECK: invoke void @_ZNSt16coroutine_traitsIJvEE12promise_typeC1Ev(
  // CHECK-NEXT: to label %{{.+}} unwind label %[[DeallocPad:.+]]

  // CHECK: [[DeallocPad]]:
  // CHECK-NEXT: landingpad
  // CHECK-NEXT:   cleanup
  // THROWEND:        br label %[[Dealloc:.+]]
  // NOTHROWEND:      icmp ne ptr %[[#]], null
  // NOTHROWEND-NEXT: br i1 %[[#]], label %[[Dealloc:.+]], label

  Cleanup cleanup;
  may_throw();

  // if may_throw throws, check that we destroy the promise and free the memory.

  // CHECK: invoke void @_Z9may_throwv(
  // CHECK-NEXT: to label %{{.+}} unwind label %[[CatchPad:.+]]

  // CHECK: [[CatchPad]]:
  // CHECK-NEXT:  landingpad
  // CHECK-NEXT:       catch ptr null
  // CHECK:  call void @_ZN7CleanupD1Ev(
  // CHECK:  br label %[[Catch:.+]]

  // CHECK: [[Catch]]:
  // CHECK:    call ptr @__cxa_begin_catch(
  // CHECK:    call void @_ZNSt16coroutine_traitsIJvEE12promise_type19unhandled_exceptionEv(
  // THROWEND:        invoke void @__cxa_end_catch()
  // THROWEND-NEXT:     to label %[[Cont:.+]] unwind
  // NOTHROWEND:      call void @__cxa_end_catch()
  // NOTHROWEND-NEXT:   br label %[[Cont2:.+]]

  // THROWEND:      [[Cont]]:
  // THROWEND-NEXT:   br label %[[Cont2:.+]]
  // CHECK:         [[Cont2]]:
  // CHECK-NEXT:      br label %[[Cleanup:.+]]

  // CHECK: [[Cleanup]]:
  // CHECK: call void @_ZNSt16coroutine_traitsIJvEE12promise_typeD1Ev(
  // CHECK: %[[Mem0:.+]] = call ptr @llvm.coro.free(
  // CHECK: call void @_ZdlPv(ptr noundef %[[Mem0]]

  // CHECK: [[Dealloc]]:
  // THROWEND:   %[[Mem:.+]] = call ptr @llvm.coro.free(
  // THROWEND:   call void @_ZdlPv(ptr noundef %[[Mem]])

  co_return;
}

// CHECK-LABEL: define{{.*}} void @_Z1gv(
void g() {
  for (;;)
    co_await suspend_always{};
  // Since this is the endless loop there should be no fallthrough handler (call to 'return_void').
  // CHECK-NOT: call void @_ZNSt16coroutine_traitsIJvEE12promise_type11return_voidEv
}
