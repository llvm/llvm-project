// RUN: %clang_cc1 -std=c++1z -fcoroutines-ts -triple=x86_64-pc-windows-msvc18.0.0 -emit-llvm %s -o - -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -std=c++1z -fcoroutines-ts -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck --check-prefix=CHECK-LPAD %s

namespace std::experimental {
template <typename R, typename... T> struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

template <class Promise = void> struct coroutine_handle;

template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) noexcept;
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};
template <class Promise> struct coroutine_handle : coroutine_handle<void> {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
};
} // namespace std::experimental

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct coro_t {
  struct promise_type {
    coro_t get_return_object() noexcept;
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

struct Cleanup {
  ~Cleanup();
};
void may_throw();

coro_t f() {
  Cleanup x;
  may_throw();
  co_return;
}

// CHECK: @"?f@@YA?AUcoro_t@@XZ"(
// CHECK:   invoke void @"?may_throw@@YAXXZ"()
// CHECK:       to label %[[CONT:.+]] unwind label %[[EHCLEANUP:.+]]
// CHECK: [[EHCLEANUP]]:
// CHECK:   %[[INNERPAD:.+]] = cleanuppad within none []
// CHECK:   call void @"??1Cleanup@@QEAA@XZ"(
// CHECK:   cleanupret from %{{.+}} unwind label %[[CATCHDISPATCH:.+]]

// CHECK: [[CATCHDISPATCH]]:
// CHECK:   catchswitch within none [label %[[CATCHPAD:.+]]] unwind label %[[COROENDBB:.+]]
// CHECK: [[CATCHPAD]]:
// CHECK:   call void @"?unhandled_exception@promise_type@coro_t@@QEAAXXZ"

// CHECK: [[COROENDBB]]:
// CHECK-NEXT: %[[CLPAD:.+]] = cleanuppad within none
// CHECK-NEXT: call i1 @llvm.coro.end(ptr null, i1 true) [ "funclet"(token %[[CLPAD]]) ]
// CHECK-NEXT: cleanupret from %[[CLPAD]] unwind label

// CHECK-LPAD: @_Z1fv(
// CHECK-LPAD:   invoke void @_Z9may_throwv()
// CHECK-LPAD:       to label %[[CONT:.+]] unwind label %[[EHCLEANUP:.+]]
// CHECK-LPAD: [[EHCLEANUP]]:
// CHECK-LPAD:    landingpad { ptr, i32 }
// CHECK-LPAD:          catch
// CHECK-LPAD:   call void @_ZN7CleanupD1Ev(
// CHECK-LPAD:   call ptr @__cxa_begin_catch
// CHECK-LPAD:   call void @_ZN6coro_t12promise_type19unhandled_exceptionEv
// CHECK-LPAD:   invoke void @__cxa_end_catch()
// CHECK-LPAD:             to label %{{.+}} unwind label %[[UNWINDBB:.+]]

// CHECK-LPAD: [[UNWINDBB]]:
// CHECK-LPAD:   %[[I1RESUME:.+]] = call i1 @llvm.coro.end(ptr null, i1 true)
// CHECK-LPAD:   br i1  %[[I1RESUME]], label %[[EHRESUME:.+]], label
// CHECK-LPAD: [[EHRESUME]]:
// CHECK-LPAD-NEXT:  %[[exn:.+]] = load ptr, ptr %exn.slot, align 8
// CHECK-LPAD-NEXT:  %[[sel:.+]] = load i32, ptr %ehselector.slot, align 4
// CHECK-LPAD-NEXT:  %[[val1:.+]] = insertvalue { ptr, i32 } poison, ptr %[[exn]], 0
// CHECK-LPAD-NEXT:  %[[val2:.+]] = insertvalue { ptr, i32 } %[[val1]], i32 %[[sel]], 1
// CHECK-LPAD-NEXT:  resume { ptr, i32 } %[[val2]]
