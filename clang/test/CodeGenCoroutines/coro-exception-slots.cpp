// RUN: %clang_cc1 -std=c++20 -triple arm64-apple-macos -fcxx-exceptions \
// RUN:            -fexceptions -I %S/Inputs -disable-llvm-passes -emit-llvm \
// RUN:            -o - %s | FileCheck %s


#include "coroutine.h"

struct Task {
  struct promise_type {
    promise_type();
    Task get_return_object();
    struct InitialAwaiter {
      bool await_ready();
      void await_suspend(std::coroutine_handle<>);
      void await_resume();
    };
    InitialAwaiter initial_suspend();
    std::suspend_always final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
};

Task coro() {
  co_await std::suspend_always{};
}
// CHECK-LABEL: define {{.*}} @_Z4corov(
// CHECK: %exn.slot = alloca ptr, align 8, !coro.outside.frame
// CHECK: %ehselector.slot = alloca i32, align 4, !coro.outside.frame
