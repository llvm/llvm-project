// Tests that a coro_await_elidable coroutine with a suspend_never final suspend
// suppresses only the deallocation for a resumed elided callee.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 \
// RUN:   -mllvm -coro-elide-branch-ratio=0 -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct gate {
  std::coroutine_handle<> waiter = nullptr;
  bool open = false;

  struct awaiter {
    gate &g;
    bool await_ready() noexcept { return g.open; }
    void await_suspend(std::coroutine_handle<> h) noexcept { g.waiter = h; }
    void await_resume() noexcept {}
  };

  awaiter operator co_await() noexcept { return {*this}; }
};

struct [[clang::coro_await_elidable]] task {
  struct promise_type {
    std::coroutine_handle<> continuation = nullptr;

    task get_return_object() noexcept {
      return {std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }

    void return_void() noexcept {
      if (continuation)
        continuation.resume();
    }

    void unhandled_exception() noexcept { __builtin_abort(); }
  };

  std::coroutine_handle<promise_type> handle;

  bool await_ready() noexcept { return false; }

  void await_suspend(std::coroutine_handle<> h) noexcept {
    handle.promise().continuation = h;
  }

  void await_resume() noexcept {}
};

task callee(gate &g, int &value) {
  co_await g;
  value = 42;
}

task caller(gate &g, int &value, bool &finished) {
  co_await callee(g, value);
  finished = true;
}

// CHECK-LABEL: define internal void @_Z6calleeR4gateRi.resume(
// CHECK:         %[[DESTROY_ADDR:.+]] = getelementptr inbounds{{.*}} i8, ptr %{{.+}}, i64 8
// CHECK-NEXT:    %[[DESTROY:.+]] = load ptr, ptr %[[DESTROY_ADDR]]
// CHECK-NEXT:    %[[IS_ELIDED:.+]] = icmp eq ptr %[[DESTROY]], @_Z6calleeR4gateRi.cleanup
// CHECK:         store i32 42,
// CHECK:         br i1 %[[IS_ELIDED]], label %[[CORO_END:.+]], label %[[CORO_FREE:.+]]
// CHECK:       [[CORO_FREE]]:
// CHECK-NEXT:    tail call void @_Zdl
// CHECK:       [[CORO_END]]:
// CHECK-NEXT:    ret void

// CHECK-LABEL: define internal {{.*}}void @_Z6calleeR4gateRi.destroy(
// CHECK:         call void @_Zdl

// CHECK-LABEL: define internal {{.*}}void @_Z6calleeR4gateRi.cleanup(
// CHECK-NOT:     call void @_Zdl
// CHECK:         ret void
