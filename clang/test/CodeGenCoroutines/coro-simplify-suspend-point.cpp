// Test that we can perform suspend point simplification
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -std=c++20 -O1 -emit-llvm -c  %s -o %t && %clang -c %t

#include "Inputs/coroutine.h"

struct detached_task {
  struct promise_type {
    detached_task get_return_object() noexcept {
      return detached_task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    void return_void() noexcept {}

    struct final_awaiter {
      bool await_ready() noexcept { return false; }
      std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
        h.destroy();
        return std::noop_coroutine();
      }
      void await_resume() noexcept {}
    };

    void unhandled_exception() noexcept {}

    final_awaiter final_suspend() noexcept { return {}; }

    std::suspend_always initial_suspend() noexcept { return {}; }
  };

  ~detached_task() {
    if (coro_) {
      coro_.destroy();
      coro_ = {};
    }
  }

  void start() && {
    auto tmp = coro_;
    coro_ = {};
    tmp.resume();
  }

  std::coroutine_handle<promise_type> coro_;
};

class SelfResumeAwaiter final
{
public:
    bool await_ready() noexcept { return false; }
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> h) { 
      return h; 
    }
    void await_resume() noexcept {}
};

// Check that there is only one call left: coroutine destroy
// CHECK-LABEL: define{{.*}}void @_Z3foov.resume
// CHECK-NOT: call{{.*}}
// CHECK: tail call{{.*}}void %{{[0-9+]}}(
// CHECK-NOT: call{{.*}}
// CHECK: define
detached_task foo() {
  co_await SelfResumeAwaiter{};
  co_return;
}
