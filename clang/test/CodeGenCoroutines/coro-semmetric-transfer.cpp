// RUN: %clang -std=c++14 -fcoroutines-ts -emit-llvm -S -O1 %s -o -

#include "Inputs/coroutine.h"

namespace coro = std::experimental::coroutines_v1;

struct detached_task {
  struct promise_type {
    detached_task get_return_object() noexcept {
      return detached_task{coro::coroutine_handle<promise_type>::from_promise(*this)};
    }

    void return_void() noexcept {}

    struct final_awaiter {
      bool await_ready() noexcept { return false; }
      coro::coroutine_handle<> await_suspend(coro::coroutine_handle<promise_type> h) noexcept {
        h.destroy();
        return {};
      }
      void await_resume() noexcept {}
    };

    void unhandled_exception() noexcept {}

    final_awaiter final_suspend() noexcept { return {}; }

    coro::suspend_always initial_suspend() noexcept { return {}; }
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

  coro::coroutine_handle<promise_type> coro_;
};

detached_task foo() {
  co_return;
}

// check that the lifetime of the coroutine handle used to obtain the address ended right away.
// CHECK:       %{{.*}} = call i8* @{{.*address.*}}(%"struct.std::experimental::coroutines_v1::coroutine_handle.0"* nonnull %{{.*}})
// CHECK-NEXT:  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %{{.*}})
