// This tests that the coroutine elide optimization could happen succesfully.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct Task {
  struct promise_type {
    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }
      template <typename PromiseType>
      std::coroutine_handle<> await_suspend(std::coroutine_handle<PromiseType> h) noexcept {
        if (!h)
          return std::noop_coroutine();
        return h.promise().continuation;
      }
      void await_resume() noexcept {}
    };
    Task get_return_object() noexcept {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    FinalAwaiter final_suspend() noexcept { return {}; }
    void unhandled_exception() noexcept {}
    void return_value(int x) noexcept {
      _value = x;
    }
    std::coroutine_handle<> continuation;
    int _value;
  };

  Task(std::coroutine_handle<promise_type> handle) : handle(handle) {}
  ~Task() {
    if (handle)
      handle.destroy();
  }

  struct Awaiter {
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<void> continuation) noexcept {}
    int await_resume() noexcept {
      return 43;
    }
  };

  auto operator co_await() {
    return Awaiter{};
  }

private:
  std::coroutine_handle<promise_type> handle;
};

Task task0() {
  co_return 43;
}

Task task1() {
  co_return co_await task0();
}

// CHECK-LABEL: define{{.*}} void @_Z5task1v.resume
// CHECK-NOT: call{{.*}}_Znwm
