// This tests that the symmetric transfer at the final suspend point could happen successfully.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct Task {
  struct promise_type {
    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }
      template <typename PromiseType>
      std::coroutine_handle<> await_suspend(std::coroutine_handle<PromiseType> h) noexcept {
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
    std::coroutine_handle<promise_type> handle;
    Awaiter(std::coroutine_handle<promise_type> handle) : handle(handle) {}
    bool await_ready() const noexcept { return false; }
    std::coroutine_handle<void> await_suspend(std::coroutine_handle<void> continuation) noexcept {
      handle.promise().continuation = continuation;
      return handle;
    }
    int await_resume() noexcept {
      int ret = handle.promise()._value;
      handle.destroy();
      return ret;
    }
  };

  auto operator co_await() {
    auto handle_ = handle;
    handle = nullptr;
    return Awaiter(handle_);
  }

private:
  std::coroutine_handle<promise_type> handle;
};

Task task0() {
  co_return 43;
}

// CHECK-LABEL: define{{.*}} void @_Z5task0v.resume
// This checks we are still in the scope of the current function.
// CHECK-NOT: {{^}}}
// CHECK: musttail call fastcc void
// CHECK-NEXT: ret void
