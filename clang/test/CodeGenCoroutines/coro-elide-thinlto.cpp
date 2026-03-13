// REQUIRES: x86_64-linux
// This tests that the coroutine elide optimization could happen succesfully with ThinLTO.
// This test is adapted from coro-elide.cpp and splits functions into two files.
//
// RUN: split-file %s %t
// RUN: %clang --target=x86_64-linux -std=c++20 -O2 -flto=thin -I %S -c %t/coro-elide-callee.cpp -o %t/coro-elide-callee.bc
// RUN: %clang --target=x86_64-linux -std=c++20 -O2 -flto=thin -I %S -c %t/coro-elide-caller.cpp -o %t/coro-elide-caller.bc
// RUN: llvm-lto --thinlto %t/coro-elide-callee.bc %t/coro-elide-caller.bc -o %t/summary
// RUN: %clang_cc1 -O2 -x ir %t/coro-elide-caller.bc -fthinlto-index=%t/summary.thinlto.bc -emit-llvm -o - | FileCheck %s

//--- coro-elide-task.h
#pragma once
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

//--- coro-elide-callee.cpp
#include "coro-elide-task.h"
Task task0() {
  co_return 43;
}

//--- coro-elide-caller.cpp
#include "coro-elide-task.h"

Task task0();

Task task1() {
  co_return co_await task0();
}

// CHECK-LABEL: define{{.*}} void @_Z5task1v.resume
// CHECK-NOT: {{.*}}_Znwm
