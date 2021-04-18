// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -std=c++2a %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -std=c++2a %s -emit-llvm -disable-llvm-passes -o - | opt -always-inline -S | FileCheck --check-prefix=INLINE %s

#include "Inputs/coroutine.h"

namespace coro = std::experimental::coroutines_v1;

class task {
public:
  class promise_type {
  public:
    task get_return_object() noexcept;
    coro::suspend_always initial_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;

    struct final_awaiter {
      bool await_ready() noexcept;
      void await_suspend(coro::coroutine_handle<promise_type> h) noexcept;
      void await_resume() noexcept;
    };

    final_awaiter final_suspend() noexcept;

    coro::coroutine_handle<> continuation;
  };

  task(task &&t) noexcept;
  ~task();

  class awaiter {
  public:
    bool await_ready() noexcept;
    void await_suspend(coro::coroutine_handle<> continuation) noexcept;
    void await_resume() noexcept;

  private:
    friend task;
    explicit awaiter(coro::coroutine_handle<promise_type> h) noexcept;
    coro::coroutine_handle<promise_type> coro_;
  };

  awaiter operator co_await() &&noexcept;

private:
  explicit task(coro::coroutine_handle<promise_type> h) noexcept;
  coro::coroutine_handle<promise_type> coro_;
};

task cee();

__attribute__((always_inline)) inline task bar() {
  co_await cee();
  co_return;
}

task foo() {
  co_await bar();
  co_return;
}

// check that Clang front-end will tag bar with both alwaysinline and coroutine presplit
// CHECK:       define linkonce_odr void @_Z3barv({{.*}}) #[[ATTR:[0-9]+]] {{.*}}
// CHECK:       attributes #[[ATTR]] = { alwaysinline {{.*}} "coroutine.presplit"="0" {{.*}}}

// check that bar is not inlined even it's marked as always_inline
// INLINE-LABEL: define dso_local void @_Z3foov(
// INLINE:         call void @_Z3barv(
