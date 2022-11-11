// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify
#include "Inputs/std-coroutine.h"

struct MyTask{
  struct promise_type {
    MyTask get_return_object();
    std::suspend_always initial_suspend() { return {}; }

    void unhandled_exception();
    void return_void();
    auto final_suspend() noexcept {
      struct Awaiter {
        bool await_ready() noexcept { return false; }
        std::coroutine_handle<promise_type> await_suspend(std::coroutine_handle<promise_type> h) noexcept;
        void await_resume() noexcept;
      };

      return Awaiter{};
    }

    // The coroutine to resume when we're done.
    std::coroutine_handle<promise_type> resume_when_done;
  };
};

MyTask DoSomething() {
  static_assert(__is_same(void, decltype(co_await 0))); // expected-error {{'co_await' cannot be used in an unevaluated context}}
  co_return;
}

MyTask DoAnotherthing() {
  static_assert(__is_same(void, decltype(co_yield 0))); // expected-error {{'co_yield' cannot be used in an unevaluated context}}
  co_return;
}
