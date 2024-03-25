// RUN: %clang_cc1 -fsyntax-only -Wunused-parameter -verify -std=c++20 %s

#include "Inputs/std-coroutine.h"

struct awaitable {
  bool await_ready() noexcept;
  void await_resume() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
};

struct task : awaitable {
  struct promise_type {
    task get_return_object() noexcept;
    awaitable initial_suspend() noexcept;
    awaitable final_suspend() noexcept;
    void unhandled_exception() noexcept;
    void return_void() noexcept;
  };
};

task foo(int a) { // expected-warning{{unused parameter 'a'}}
  co_return;
}

task bar(int a, int b) { // expected-warning{{unused parameter 'b'}}
  a = a + 1;
  co_return;
}

void create_closure() {
  auto closure = [](int c) -> task { // expected-warning{{unused parameter 'c'}}
    co_return;
  };
}
