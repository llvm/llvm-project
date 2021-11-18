// This file is to test the mixed use of `std::experimental::coroutine_traits` and `std::coroutine_traits`
// which is similar to coroutine-mixed-exp-namesapce. This file tests the relative order of
// included header wouldn't affect the diagnostic messages.
// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

#include "Inputs/std-coroutine-exp-namespace.h"
#include "Inputs/std-coroutine.h"

struct my_awaitable {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<> coro) noexcept;
  void await_resume() noexcept;
};

struct promise_void {
  void get_return_object();
  my_awaitable initial_suspend();
  my_awaitable final_suspend() noexcept;
  void return_void();
  void unhandled_exception();
};

template <>
struct std::coroutine_traits<void> { using promise_type = promise_void; };

void test() {
  co_return; // expected-error {{Found mixed use of std namespace and std::experimental namespace for coroutine, which is disallowed. The coroutine components in std::experimental namespace is deprecated. Please use coroutine components under std namespace.}}
}
