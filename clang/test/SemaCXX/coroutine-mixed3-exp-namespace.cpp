// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

#include "Inputs/std-coroutine.h"

namespace std::experimental {
using std::coroutine_handle;
using std::coroutine_traits; // expected-note{{declared here}}
} // namespace std::experimental

struct my_awaitable {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<> coro) noexcept;
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
  co_return;
  // expected-warning@-1{{support for std::experimental::coroutine_traits will be removed}}
}
