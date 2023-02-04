// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

#include "Inputs/std-coroutine.h"

namespace std::experimental {
// expected-note@+1{{declared here}}
template <typename T> using coroutine_traits = std::coroutine_traits<T>;
using std::coroutine_handle;
} // namespace std::experimental

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
  co_return; // expected-error {{mixed use of std and std::experimental namespaces for coroutine components}}
  // expected-warning@-1{{support for 'std::experimental::coroutine_traits' will be removed}}
  // expected-note@Inputs/std-coroutine.h:18 {{'coroutine_traits' declared here}}
}
