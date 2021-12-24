// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++14 -fcoroutines-ts -fcxx-exceptions -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify -fblocks -Wall -Wextra -Wno-error=unreachable-code

#include "Inputs/std-coroutine-exp-namespace.h"

using std::experimental::suspend_always;
using std::experimental::suspend_never;

struct awaitable {
  bool await_ready();
  void await_suspend(std::experimental::coroutine_handle<>); // FIXME: coroutine_handle
  void await_resume();
} a;

struct object {
  ~object() {}
};

struct promise_void_return_value {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void unhandled_exception();
  void return_value(object);
};

struct VoidTagReturnValue {
  struct promise_type {
    VoidTagReturnValue get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void return_value(object);
  };
};

template <typename T1>
struct std::experimental::coroutine_traits<void, T1> { using promise_type = promise_void_return_value; };

VoidTagReturnValue test() {
  object x = {};
  try {
    co_return {}; // expected-warning {{support for std::experimental::coroutine_traits will be removed}}
  } catch (...) {
    throw;
  }
}
