// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -Winvalid-noreturn -verify

#include "Inputs/std-coroutine.h"

struct Promise;

struct Awaitable {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  void await_resume();
};

struct Coro : std::coroutine_handle<> {
  using promise_type = Promise;
};

struct Promise {
  Coro get_return_object();
  std::suspend_always initial_suspend() noexcept;
  std::suspend_always final_suspend() noexcept;
  void return_void();
  void unhandled_exception();
};

[[noreturn]] Coro test() { // expected-warning {{coroutine 'test' cannot be declared 'noreturn' as it always returns a coroutine handle}}
  co_await Awaitable{};
}

// NO warning here. This could be a regular function returning a `Coro` object.
[[noreturn]] Coro test2();
