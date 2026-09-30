// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

#include "Inputs/std-coroutine.h"

struct Tag {};
struct Y {
  bool await_ready() const;
  void await_suspend(std::coroutine_handle<>) const;
  void await_resume() const;
};

struct Promise {
  Tag get_return_object();
  std::suspend_always initial_suspend();
  // We intentionally omit final_suspend to trigger the error path
  void unhandled_exception();
  Y yield_value(int);
  void return_void();
};

template <class... Args> struct std::coroutine_traits<Tag, Args...> {
  using promise_type = Promise;
};

template <class T> struct S {
  // The error is diagnosed at the function declaration, not the yield statement.
  template <class U> static Tag f() { // expected-error {{no member named 'final_suspend' in 'Promise'}}
    co_yield 0;
  }
};

Tag g() { return S<int>::f<void>(); }