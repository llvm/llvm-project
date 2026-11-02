// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

#include "Inputs/std-coroutine.h"

struct Awaiter {
  int Value;

  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  int await_resume();
};

struct Task {
  struct promise_type {
    Task get_return_object();
    std::suspend_never initial_suspend() noexcept;
    std::suspend_never final_suspend() noexcept;
    Awaiter yield_value(int Value);
    void return_void();
    void unhandled_exception();
  };
};

template <int... Values>
Task yieldPack() {
  ((void)(co_yield Values), ...);
}

template Task yieldPack<1, 2, 3>();
template Task yieldPack<>();

template <int... Values>
Task awaitPack() {
  ((void)(co_await Awaiter{Values}), ...);
}

template Task awaitPack<1, 2, 3>();
template Task awaitPack<>();

template <int... Values>
Task unexpandedYield() {
  co_yield Values; // expected-error {{expression contains unexpanded parameter pack 'Values'}}
}

template <int... Values>
Task unexpandedAwait() {
  co_await Awaiter{Values}; // expected-error {{expression contains unexpanded parameter pack 'Values'}}
}
