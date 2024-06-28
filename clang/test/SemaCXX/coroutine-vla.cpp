// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -Wno-vla-cxx-extension -verify
#include "Inputs/std-coroutine.h"

struct promise;

struct coroutine : std::coroutine_handle<promise> {
  using promise_type = ::promise;
};

struct promise
{
    coroutine get_return_object();
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
};

// Test that we won't report the error incorrectly.
void bar(int n) {
  int array[n];
  return;
}

coroutine foo(int n) {
  int array[n]; // expected-error {{variable length arrays in a coroutine are not supported}}
  co_return;
}

void lambda() {
  [](int n) -> coroutine {
    int array[n]; // expected-error {{variable length arrays in a coroutine are not supported}}
    co_return;
  }(10);
}
