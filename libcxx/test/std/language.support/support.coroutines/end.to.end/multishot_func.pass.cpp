//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-coroutines

#include <coroutine>
#include <cassert>

#include "test_macros.h"

// This file tests, multishot, movable std::function like thing using coroutine
// for compile-time type erasure and unerasure.
template <typename R> struct func {
  struct Input {R a, b;};

  struct promise_type {
    Input* I;
    R result;
    func get_return_object() { return {this}; }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    template <typename F>
    std::suspend_always yield_value(F&& f) {
      result = f(I->a, I->b);
      return {};
    }
    void unhandled_exception() {}
  };

  R operator()(Input I) {
    h.promise().I = &I;
    h.resume();
    R result = h.promise().result;
    return result;
  };

  func() {}
  func(func &&rhs) : h(rhs.h) { rhs.h = nullptr; }
  func(func const &) = delete;

  func &operator=(func &&rhs) {
    if (this != &rhs) {
      if (h)
        h.destroy();
      h = rhs.h;
      rhs.h = nullptr;
    }
    return *this;
  }

  template <typename F> static func Create(F f) {
    for (;;) {
      co_yield f;
    }
  }

  template <typename F> func(F f) : func(Create(f)) {}

  ~func() {
    if (h)
      h.destroy();
  }

private:
  func(promise_type *promise)
      : h(std::coroutine_handle<promise_type>::from_promise(*promise)) {}
  std::coroutine_handle<promise_type> h;
};

int Do(int acc, int n, func<int> f) {
  for (int i = 0; i < n; ++i)
    acc = f({acc, i});
  return acc;
}

int main(int, char**) {
  int result = Do(1, 10, [](int a, int b) {return a + b;});
  assert(result == 46);

  return 0;
}
