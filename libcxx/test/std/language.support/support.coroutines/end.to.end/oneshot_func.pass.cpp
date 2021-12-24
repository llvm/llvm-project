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
#include <vector>
#include <cassert>

#include "test_macros.h"

// This file tests, one shot, movable std::function like thing using coroutine
// for compile-time type erasure and unerasure.

template <typename R> struct func {
  struct promise_type {
    R result;
    func get_return_object() { return {this}; }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_value(R v) { result = v; }
    void unhandled_exception() {}
  };

  R operator()() {
    h.resume();
    R result = h.promise().result;
    h.destroy();
    h = nullptr;
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

  template <typename F> static func Create(F f) { co_return f(); }

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

int main(int, char**) {
  func<int> f = func<int>::Create([]() { return 44; });
  assert(f() == 44);

  return 0;
}
