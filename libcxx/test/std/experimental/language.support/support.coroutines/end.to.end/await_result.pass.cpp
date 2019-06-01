// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <experimental/coroutine>
#include <cassert>

#include "test_macros.h"

using namespace std::experimental;

struct coro_t {
  struct promise_type {
    coro_t get_return_object() {
      coroutine_handle<promise_type>{};
      return {};
    }
    suspend_never initial_suspend() { return {}; }
    suspend_never final_suspend() { return {}; }
    void return_void() {}
    static void unhandled_exception() {}
  };
};

struct B {
  ~B() {}
  bool await_ready() { return true; }
  B await_resume() { return {}; }
  template <typename F> void await_suspend(F) {}
};


struct A {
  ~A() {}
  bool await_ready() { return true; }
  int await_resume() { return 42; }
  template <typename F> void await_suspend(F) {}
};

int last_value = -1;
void set_value(int x) {
  last_value = x;
}

coro_t f(int n) {
  if (n == 0) {
    set_value(0);
    co_return;
  }
  int val = co_await A{};
  ((void)val);
  set_value(42);
}

coro_t g() { B val = co_await B{}; }

int main(int, char**) {
  last_value = -1;
  f(0);
  assert(last_value == 0);
  f(1);
  assert(last_value == 42);

  return 0;
}
