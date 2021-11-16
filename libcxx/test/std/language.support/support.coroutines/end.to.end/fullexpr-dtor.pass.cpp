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

int alive = 0;
int ctor_called = 0;
int dtor_called = 0;
void reset() {
  assert(alive == 0);
  alive = 0;
  ctor_called = 0;
  dtor_called = 0;
}
struct Noisy {
  Noisy() { ++alive; ++ctor_called; }
  ~Noisy() { --alive; ++dtor_called; }
  Noisy(Noisy const&) = delete;
};

struct Bug {
  bool await_ready() { return true; }
  void await_suspend(std::coroutine_handle<>) {}
  Noisy await_resume() { return {}; }
};
struct coro2 {
  struct promise_type {
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    coro2 get_return_object() { return {}; }
    void return_void() {}
    Bug yield_value(int) { return {}; }
    void unhandled_exception() {}
  };
};

// Checks that destructors are correctly invoked for the object returned by
// coawait.
coro2 a() {
  reset();
  {
    auto x = co_await Bug{};
    assert(alive == 1);
    assert(ctor_called == 1);
    assert(dtor_called == 0);
    ((void)x);
  }
  assert(alive == 0);
  assert(dtor_called == 1);
}

coro2 b() {
  reset();
  {
    co_await Bug{};
    assert(ctor_called == 1);
    assert(dtor_called == 1);
    assert(alive == 0);
  }
  assert(ctor_called == 1);
  assert(dtor_called == 1);
  assert(alive == 0);

}

coro2 c() {
  reset();
  {
    auto x = co_yield 42;
    assert(alive == 1);
    assert(ctor_called == 1);
    assert(dtor_called == 0);
  }
  assert(alive == 0);
  assert(ctor_called == 1);
  assert(dtor_called == 1);
}

coro2 d() {
  reset();
  {
    co_yield 42;
    assert(ctor_called == 1);
    assert(dtor_called == 1);
    assert(alive == 0);
  }
  assert(alive == 0);
  assert(ctor_called == 1);
  assert(dtor_called == 1);
}

int main(int, char**) {
  a();
  b();
  c();
  d();

  return 0;
}
