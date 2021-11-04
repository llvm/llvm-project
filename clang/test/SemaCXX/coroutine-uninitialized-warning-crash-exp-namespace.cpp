// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++14 -fcoroutines-ts -fsyntax-only -Wall -Wextra -Wuninitialized  -fblocks
#include "Inputs/std-coroutine-exp-namespace.h"

using namespace std::experimental;

struct A {
  bool await_ready() { return true; }
  int await_resume() { return 42; }
  template <typename F>
  void await_suspend(F) {}
};

struct coro_t {
  struct promise_type {
    coro_t get_return_object() { return {}; }
    suspend_never initial_suspend() { return {}; }
    suspend_never final_suspend() noexcept { return {}; }
    A yield_value(int) { return {}; }
    void return_void() {}
    static void unhandled_exception() {}
  };
};

coro_t f(int n) {
  if (n == 0)
    co_return;
  co_yield 42;
  int x = co_await A{};
}

template <class Await>
coro_t g(int n) {
  if (n == 0)
    co_return;
  co_yield 42;
  int x = co_await Await{};
}

int main() {
  f(0);
  g<A>(0);
}
