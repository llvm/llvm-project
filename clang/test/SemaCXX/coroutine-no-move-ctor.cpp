// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 -fsyntax-only -verify
// expected-no-diagnostics

#include "Inputs/std-coroutine.h"

class invoker {
public:
  class invoker_promise {
  public:
    invoker get_return_object() { return invoker{}; }
    auto initial_suspend() { return std::suspend_never{}; }
    auto final_suspend() noexcept { return std::suspend_never{}; }
    void return_void() {}
    void unhandled_exception() {}
  };
  using promise_type = invoker_promise;
  invoker() {}
  invoker(const invoker &) = delete;
  invoker &operator=(const invoker &) = delete;
  invoker(invoker &&) = delete;
  invoker &operator=(invoker &&) = delete;
};

invoker f() {
  co_return;
}
