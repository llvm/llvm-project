// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -std=c++20 -fsyntax-only -verify

// Verify that coroutines with promise types that inherit operator new/delete from a base class via
// using declarations don't crash during compilation.
// This is a regression test for a bug where DiagnoseTypeAwareAllocators didn't handle
// UsingShadowDecl properly.

// expected-no-diagnostics

#include "Inputs/std-coroutine.h"

namespace std {
  typedef __SIZE_TYPE__ size_t;
}

struct PromiseBase {
  static void *operator new(std::size_t size);
  static void operator delete(void *ptr, std::size_t size);
};

struct Task {
  struct promise_type : PromiseBase {
    using PromiseBase::operator new;
    using PromiseBase::operator delete;

    Task get_return_object() {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };

  std::coroutine_handle<promise_type> handle;
};

Task example_coroutine() {
  co_return;
}
