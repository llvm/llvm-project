// RUN: %clang_cc1 -fsyntax-only -Wunused-parameter -verify -std=c++20 -Wno-coroutines-unsupported-target %s

#include "Inputs/std-coroutine.h"

struct awaitable {
  bool await_ready() noexcept;
  void await_resume() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
};

struct promise_arg {};

struct task : awaitable {
  struct promise_type {
    promise_type();
    promise_type(promise_arg);
    task get_return_object() noexcept;
    awaitable initial_suspend() noexcept;
    awaitable final_suspend() noexcept;
    void unhandled_exception() noexcept;
    void return_void() noexcept;
  };
};

struct allocation_arg {};

struct task_with_new {
  struct promise_type {
    void *operator new(decltype(sizeof(0)));
    void *operator new(decltype(sizeof(0)), allocation_arg);
    task_with_new get_return_object();
    awaitable initial_suspend();
    awaitable final_suspend() noexcept;
    void unhandled_exception();
    void return_void();
  };
};

struct task_with_variadic_new {
  struct promise_type {
    void *operator new(decltype(sizeof(0)), ...);
    task_with_variadic_new get_return_object();
    awaitable initial_suspend();
    awaitable final_suspend() noexcept;
    void unhandled_exception();
    void return_void();
  };
};

task foo(int a) { // expected-warning{{unused parameter 'a'}}
  co_return;
}

task promise_constructor_uses_parameter(promise_arg a) { co_return; }

task_with_new class_specific_new_fallback(
    int a) { // expected-warning{{unused parameter 'a'}}
  co_return;
}

task_with_new placement_allocation_uses_parameter(allocation_arg a) {
  co_return;
}

task_with_variadic_new variadic_allocation_uses_parameter(int a) {
  co_return;
}

task bar(int a, int b) { // expected-warning{{unused parameter 'b'}}
  a = a + 1;
  co_return;
}

void create_closure() {
  auto closure = [](int c) -> task { // expected-warning{{unused parameter 'c'}}
    co_return;
  };
}
