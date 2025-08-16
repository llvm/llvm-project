// RUN: %clang_cc1 -std=c++20 -verify %s 

#include "Inputs/coroutine.h"

// Coroutine type with `std::suspend_never` for initial/final suspend
struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

struct [[clang::coro_await_suspend_destroy]] WrongReturnTypeAwaitable {
  bool await_ready() { return false; }
  bool await_suspend_destroy(auto& promise) { return true; } // expected-error {{return type of 'await_suspend_destroy' is required to be 'void' (have 'bool')}}
  void await_suspend(auto handle) {
    await_suspend_destroy(handle.promise());
    handle.destroy();
  }
  void await_resume() {}
};

Task test_invalid_destroying_await() {
  co_await WrongReturnTypeAwaitable{}; // expected-note {{call to 'await_suspend_destroy<Task::promise_type>' implicitly required by coroutine function here}}
}

struct [[clang::coro_await_suspend_destroy]] MissingMethodAwaitable {
  bool await_ready() { return false; }
  // Missing await_suspend_destroy method
  void await_suspend(auto handle) {
    handle.destroy();
  }
  void await_resume() {}
};

Task test_missing_method() {
  co_await MissingMethodAwaitable{}; // expected-error {{no member named 'await_suspend_destroy' in 'MissingMethodAwaitable'}}
}

struct [[clang::coro_await_suspend_destroy]] WrongParameterTypeAwaitable {
  bool await_ready() { return false; }
  void await_suspend_destroy(int x) {} // expected-note {{passing argument to parameter 'x' here}}
  void await_suspend(auto handle) {
    await_suspend_destroy(handle.promise());
    handle.destroy();
  }
  void await_resume() {}
};

Task test_wrong_parameter_type() {
  co_await WrongParameterTypeAwaitable{}; // expected-error {{no viable conversion from 'std::coroutine_traits<Task>::promise_type' (aka 'Task::promise_type') to 'int'}}
}
