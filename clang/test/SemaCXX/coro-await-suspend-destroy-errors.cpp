// RUN: %clang_cc1 -std=c++20 -verify %s

#include "Inputs/std-coroutine.h"

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

struct WrongReturnTypeAwaitable {
  bool await_ready() { return false; }
  bool await_suspend_destroy(auto& promise) { return true; } // expected-error {{return type of 'await_suspend_destroy' is required to be 'void' (have 'bool')}}
  [[clang::coro_await_suspend_destroy]] 
  bool await_suspend(auto handle) {}
  void await_resume() {}
};

Task test_wrong_return_type() {
  co_await WrongReturnTypeAwaitable{}; // expected-note {{call to 'await_suspend_destroy<Task::promise_type>' implicitly required by coroutine function here}}
}

struct NoSuchMemberAwaitable {
  bool await_ready() { return false; }
  [[clang::coro_await_suspend_destroy]] 
  void await_suspend(auto handle) {}
  void await_resume() {}
};

Task test_no_method() {
  co_await NoSuchMemberAwaitable{}; // expected-error {{no member named 'await_suspend_destroy' in 'NoSuchMemberAwaitable'}}
}

struct WrongOverloadAwaitable {
  bool await_ready() { return false; }
  void await_suspend_destroy(int x) {} // expected-note {{passing argument to parameter 'x' here}}
  [[clang::coro_await_suspend_destroy]] 
  void await_suspend(auto handle) {}
  void await_resume() {}
};

Task test_wrong_overload() {
  co_await WrongOverloadAwaitable{}; // expected-error {{no viable conversion from 'std::coroutine_traits<Task>::promise_type' (aka 'typename Task::promise_type') to 'int'}}
}

struct ReturnTypeMismatchAwaiter {
  bool await_ready() { return false; }
  void await_suspend_destroy(auto& promise) {} // expected-error {{return type of 'await_suspend' ('bool') must match return type of 'await_suspend_destroy' ('void')}}
  [[clang::coro_await_suspend_destroy]] 
  bool await_suspend(auto handle) { return true; }
  void await_resume() {}
};

Task test_return_type_mismatch() {
  co_await ReturnTypeMismatchAwaiter{}; // expected-note {{call to 'await_suspend_destroy<Task::promise_type>' implicitly required by coroutine function here}}
}
