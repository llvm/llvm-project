// RUN: %clang_cc1 -std=c++20 -triple i686-pc-windows-msvc -verify -Wno-coroutines-unsupported-target %s

#include "Inputs/std-coroutine.h"

struct Noisy {
  int val;
  Noisy(int v);
  Noisy(const Noisy&) = delete;
  Noisy(Noisy&& o) = delete; // expected-note 2 {{'Noisy' has been explicitly marked deleted here}}
  ~Noisy();
};

struct Awaiter {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  int await_resume() noexcept;
};

struct NoisyAwaiter {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  Noisy await_resume() noexcept;
};

struct task {
  struct promise_type {
    task get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

void consume(Noisy x); // expected-note 2 {{passing argument to parameter 'x' here}}

task my_coroutine() {
  consume(Noisy(42)); // OK, no suspend, no bypass
  consume(co_await NoisyAwaiter{}); // expected-error {{call to deleted constructor of 'Noisy'}}
  consume(Noisy(co_await Awaiter{})); // expected-error {{call to deleted constructor of 'Noisy'}}
}
