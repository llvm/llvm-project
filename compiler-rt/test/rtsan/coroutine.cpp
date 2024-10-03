// RUN: %clangxx -std=c++20 -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Coroutines allocate memory and are not allowed in a [[clang::nonblocking]] function.

#include <coroutine>

struct SimpleCoroutine {
  struct promise_type {
    SimpleCoroutine get_return_object() { return SimpleCoroutine{}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
    void return_void() {}
  };
};

SimpleCoroutine example_coroutine() { co_return; }

void calls_a_coroutine() [[clang::nonblocking]] { example_coroutine(); }

int main() {
  calls_a_coroutine();
  return 0;
}

// CHECK: ==ERROR: RealtimeSanitizer

// Somewhere in the stack this should be mentioned
// CHECK: calls_a_coroutine
