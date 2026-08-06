// RUN: %clangxx_pgogen -std=c++20 -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show -function=foo -counts %t.profraw | FileCheck %s

#include <coroutine>

struct State {
  struct promise_type {
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    State get_return_object() noexcept {
      return State{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}
  };
  std::coroutine_handle<promise_type> handle;
};

struct Awaitable {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) noexcept { h.resume(); }
  void await_resume() noexcept {}
};

__attribute__((noinline)) State foo(int x) {
  for (int i = 0; i < x; ++i) {
    co_await Awaitable{};
  }
  co_return;
}

int main() {
  foo(5);
  return 0;
}

// Note: Index 6 in Block counts (the 7th value '1') represents the function entry count of foo(5).
// CHECK: Counters:
// CHECK-NEXT:  {{.*foo.*}}:
// CHECK-NEXT:    Hash: {{.*}}
// CHECK-NEXT:    Counters: 8
// CHECK-NEXT:    Block counts: [0, 5, 5, 5, 1, 1, 1, 1]
// CHECK-NOT: .resume
// CHECK-NOT: .destroy
// CHECK-NOT: .cleanup
