// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:      -O1 -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

// A simple awaiter type with an await_suspend method that can't be
// inlined.
struct Awaiter {
  const int& x;

  bool await_ready() { return false; }
  std::coroutine_handle<> await_suspend(const std::coroutine_handle<> h);
  void await_resume() {}
};

struct MyTask {
  // A lazy promise with an await_transform method that supports awaiting
  // integer references using the Awaiter struct above.
  struct promise_type {
    MyTask get_return_object() { return {}; }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception();

    auto await_transform(const int& x) { return Awaiter{x}; }
  };
};

// A global array of integers.
int g_array[32];

// A coroutine that awaits each integer in the global array.
MyTask FooBar() {
  for (const int& x : g_array) {
    co_await x;
  }
}

// CHECK: %[[RET:.+]] = {{.*}}call{{.*}}@_ZN7Awaiter13await_suspendESt16coroutine_handleIvE
// CHECK: %[[RESUME_ADDR:.+]] = load ptr, ptr %[[RET]],
// CHECK: musttail call fastcc void %[[RESUME_ADDR]]({{.*}}%[[RET]]
// CHECK: ret

