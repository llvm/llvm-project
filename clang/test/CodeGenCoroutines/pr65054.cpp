// The output of O0 is highly redundant and hard to test. Also it is not good
// limit the output of O0. So we test the optimized output from O0. The idea
// is the optimizations shouldn't change the semantics of the program.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:      -O0 -emit-llvm %s -o - -disable-O0-optnone \
// RUN:      | opt -passes='sroa,mem2reg,simplifycfg' -S | FileCheck %s --check-prefix=CHECK-O0

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
    MyTask get_return_object() {
      return MyTask{
          std::coroutine_handle<promise_type>::from_promise(*this),
      };
    }

    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception();

    auto await_transform(const int& x) { return Awaiter{x}; }
  };

  std::coroutine_handle<> h;
};

// A global array of integers.
int g_array[32];

// A coroutine that awaits each integer in the global array.
MyTask FooBar() {
  for (const int& x : g_array) {
    co_await x;
  }
}

// CHECK-O0: define{{.*}}@_Z6FooBarv.resume
// CHECK-O0: call{{.*}}@_Z6FooBarv.__await_suspend_wrapper__await(
// CHECK-O0-NOT: store
// CHECK-O0: ret void
