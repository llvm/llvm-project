// An end-to-end test to make sure things get processed correctly.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -O3 | \
// RUN:     FileCheck %s

#include "Inputs/coroutine.h"

struct SomeAwaitable {
  // Resume the supplied handle once the awaitable becomes ready,
  // returning a handle that should be resumed now for the sake of symmetric transfer.
  // If the awaitable is already ready, return an empty handle without doing anything.
  //
  // Defined in another translation unit. Note that this may contain
  // code that synchronizees with another thread.
  std::coroutine_handle<> Register(std::coroutine_handle<>);
};

// Defined in another translation unit.
void DidntSuspend();

struct Awaiter {
  SomeAwaitable&& awaitable;
  bool suspended;

  bool await_ready() { return false; }

  std::coroutine_handle<> await_suspend(const std::coroutine_handle<> h) {
    // Assume we will suspend unless proven otherwise below. We must do
    // this *before* calling Register, since we may be destroyed by another
    // thread asynchronously as soon as we have registered.
    suspended = true;

    // Attempt to hand off responsibility for resuming/destroying the coroutine.
    const auto to_resume = awaitable.Register(h);

    if (!to_resume) {
      // The awaitable is already ready. In this case we know that Register didn't
      // hand off responsibility for the coroutine. So record the fact that we didn't
      // actually suspend, and tell the compiler to resume us inline.
      suspended = false;
      return h;
    }

    // Resume whatever Register wants us to resume.
    return to_resume;
  }

  void await_resume() {
    // If we didn't suspend, make note of that fact.
    if (!suspended) {
      DidntSuspend();
    }
  }
};

struct MyTask{
  struct promise_type {
    MyTask get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception();

    Awaiter await_transform(SomeAwaitable&& awaitable) {
      return Awaiter{static_cast<SomeAwaitable&&>(awaitable)};
    }
  };
};

MyTask FooBar() {
  co_await SomeAwaitable();
}

// CHECK-LABEL: @_Z6FooBarv
// CHECK: %[[to_resume:.*]] = {{.*}}call ptr @_ZN13SomeAwaitable8RegisterESt16coroutine_handleIvE
// CHECK-NEXT: %[[to_bool:.*]] = icmp eq ptr %[[to_resume]], null
// CHECK-NEXT: br i1 %[[to_bool]], label %[[then:.*]], label %[[else:.*]]

// CHECK: [[then]]:
// We only access the coroutine frame conditionally as the sources did.
// CHECK:   store i8 0,
// CHECK-NEXT: br label %[[else]]

// CHECK: [[else]]:
// No more access to the coroutine frame until suspended.
// CHECK-NOT: store
// CHECK: }
