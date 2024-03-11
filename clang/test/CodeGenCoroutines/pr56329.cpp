// Test for PR56919. Tests the we won't contain the resumption of final suspend point.
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %s -O3 -S -emit-llvm -o - | FileCheck %s
// This test is expected to fail on PowerPC.
// XFAIL: target=powerpc{{.*}}

#include "Inputs/coroutine.h"

void _exit(int status) __attribute__ ((__noreturn__));

class Promise;

// An object that can be co_awaited, but we always resume immediately from
// await_suspend.
struct ResumeFromAwaitSuspend{};

struct Task {
  using promise_type = Promise;
  Promise& promise;
};

struct Promise {
  static std::coroutine_handle<Promise> GetHandle(Promise& promise) {
    return std::coroutine_handle<Promise>::from_promise(promise);
  }

  void unhandled_exception() {}
  Task get_return_object() { return Task{*this}; }
  void return_void() {}

  // Always suspend before starting the coroutine body. We actually run the body
  // when we are co_awaited.
  std::suspend_always initial_suspend() { return {}; }

  // We support awaiting tasks. We do so by configuring them to resume us when
  // they are finished, and then resuming them from their initial suspend.
  auto await_transform(Task&& task) {
    struct Awaiter {
      bool await_ready() { return false; }

      std::coroutine_handle<> await_suspend(
          const std::coroutine_handle<> handle) {
        // Tell the child to resume the parent once it finishes.
        child.resume_at_final_suspend = GetHandle(parent);

        // Run the child.
        return GetHandle(child);
      }

      void await_resume() {
        // The child is now at its final suspend point, and can be destroyed.
        return GetHandle(child).destroy();
      }

      Promise& parent;
      Promise& child;
    };

    return Awaiter{
        .parent = *this,
        .child = task.promise,
    };
  }

  // Make evaluation of `co_await ResumeFromAwaitSuspend{}` go through the
  // await_suspend path, but cause it to resume immediately by returning our own
  // handle to resume.
  auto await_transform(ResumeFromAwaitSuspend) {
    struct Awaiter {
      bool await_ready() { return false; }

      std::coroutine_handle<> await_suspend(const std::coroutine_handle<> h) {
        return h;
      }

      void await_resume() {}
    };

    return Awaiter{};
  }

  // Always suspend at the final suspend point, transferring control back to our
  // caller. We expect never to be resumed from the final suspend.
  auto final_suspend() noexcept {
    struct FinalSuspendAwaitable final {
      bool await_ready() noexcept { return false; }

      std::coroutine_handle<> await_suspend(std::coroutine_handle<>) noexcept {
        return promise.resume_at_final_suspend;
      }

      void await_resume() noexcept {
        _exit(1);
      }

      Promise& promise;
    };

    return FinalSuspendAwaitable{.promise = *this};
  }

  // The handle we will resume once we hit final suspend.
  std::coroutine_handle<> resume_at_final_suspend;
};

Task Inner();

Task Outer() {
  co_await ResumeFromAwaitSuspend();
  co_await Inner();
}

// CHECK: define{{.*}}@_Z5Outerv.resume(
// CHECK-NOT: }
// CHECK-NOT: _exit
// CHECK: musttail call
// CHECK: musttail call
// CHECK: musttail call
// CHECK-NEXT: ret void
// CHECK-EMPTY:
// CHECK-NEXT: unreachable:
// CHECK-NEXT: unreachable
// CHECK-NEXT: }
