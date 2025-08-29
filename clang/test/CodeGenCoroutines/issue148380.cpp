// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -O2 | FileCheck %s --check-prefix=CHECK-OPT

// This test just confirms that `[[clang::coro_await_suspend_destroy]]` works
// around the optimization problem from PR148380.
//
// See `coro-await-suspend-destroy.cpp` for a test showing the detailed control
// flow in un-lowered, un-optimized IR.

#include "Inputs/coroutine.h"

struct coro {
  struct promise_type {
    auto get_return_object() { return coro{}; }
    auto initial_suspend() noexcept { return std::suspend_never{}; }
    auto final_suspend() noexcept { return std::suspend_never{}; }
    auto unhandled_exception() {}
    auto return_void() {}
  };

  auto await_ready() { return false; }
  void await_suspend_destroy(auto& promise) {}
  [[clang::coro_await_suspend_destroy]] auto await_suspend(auto handle) {
    // The attribute causes this stub not to be called.  Instead, we call
    // `await_suspend_destroy()`, as on the next line.
    await_suspend_destroy(handle.promise());
    handle.destroy();
  }
  auto await_resume() {}
};

coro f1() noexcept;
coro f2() noexcept
{
    co_await f1();
}

// CHECK-OPT: define{{.+}} void @_Z2f2v({{.+}} {
// CHECK-OPT-NEXT: entry:
// CHECK-OPT-NEXT: tail call void @_Z2f1v()
// CHECK-OPT-NEXT: ret void
// CHECK-OPT-NEXT: }
