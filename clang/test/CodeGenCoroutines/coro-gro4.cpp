// Test that the GRO destructor does not enter the resume or destroy parts
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 -disable-llvm-passes -emit-llvm %s -o - | opt -passes='default<O0>,default<O2>' -S | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

extern "C" void gro_destroy() noexcept;

struct task {
  struct promise_type {
    task get_return_object() noexcept { return task{std::coroutine_handle<promise_type>::from_promise(*this)}; }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() {}
  };

  task(std::coroutine_handle<promise_type> handle) : m_coro(handle) {}
  task(task &&o) noexcept : m_coro(o.m_coro) { o.m_coro = nullptr; }
  task(const task&) = delete;
  task& operator=(const task&) = delete;
  task& operator=(task&&) = delete;

  ~task() {
    gro_destroy();
    if (m_coro)
      m_coro.destroy();
  }

  std::coroutine_handle<promise_type> m_coro;
};

struct wrapper {
  using promise_type = task::promise_type;

  wrapper(task &&t) noexcept : m_task(static_cast<task&&>(t)) {}

  task m_task;
};

wrapper fn() { co_return; }

// CHECK: define dso_local void @_Z2fnv
// CHECK: call void @gro_destroy()
// CHECK: ret void

// CHECK: define internal fastcc void @_Z2fnv.resume
// CHECK-NOT: call void @gro_destroy()

// CHECK: define internal fastcc void @_Z2fnv.destroy
// CHECK-NOT: call void @gro_destroy()
