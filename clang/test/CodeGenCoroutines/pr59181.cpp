// Test for PR59181. Tests that no conditional cleanup is created around await_suspend.
//
// REQUIRES: x86-registered-target
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - -std=c++20 -disable-llvm-passes -fsanitize-address-use-after-scope | FileCheck %s

#include "Inputs/coroutine.h"

struct Task {
  int value_;
  struct promise_type {
    Task get_return_object() {
      return Task{0};
    }

    std::suspend_never initial_suspend() noexcept {
      return {};
    }

    std::suspend_never final_suspend() noexcept {
      return {};
    }

    void return_value(Task t) noexcept {}
    void unhandled_exception() noexcept {}

    auto await_transform(Task t) {
      struct Suspension {
        auto await_ready() noexcept { return false;}
        auto await_suspend(std::coroutine_handle<> coro) {
          coro.destroy();
        }

        auto await_resume() noexcept {
          return 0;
        }
      };
      return Suspension{};
    }
  };
};

Task bar(bool cond) {
  co_return cond ? Task{ co_await Task{}}: Task{};
}

void foo() {
  bar(true);
}

// CHECK: cleanup.cont:{{.*}}
// CHECK-NEXT: load i8
// CHECK-NEXT: trunc
// CHECK-NEXT: store i1 false
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[REF:%ref.tmp[0-9]+]])

// CHECK: await.suspend:{{.*}}
// CHECK-NOT: call void @llvm.lifetime.start.p0(i64 8, ptr [[REF]])
// CHECK: call void @_ZZN4Task12promise_type15await_transformES_EN10Suspension13await_suspendESt16coroutine_handleIvE
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[REF]])
