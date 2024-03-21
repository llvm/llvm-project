// RUN: %clang_cc1 -std=c++20 -triple=x86_64-- -emit-llvm -fcxx-exceptions \
// RUN:            -disable-llvm-passes %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct task {
  struct promise_type {
    task get_return_object();
    std::suspend_never initial_suspend();
    std::suspend_never final_suspend() noexcept;
    void return_void();
    void unhandled_exception() noexcept;
  };
};

task f() try {
  co_return;
} catch(...) {
}

// CHECK-LABEL: define{{.*}} void @_Z1fv(
// CHECK: call void @llvm.coro.await.suspend.void(
// CHECK: call void @_ZN4task12promise_type11return_voidEv(
