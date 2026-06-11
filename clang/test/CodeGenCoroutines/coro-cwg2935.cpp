// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:   -fcxx-exceptions -fexceptions -emit-llvm -disable-llvm-passes \
// RUN:   %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct task {
  ~task();

  struct promise_type {
    task get_return_object();
    std::suspend_never initial_suspend();
    std::suspend_never final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
};

task f() {
  co_return;
}

// CHECK-LABEL: define{{.*}} void @_Z1fv(
// CHECK: %[[RESULT_ACTIVE:.+]] = alloca i1
// CHECK: store i1 false, ptr %[[RESULT_ACTIVE]]
// CHECK: invoke void @_ZN4task12promise_type17get_return_objectEv(
// CHECK: store i1 true, ptr %[[RESULT_ACTIVE]]
// CHECK: invoke void @_ZN4task12promise_type15initial_suspendEv(
// CHECK: init.ready:
// CHECK-NEXT: store i1 false, ptr %[[RESULT_ACTIVE]]
// CHECK-NEXT: call void @_ZNSt13suspend_never12await_resumeEv(
// CHECK: %[[IS_ACTIVE:.+]] = load i1, ptr %[[RESULT_ACTIVE]]
// CHECK-NEXT: br i1 %[[IS_ACTIVE]], label %[[CLEANUP_ACTION:.+]], label %[[CLEANUP_DONE:.+]]
// CHECK: [[CLEANUP_ACTION]]:
// CHECK-NEXT: call void @_ZN4taskD1Ev(
