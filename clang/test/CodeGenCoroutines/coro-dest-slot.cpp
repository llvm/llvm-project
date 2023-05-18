// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

using namespace std;

struct coro {
  struct promise_type {
    coro get_return_object();
    suspend_always initial_suspend();
    suspend_never final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
  };
};

extern "C" coro f(int) { co_return; }
// Verify that cleanup.dest.slot is eliminated in a coroutine.
// CHECK-LABEL: f(
// CHECK: %[[INIT_SUSPEND:.+]] = call i8 @llvm.coro.suspend(
// CHECK-NEXT: switch i8 %[[INIT_SUSPEND]], label
// CHECK-NEXT:   i8 0, label %[[INIT_READY:.+]]
// CHECK-NEXT:   i8 1, label %[[INIT_CLEANUP:.+]]
// CHECK-NEXT: ]
// CHECK: %[[CLEANUP_DEST0:.+]] = phi i32 [ 0, %[[INIT_READY]] ], [ 2, %[[INIT_CLEANUP]] ]

// CHECK: %[[FINAL_SUSPEND:.+]] = call i8 @llvm.coro.suspend(
// CHECK-NEXT: switch i8 %{{.*}}, label %coro.ret [
// CHECK-NEXT:   i8 0, label %[[FINAL_READY:.+]]
// CHECK-NEXT:   i8 1, label %[[FINAL_CLEANUP:.+]]
// CHECK-NEXT: ]

// CHECK: call void @_ZNSt13suspend_never12await_resumeEv(
// CHECK: %[[CLEANUP_DEST1:.+]] = phi i32 [ 0, %[[FINAL_READY]] ], [ 2, %[[FINAL_CLEANUP]] ]
// CHECK: %[[CLEANUP_DEST2:.+]] = phi i32 [ %[[CLEANUP_DEST0]], %{{.+}} ], [ %[[CLEANUP_DEST1]], %{{.+}} ], [ 0, %{{.+}} ]
// CHECK: call ptr @llvm.coro.free(
// CHECK: switch i32 %[[CLEANUP_DEST2]], label %{{.+}} [
// CHECK-NEXT: i32 0
// CHECK-NEXT: i32 2
// CHECK-NEXT: ]
