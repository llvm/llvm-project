// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++14 -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

using namespace std::experimental;

struct coro {
  struct promise_type {
    coro get_return_object();
    suspend_always initial_suspend();
    suspend_never final_suspend();
    void return_void();
    static void unhandled_exception();
  };
};

extern "C" coro f(int) { co_return; }
// Verify that cleanup.dest.slot is eliminated in a coroutine.
// CHECK-LABEL: f(
// CHECK: call void @_ZNSt12experimental13coroutines_v113suspend_never12await_resumeEv(
// CHECK: %[[CLEANUP_DEST:.+]] = phi i32 [ 0, %{{.+}} ], [ 2, %{{.+}} ], [ 2, %{{.+}} ]
// CHECK: call i8* @llvm.coro.free(
// CHECK: switch i32 %cleanup.dest.slot.0, label %{{.+}} [
// CHECK-NEXT: i32 0
// CHECK-NEXT: i32 2
// CHECK-NEXT: ]
