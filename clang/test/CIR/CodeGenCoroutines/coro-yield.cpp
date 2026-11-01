// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -Wno-coroutine-missing-unhandled-exception %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

#include "Inputs/coroutine.h"

struct YieldAwaiter {
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<>) {}
  int &await_resume() { return value; }
  int value;
};

struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    YieldAwaiter yield_value(int) { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z17test_yield_assignv
// OGCG-LABEL: define dso_local void @_Z17test_yield_assignv
Task test_yield_assign() {
  // CIR: %[[VAL:.*]] = cir.const #cir.int<42> : !s32i
  // CIR: cir.await(yield, ready : {
  // CIR:   %[[READY:.*]] = cir.call @_ZN12YieldAwaiter11await_readyEv(%{{.*}}) : (!cir.ptr<!rec_YieldAwaiter>{{.*}}) -> (!cir.bool{{.*}})
  // CIR:   cir.condition(%[[READY]])
  // CIR: }, suspend : {
  // CIR:   cir.call @_ZN12YieldAwaiter13await_suspendESt16coroutine_handleIvE(%{{.*}})
  // CIR:   cir.coro.suspend_point
  // CIR: }, resume : {
  // CIR:   %[[RES:.*]] = cir.call @_ZN12YieldAwaiter12await_resumeEv(%{{.*}}) : (!cir.ptr<!rec_YieldAwaiter>{{.*}}) -> (!cir.ptr<!s32i>{{.*}})
  // CIR:   cir.store %[[RES]], %[[TMP_ADDR:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR:   cir.yield
  // CIR: },)
  // CIR: %[[LOADED_PTR:.*]] = cir.load %[[TMP_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.store align(4) %[[VAL]], %[[LOADED_PTR]] : !s32i, !cir.ptr<!s32i>

  // OGCG: %[[RES_OGCG:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @_ZN12YieldAwaiter12await_resumeEv(ptr {{.*}})
  // OGCG: store i32 42, ptr %[[RES_OGCG]], align 4
  (co_yield 0) = 42;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z14test_yield_refv
// OGCG-LABEL: define dso_local void @_Z14test_yield_refv
Task test_yield_ref() {
  // CIR: %[[R_ADDR:.*]] = cir.alloca "r" align(8) init const : !cir.ptr<!cir.ptr<!s32i>>
  // CIR: cir.await(yield, ready : {
  // CIR:   %[[READY:.*]] = cir.call @_ZN12YieldAwaiter11await_readyEv(%{{.*}}) : (!cir.ptr<!rec_YieldAwaiter>{{.*}}) -> (!cir.bool{{.*}})
  // CIR:   cir.condition(%[[READY]])
  // CIR: }, suspend : {
  // CIR:   cir.call @_ZN12YieldAwaiter13await_suspendESt16coroutine_handleIvE(%{{.*}})
  // CIR:   cir.coro.suspend_point
  // CIR: }, resume : {
  // CIR:   %[[RES:.*]] = cir.call @_ZN12YieldAwaiter12await_resumeEv(%{{.*}}) : (!cir.ptr<!rec_YieldAwaiter>{{.*}}) -> (!cir.ptr<!s32i>{{.*}})
  // CIR:   cir.store %[[RES]], %[[TMP_ADDR:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR:   cir.yield
  // CIR: },)
  // CIR: %[[LOADED_PTR:.*]] = cir.load %[[TMP_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.store align(8) %[[LOADED_PTR]], %[[R_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

  // OGCG: %[[R_ADDR_OGCG:.*]] = alloca ptr, align 8
  // OGCG: call noundef nonnull align 4 dereferenceable(4) ptr @_ZN12YieldAwaiter12await_resumeEv(ptr {{.*}})
  // OGCG: store ptr %{{.*}}, ptr %[[R_ADDR_OGCG]], align 8
  int &r = co_yield 0;
}
