// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir %s -check-prefix=CIR-FLAT --implicit-check-not=cir.coroutine --implicit-check-not=cir.await --implicit-check-not=cir.co_return --implicit-check-not=cir.coro.exit --implicit-check-not=cir.coro.suspend_point
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -Wno-coroutine-missing-unhandled-exception %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

#include "Inputs/coroutine.h"

struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

struct BoolAwaiter {
  bool await_ready() { return false; }
  bool await_suspend(std::coroutine_handle<>) { return false; }
  void await_resume() {}
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z15await_bool_vetov
// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z15await_bool_vetov
// OGCG-LABEL: define dso_local void @_Z15await_bool_vetov
Task await_bool_veto() {
  // CIR: cir.await(user, ready : {
  // CIR:   %[[READY:.*]] = cir.call @_ZN11BoolAwaiter11await_readyEv(%{{.*}}) : (!cir.ptr<!rec_BoolAwaiter>{{.*}}) -> (!cir.bool{{.*}})
  // CIR:   cir.condition(%[[READY]])
  // CIR: }, suspend : {
  // CIR:   %[[SUSPEND_RET:.*]] = cir.call @_ZN11BoolAwaiter13await_suspendESt16coroutine_handleIvE(%{{.*}}) : (!cir.ptr<!rec_BoolAwaiter>{{.*}}) -> (!cir.bool{{.*}})
  // CIR:   cir.coro.suspend_point(%[[SUSPEND_RET]])
  // CIR: }, resume : {
  // CIR:   cir.call @_ZN11BoolAwaiter12await_resumeEv(%{{.*}}) : (!cir.ptr<!rec_BoolAwaiter>{{.*}}) -> ()
  // CIR:   cir.yield
  // CIR: },)

  // CIR-FLAT:   %[[READY:.*]] = cir.call @_ZN11BoolAwaiter11await_readyEv(%[[AWAITER:[0-9]+]])
  // CIR-FLAT-NEXT:   cir.brcond %[[READY]] ^[[RESUME:bb[0-9]+]], ^[[SUSPEND:.*]]
  // CIR-FLAT: ^[[SUSPEND]]:
  // CIR-FLAT:   %[[SUSPEND_RET:.*]] = cir.call @_ZN11BoolAwaiter13await_suspendESt16coroutine_handleIvE(%[[AWAITER]])
  // CIR-FLAT-NEXT:   cir.brcond %[[SUSPEND_RET]] ^[[DO_SUSPEND:.*]], ^[[RESUME]]
  // CIR-FLAT: ^[[DO_SUSPEND]]:
  // CIR-FLAT:   %[[RES:.*]] = cir.coro.intrinsic.suspend
  // CIR-FLAT-NEXT:   cir.switch.flat %[[RES]] : !s8i, ^[[SUSPENDED:bb[0-9]+]] [
  // CIR-FLAT-NEXT:     0: ^[[RESUME]],
  // CIR-FLAT-NEXT:     1: ^[[DESTROY:.*]]
  // CIR-FLAT-NEXT:   ]
  // CIR-FLAT: ^[[RESUME]]:
  // CIR-FLAT-NEXT:   cir.call @_ZN11BoolAwaiter12await_resumeEv(%[[AWAITER]])

  // OGCG:   %[[READY_RES:.*]] = call noundef zeroext i1 @_ZN11BoolAwaiter11await_readyEv(ptr {{.*}})
  // OGCG:   br i1 %[[READY_RES]], label %[[AWAIT_READY_DEST:.*]], label %[[AWAIT_SUSPEND:.*]]
  // OGCG: [[AWAIT_SUSPEND]]:
  // OGCG:   %[[SUSP_RET:.*]] = call i1 @llvm.coro.await.suspend.bool(ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
  // OGCG:   br i1 %[[SUSP_RET]], label %{{.*}}, label %[[AWAIT_READY_DEST]]
  co_await BoolAwaiter{};
}
