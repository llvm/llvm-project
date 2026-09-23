// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -Wno-coroutine-missing-unhandled-exception %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

#include "Inputs/coroutine.h"

struct B {
  int x;
  int y;
  bool await_ready() { return true; }
  B await_resume() { return {}; }
  template <typename F> void await_suspend(F) {}
};

struct coro_t {
  struct promise_type {
    coro_t get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    static void unhandled_exception() {}
    B yield_value(int) { return {}; }
  };
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z22aggregate_coawait_exprv
// OGCG-LABEL: define dso_local void @_Z22aggregate_coawait_exprv
coro_t aggregate_coawait_expr() {
  // CIR: %[[VAL:.*]] = cir.alloca "val" align(4) init : !cir.ptr<!rec_B>
  // CIR: cir.await(user, ready : {
  // CIR: }, suspend : {
  // CIR: }, resume : {
  // CIR:   cir.call @_ZN1B12await_resumeEv(%{{.*}})
  // CIR:   cir.store align(4) %{{.*}}, %[[VAL]] : !rec_B, !cir.ptr<!rec_B>
  // CIR:   cir.yield
  // CIR: },)
  // OGCG: %[[VAL:.*]] = alloca %struct.B, align 4
  // OGCG: await.ready:
  // OGCG:   %[[RES:.*]] = call i64 @_ZN1B12await_resumeEv(ptr {{.*}})
  // OGCG:   store i64 %[[RES]], ptr %[[VAL]], align 4
  B val = co_await B{};
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z29aggregate_coawait_expr_unusedv
// OGCG-LABEL: define dso_local void @_Z29aggregate_coawait_expr_unusedv
coro_t aggregate_coawait_expr_unused() {
  // CIR: cir.await(user, ready : {
  // CIR: }, suspend : {
  // CIR: }, resume : {
  // CIR:   cir.call @_ZN1B12await_resumeEv(%{{.*}})
  // CIR:   cir.yield
  // CIR: },)
  // OGCG: await.ready:
  // OGCG:   %[[RES:.*]] = call i64 @_ZN1B12await_resumeEv(ptr {{.*}})
  // OGCG:   store i64 %[[RES]], ptr %{{.*}}, align 4
  co_await B{};
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z22aggregate_coyield_exprv
// OGCG-LABEL: define dso_local void @_Z22aggregate_coyield_exprv
coro_t aggregate_coyield_expr() {
  // CIR: %[[VAL:.*]] = cir.alloca "val" align(4) init : !cir.ptr<!rec_B>
  // CIR: cir.await(yield, ready : {
  // CIR: }, suspend : {
  // CIR: }, resume : {
  // CIR:   cir.call @_ZN1B12await_resumeEv(%{{.*}})
  // CIR:   cir.store align(4) %{{.*}}, %[[VAL]] : !rec_B, !cir.ptr<!rec_B>
  // CIR:   cir.yield
  // CIR: },)
  // OGCG: %[[VAL:.*]] = alloca %struct.B, align 4
  // OGCG: yield.ready:
  // OGCG:   %[[RES:.*]] = call i64 @_ZN1B12await_resumeEv(ptr {{.*}})
  // OGCG:   store i64 %[[RES]], ptr %[[VAL]], align 4
  B val = co_yield 42;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z29aggregate_coyield_expr_unusedv
// OGCG-LABEL: define dso_local void @_Z29aggregate_coyield_expr_unusedv
coro_t aggregate_coyield_expr_unused() {
  // CIR: cir.await(yield, ready : {
  // CIR: }, suspend : {
  // CIR: }, resume : {
  // CIR:   cir.call @_ZN1B12await_resumeEv(%{{.*}})
  // CIR:   cir.yield
  // CIR: },)
  // OGCG: yield.ready:
  // OGCG:   %[[RES:.*]] = call i64 @_ZN1B12await_resumeEv(ptr {{.*}})
  // OGCG:   store i64 %[[RES]], ptr %{{.*}}, align 4
  co_yield 42;
}
