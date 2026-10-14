// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -fcxx-exceptions -fexceptions -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.eh.cir
// RUN: FileCheck --input-file=%t.eh.cir %s -check-prefix=CIR-EH
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
  // CIR:   %[[RES:.*]] = cir.call @_ZN1B12await_resumeEv(%{{.*}})
  // CIR:   cir.store %[[RES]], %[[TMP:.*]] : !u64i, !cir.ptr<!u64i>
  // CIR:   %[[CAST:.*]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!u64i> -> !cir.ptr<!rec_B>
  // CIR:   %[[LOAD:.*]] = cir.load %[[CAST]] : !cir.ptr<!rec_B>, !rec_B
  // CIR:   cir.store align(4) %[[LOAD]], %[[VAL]] : !rec_B, !cir.ptr<!rec_B>
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
  // CIR:   %[[RES:.*]] = cir.call @_ZN1B12await_resumeEv(%{{.*}})
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
  // CIR:   %[[RES:.*]] = cir.call @_ZN1B12await_resumeEv(%{{.*}})
  // CIR:   cir.store %[[RES]], %[[TMP:.*]] : !u64i, !cir.ptr<!u64i>
  // CIR:   %[[CAST:.*]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!u64i> -> !cir.ptr<!rec_B>
  // CIR:   %[[LOAD:.*]] = cir.load %[[CAST]] : !cir.ptr<!rec_B>, !rec_B
  // CIR:   cir.store align(4) %[[LOAD]], %[[VAL]] : !rec_B, !cir.ptr<!rec_B>
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
  // CIR:   %[[RES:.*]] = cir.call @_ZN1B12await_resumeEv(%{{.*}})
  // CIR:   cir.yield
  // CIR: },)

  // OGCG: yield.ready:
  // OGCG:   %[[RES:.*]] = call i64 @_ZN1B12await_resumeEv(ptr {{.*}})
  // OGCG:   store i64 %[[RES]], ptr %{{.*}}, align 4
  co_yield 42;
}

struct NT {
  int x;
  ~NT();
};

struct AwaitNT {
  bool await_ready() { return true; }
  NT await_resume();
  template <typename F> void await_suspend(F) {}
};

void use(int);
void consume(NT);
void consume2(NT, NT);

struct coro_nt_t {
  struct promise_type {
    coro_nt_t get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_value(NT) {}
    static void unhandled_exception() {}
    AwaitNT yield_value(int) { return {}; }
  };
};

// The cleanup for a result with a non-trivial destructor starts right after
// the cir.await and ends with the full-expression.

// CIR-LABEL: cir.func coroutine {{.*}} @_Z18discard_nontrivialv
// CIR:   %[[TMP:.*]] = cir.alloca "agg.tmp.ensured" align(4) : !cir.ptr<!rec_NT>
// CIR:   cir.await(user, ready : {
// CIR:   }, resume : {
// CIR-NEXT: cir.call @_ZN7AwaitNT12await_resumeEv(%[[TMP]], %{{.*}})
// CIR-NEXT: cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR-NEXT:   cir.yield
// CIR-NEXT: } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[TMP]])
// CIR-NEXT:   cir.yield
// CIR-NEXT: }

// With exceptions the cleanup is also an EH cleanup, and await_resume() itself
// stays outside of it.
// CIR-EH-LABEL: cir.func coroutine {{.*}} @_Z18discard_nontrivialv
// CIR-EH:   cir.await(user, ready : {
// CIR-EH:   }, resume : {
// CIR-EH-NEXT: cir.call @_ZN7AwaitNT12await_resumeEv(%[[TMP:.*]], %{{.*}})
// CIR-EH-NEXT: cir.yield
// CIR-EH-NEXT: },)
// CIR-EH-NEXT: cir.cleanup.scope {
// CIR-EH-NEXT:   cir.yield
// CIR-EH-NEXT: } cleanup all {
// CIR-EH-NEXT:   cir.call @_ZN2NTD1Ev(%[[TMP]])

// OGCG-LABEL: define dso_local void @_Z18discard_nontrivialv
// OGCG: await.cleanup:
// OGCG-NEXT: br label %[[CLEANUP:.*]]
// OGCG: await.ready:
// OGCG-NEXT: call void @_ZN7AwaitNT12await_resumeEv(ptr {{.*}} sret(%struct.NT) {{.*}} %[[TMP:.*]], ptr
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[TMP]])
// OGCG-NEXT: br label %[[CLEANUP]]
coro_t discard_nontrivial() { co_await AwaitNT{}; }

// A discarded result is destroyed at the end of the full-expression.
// CIR-LABEL: cir.func coroutine {{.*}} @_Z16comma_nontrivialv
// CIR:   cir.await(user, ready : {
// CIR:   }, resume : {
// CIR-NEXT: cir.call @_ZN7AwaitNT12await_resumeEv(%[[TMP:.*]], %{{.*}})
// CIR-NEXT: cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:   cir.call @_Z3usei(%[[ONE]])
// CIR-NEXT:   cir.yield
// CIR-NEXT: } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[TMP]])

// OGCG-LABEL: define dso_local void @_Z16comma_nontrivialv
// OGCG: await.ready:
// OGCG-NEXT: call void @_ZN7AwaitNT12await_resumeEv(ptr {{.*}} %[[TMP:.*]], ptr
// OGCG-NEXT: call void @_Z3usei(i32 noundef 1)
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[TMP]])
coro_t comma_nontrivial() { (co_await AwaitNT{}, use(1)); }

// CIR-LABEL: cir.func coroutine {{.*}} @_Z16byval_nontrivialv
// CIR:   cir.await(user, ready : {
// CIR:   }, resume : {
// CIR-NEXT: cir.call @_ZN7AwaitNT12await_resumeEv(%[[ARG:.*]], %{{.*}})
// CIR-NEXT: cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR-NEXT:   cir.call @_Z7consume2NT(%[[ARG]])
// CIR-NEXT:   cir.yield
// CIR-NEXT: } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[ARG]])

// OGCG-LABEL: define dso_local void @_Z16byval_nontrivialv
// OGCG: await.ready:
// OGCG-NEXT: call void @_ZN7AwaitNT12await_resumeEv(ptr {{.*}} %[[ARG:.*]], ptr
// OGCG-NEXT: call void @_Z7consume2NT(ptr {{.*}} %[[ARG]])
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[ARG]])
coro_t byval_nontrivial() { consume(co_await AwaitNT{}); }

// The first argument is alive while the second co_await is suspended, so its
// cleanup encloses the second cir.await; the second one's does not. They are
// destroyed in reverse order.
// CIR-LABEL: cir.func coroutine {{.*}} @_Z19two_args_nontrivialv
// CIR:   cir.await(user, ready : {
// CIR:   }, resume : {
// CIR-NEXT: cir.call @_ZN7AwaitNT12await_resumeEv(%[[A0:.*]], %{{.*}})
// CIR-NEXT: cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR:        cir.await(user, ready : {
// CIR:        }, resume : {
// CIR-NEXT:     cir.call @_ZN7AwaitNT12await_resumeEv(%[[A1:.*]], %{{.*}})
// CIR-NEXT:     cir.yield
// CIR-NEXT:   },)
// CIR-NEXT:   cir.cleanup.scope {
// CIR-NEXT:     cir.call @_Z8consume22NTS_(%[[A0]], %[[A1]])
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } cleanup normal {
// CIR-NEXT:     cir.call @_ZN2NTD1Ev(%[[A1]])
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }
// CIR-NEXT:   cir.yield
// CIR-NEXT: } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[A0]])

// CIR-EH-LABEL: cir.func coroutine {{.*}} @_Z19two_args_nontrivialv
// CIR-EH:   cir.call @_ZN7AwaitNT12await_resumeEv(%[[A0:[0-9]+]], %{{.*}})
// CIR-EH-NEXT: cir.yield
// CIR-EH-NEXT: },)
// CIR-EH-NEXT: cir.cleanup.scope {
// CIR-EH-NEXT:   cir.await(user, ready : {
// CIR-EH:          cir.call @_ZN7AwaitNT12await_resumeEv(%[[A1:[0-9]+]], %{{.*}})
// CIR-EH-NEXT:     cir.yield
// CIR-EH-NEXT:   },)
// CIR-EH-NEXT:   cir.cleanup.scope {
// CIR-EH-NEXT:     cir.call @_Z8consume22NTS_(%[[A0]], %[[A1]])
// CIR-EH-NEXT:     cir.yield
// CIR-EH-NEXT:   } cleanup all {
// CIR-EH-NEXT:     cir.call @_ZN2NTD1Ev(%[[A1]])
// CIR-EH-NEXT:     cir.yield
// CIR-EH-NEXT:   }
// CIR-EH-NEXT:   cir.yield
// CIR-EH-NEXT: } cleanup all {
// CIR-EH-NEXT:   cir.call @_ZN2NTD1Ev(%[[A0]])

// OGCG-LABEL: define dso_local void @_Z19two_args_nontrivialv
// OGCG: await.cleanup:
// OGCG-NEXT: br label %[[OUTER:.*]]
// OGCG: await2.cleanup:
// OGCG-NEXT: br label %[[INNER:.*]]
// OGCG: await2.ready:
// OGCG: call void @_Z8consume22NTS_(ptr {{.*}} %[[A0:.*]], ptr {{.*}} %[[A1:.*]])
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[A1]])
// OGCG-NEXT: br label %[[INNER]]
// OGCG: [[INNER]]:
// OGCG-NOT: br
// OGCG: call void @_ZN2NTD1Ev(ptr {{.*}} %[[A0]])
// OGCG-NEXT: br label %[[OUTER]]
coro_t two_args_nontrivial() { consume2(co_await AwaitNT{}, co_await AwaitNT{}); }

// co_yield, and a promise call taking the result by value.
// CIR-LABEL: cir.func coroutine {{.*}} @_Z27coreturn_coyield_nontrivialv
// CIR:   cir.await(yield, ready : {
// CIR:   }, resume : {
// CIR-NEXT: cir.call @_ZN7AwaitNT12await_resumeEv(%[[ARG:.*]], %{{.*}})
// CIR-NEXT: cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR-NEXT:   cir.call @_ZN9coro_nt_t12promise_type12return_valueE2NT(%{{.*}}, %[[ARG]])
// CIR-NEXT:   cir.yield
// CIR-NEXT: } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[ARG]])

// OGCG-LABEL: define dso_local void @_Z27coreturn_coyield_nontrivialv
// OGCG: yield.ready:
// OGCG-NEXT: call void @_ZN7AwaitNT12await_resumeEv(ptr {{.*}} %[[ARG:.*]], ptr
// OGCG-NEXT: call void @_ZN9coro_nt_t12promise_type12return_valueE2NT(ptr {{.*}}, ptr {{.*}} %[[ARG]])
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[ARG]])
coro_nt_t coreturn_coyield_nontrivial() { co_return co_yield 42; }

// Inside a conditional the destructor is guarded by a flag, which is set once
// the cir.await has completed.
// CIR-LABEL: cir.func coroutine {{.*}} @_Z23cond_discard_nontrivialb
// CIR:   %[[FLAG:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.cleanup.scope {
// CIR-NEXT: %[[FALSE:.*]] = cir.const #false
// CIR-NEXT: cir.store %[[FALSE]], %[[FLAG]] : !cir.bool
// CIR:     cir.ternary(%{{.*}}, true {
// CIR:       cir.await(user, ready : {
// CIR:       }, resume : {
// CIR-NEXT:    cir.call @_ZN7AwaitNT12await_resumeEv(%[[TMP:.*]], %{{.*}})
// CIR-NEXT:    cir.yield
// CIR-NEXT:  },)
// CIR-NEXT:  %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:  cir.store %[[TRUE]], %[[FLAG]] : !cir.bool
// CIR:     } cleanup normal {
// CIR-NEXT:  %[[ACTIVE:.*]] = cir.load {{.*}} %[[FLAG]] :
// CIR-NEXT:  cir.if %[[ACTIVE]] {
// CIR-NEXT:    cir.call @_ZN2NTD1Ev(%[[TMP]])

// OGCG-LABEL: define dso_local void @_Z23cond_discard_nontrivialb
// OGCG: store i1 false, ptr %[[FLAG:cleanup.cond]]
// OGCG: await.cleanup:
// OGCG-NEXT: br label %[[CLEANUP:.*]]
// OGCG: await.ready:
// OGCG-NEXT: call void @_ZN7AwaitNT12await_resumeEv(ptr {{.*}} %[[TMP:.*]], ptr
// OGCG-NEXT: store i1 true, ptr %[[FLAG]]
// OGCG: cleanup.action:
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[TMP]])
coro_t cond_discard_nontrivial(bool c) { c ? (void)co_await AwaitNT{} : (void)0; }

// A destination that is destroyed by its owner gets no extra temporary or
// cleanup: the variable's own cleanup is the only destructor call.
// CIR-LABEL: cir.func coroutine {{.*}} @_Z19var_init_nontrivialv
// CIR-NOT: agg.tmp.ensured
// CIR:   %[[V:.*]] = cir.alloca "v" align(4) init : !cir.ptr<!rec_NT>
// CIR-NOT: agg.tmp.ensured
// CIR:   cir.await(user, ready : {
// CIR:   }, resume : {
// CIR-NEXT: cir.call @_ZN7AwaitNT12await_resumeEv(%[[V]], %{{.*}})
// CIR-NEXT: cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR-NOT:    @_ZN2NTD1Ev
// CIR:      } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[V]])
// CIR-NOT:  @_ZN2NTD1Ev
// CIR:      }, finalSuspend : {

// OGCG-LABEL: define dso_local void @_Z19var_init_nontrivialv
// OGCG-NOT: agg.tmp.ensured
// OGCG: await.ready:
// OGCG-NEXT: call void @_ZN7AwaitNT12await_resumeEv(ptr {{.*}} %[[V:v]], ptr
// OGCG-NOT: @_ZN2NTD1Ev
// OGCG: call void @_Z3usei(
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[V]])
// OGCG-NOT: @_ZN2NTD1Ev
// OGCG: coro.final:
coro_t var_init_nontrivial() { NT v = co_await AwaitNT{}; use(v.x); }

bool check(NT);

// In a loop condition the full-expression's cleanups are done before the
// cir.condition.
// CIR-LABEL: cir.func coroutine {{.*}} @_Z20loop_cond_nontrivialv
// CIR:   cir.while {
// CIR:     cir.await(user, ready : {
// CIR:     }, resume : {
// CIR-NEXT:  cir.call @_ZN7AwaitNT12await_resumeEv(%[[ARG:.*]], %{{.*}})
// CIR-NEXT:  cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR-NEXT:   %[[CHECK:.*]] = cir.call @_Z5check2NT(%[[ARG]])
// CIR-NEXT:   cir.store {{.*}} %[[CHECK]], %[[SPILL:.*]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT:   cir.yield
// CIR-NEXT: } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[ARG]])
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: %[[COND:.*]] = cir.load {{.*}} %[[SPILL]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.condition(%[[COND]])

// OGCG-LABEL: define dso_local void @_Z20loop_cond_nontrivialv
// OGCG: await.ready:
// OGCG-NEXT: call void @_ZN7AwaitNT12await_resumeEv(ptr {{.*}} %[[ARG:.*]], ptr
// OGCG-NEXT: %[[CHECK:.*]] = call {{.*}} i1 @_Z5check2NT(ptr {{.*}} %[[ARG]])
// OGCG-NEXT: store i1 %[[CHECK]], ptr %[[SPILL:.*]], align 1
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[ARG]])
coro_t loop_cond_nontrivial() {
  while (check(co_await AwaitNT{})) {
  }
}

struct InitAwaitNT {
  bool await_ready() { return true; }
  void await_suspend(std::coroutine_handle<>) {}
  NT await_resume();
};

struct FinalAwaitNT {
  bool await_ready() noexcept { return true; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  NT await_resume() noexcept;
};

struct coro_init_final_nt_t {
  struct promise_type {
    coro_init_final_nt_t get_return_object() { return {}; }
    InitAwaitNT initial_suspend() { return {}; }
    FinalAwaitNT final_suspend() noexcept { return {}; }
    void return_void() {}
    static void unhandled_exception() {}
  };
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z21init_final_nontrivialv
// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, resume : {
// CIR-NEXT: cir.call @_ZN11InitAwaitNT12await_resumeEv(%[[INIT:.*]], %{{.*}})
// CIR-NEXT: cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR-NEXT:   cir.yield
// CIR-NEXT: } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[INIT]])
// CIR: }, finalSuspend : {
// CIR:   cir.await(final, ready : {
// CIR:   }, resume : {
// CIR-NEXT: cir.call @_ZN12FinalAwaitNT12await_resumeEv(%[[FINAL:.*]], %{{.*}})
// CIR-NEXT: cir.yield
// CIR-NEXT: },)
// CIR-NEXT: cir.cleanup.scope {
// CIR-NEXT:   cir.yield
// CIR-NEXT: } cleanup normal {
// CIR-NEXT:   cir.call @_ZN2NTD1Ev(%[[FINAL]])

// With exceptions, the initial await_resume() is emitted in a try/catch that
// owns and destroys the result; nothing is pushed after the cir.await.
// CIR-EH-LABEL: cir.func coroutine {{.*}} @_Z21init_final_nontrivialv
// CIR-EH: cir.coroutine initialSuspend : {
// CIR-EH:   cir.await(init, ready : {
// CIR-EH:   }, resume : {
// CIR-EH:     cir.try {
// CIR-EH-NEXT:  cir.call @_ZN11InitAwaitNT12await_resumeEv(%[[INIT:.*]], %{{.*}})
// CIR-EH-NEXT:  cir.cleanup.scope {
// CIR-EH:       } cleanup all {
// CIR-EH-NEXT:    cir.call @_ZN2NTD1Ev(%[[INIT]])
// CIR-EH:     } catch all
// CIR-EH:   },)
// CIR-EH-NEXT: cir.yield
// CIR-EH-NEXT: }, body : {
// CIR-EH:   }, finalSuspend : {
// CIR-EH:   cir.await(final, ready : {
// CIR-EH:   }, resume : {
// CIR-EH-NEXT: cir.call @_ZN12FinalAwaitNT12await_resumeEv(%[[FINAL:.*]], %{{.*}})
// CIR-EH-NEXT: cir.yield
// CIR-EH-NEXT: },)
// CIR-EH-NEXT: cir.cleanup.scope {
// CIR-EH-NEXT:   cir.yield
// CIR-EH-NEXT: } cleanup all {
// CIR-EH-NEXT:   cir.call @_ZN2NTD1Ev(%[[FINAL]])

// OGCG-LABEL: define dso_local void @_Z21init_final_nontrivialv
// OGCG: init.ready:
// OGCG-NEXT: call void @_ZN11InitAwaitNT12await_resumeEv(ptr {{.*}} %[[INIT:.*]], ptr
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[INIT]])
// OGCG: final.ready:
// OGCG-NEXT: call void @_ZN12FinalAwaitNT12await_resumeEv(ptr {{.*}} %[[FINAL:.*]], ptr
// OGCG-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} %[[FINAL]])
coro_init_final_nt_t init_final_nontrivial() { co_return; }
