// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-COMMON,CIR-NOEH,RESUME,RESUME-NOEH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -fcxx-exceptions -fexceptions -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.eh.cir
// RUN: FileCheck --input-file=%t.eh.cir %s -check-prefixes=CIR-EH,CIR-COMMON,RESUME,RESUME-EH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -Wno-coroutine-missing-unhandled-exception %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=OGCG,RESUME-IR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm -disable-llvm-passes -Wno-coroutine-missing-unhandled-exception %s -o %t.eh.ll
// RUN: FileCheck --input-file=%t.eh.ll %s -check-prefixes=OGCG-EH,RESUME-IR

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
  // CIR:   cir.copy %[[CAST]] align(8) to %[[VAL]] align(4) : !cir.ptr<!rec_B>
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
  // CIR:   cir.copy %[[CAST]] align(8) to %[[VAL]] align(4) : !cir.ptr<!rec_B>
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

// With exceptions, destroy the initial resume result in the try block before
// setting the flag that allows the body to run.
// CIR-EH-LABEL: cir.func coroutine {{.*}} @_Z21init_final_nontrivialv
// CIR-EH: %[[RESUME_EH:.*]] = cir.alloca "resume.eh"
// CIR-EH: cir.coroutine initialSuspend : {
// CIR-EH:   cir.await(init, ready : {
// CIR-EH:   }, resume : {
// CIR-EH:     cir.try {
// CIR-EH-NEXT:  cir.call @_ZN11InitAwaitNT12await_resumeEv(%[[INIT:.*]], %{{.*}})
// CIR-EH-NEXT:  cir.cleanup.scope {
// CIR-EH-NEXT:    cir.yield
// CIR-EH-NEXT:  } cleanup all {
// CIR-EH-NEXT:    cir.call @_ZN2NTD1Ev(%[[INIT]])
// CIR-EH-NEXT:    cir.yield
// CIR-EH-NEXT:  }
// CIR-EH-NEXT:  %[[TRUE:.*]] = cir.const #true
// CIR-EH-NEXT:  cir.store %[[TRUE]], %[[RESUME_EH]]
// CIR-EH-NEXT:  cir.yield
// CIR-EH-NEXT: } catch all
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

// OGCG-EH-LABEL: define {{.*}} void @_Z21init_final_nontrivialv(
// OGCG-EH:       init.ready:
// OGCG-EH-NEXT:    store i1 true, ptr %[[RESUME_EH:.*]], align 1
// OGCG-EH-NEXT:    invoke void @_ZN11InitAwaitNT12await_resumeEv(ptr {{.*}} %[[INIT:.*]], ptr
// OGCG-EH-NEXT:      to label %[[CONT:.*]] unwind label
// OGCG-EH:       [[CONT]]:
// OGCG-EH-NEXT:    call void @_ZN2NTD1Ev(ptr {{.*}} %[[INIT]])
// OGCG-EH-NEXT:    store i1 false, ptr %[[RESUME_EH]], align 1
coro_init_final_nt_t init_final_nontrivial() { co_return; }

void next();

// The try/catch is only needed if the resume expression can throw: here
// neither await_resume() nor the result's destructor can.

struct NoexceptInitAw {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  NT await_resume() noexcept;
};

struct NoexceptInitTask {
  struct promise_type {
    NoexceptInitTask get_return_object() noexcept;
    NoexceptInitAw initial_suspend() noexcept;
    std::suspend_never final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

NoexceptInitTask noexcept_initial_resume() {
  next();
  co_return;
}

// CIR-COMMON-LABEL: cir.func coroutine {{.*}} @_Z23noexcept_initial_resumev(
// CIR-COMMON-NOT:       resume.eh
// CIR-COMMON:           %[[INIT:.*]] = cir.alloca "agg.tmp.ensured" align(4) : !cir.ptr<!rec_NT>
// CIR-COMMON-NOT:       resume.eh
// CIR-COMMON:           cir.await(init, ready : {
// CIR-COMMON:           }, resume : {
// CIR-COMMON-NEXT:        cir.call @_ZN14NoexceptInitAw12await_resumeEv(%[[INIT]], %{{.*}})
// CIR-COMMON-NEXT:        cir.yield
// CIR-COMMON-NEXT:      },)
// CIR-COMMON-NEXT:      cir.cleanup.scope {
// CIR-COMMON-NEXT:        cir.yield
// CIR-NOEH-NEXT: } cleanup normal {
// CIR-EH-NEXT:   } cleanup all {
// CIR-COMMON-NEXT:        cir.call @_ZN2NTD1Ev(%[[INIT]])
// CIR-COMMON-NEXT:        cir.yield
// CIR-COMMON-NEXT:      }
// CIR-COMMON-NEXT:      cir.yield
// CIR-COMMON-NEXT:    }, body : {
// CIR-COMMON-NOT:       resume.eh
// CIR-COMMON:         }, finalSuspend : {

// OGCG-EH-LABEL: define {{.*}} void @_Z23noexcept_initial_resumev(
// OGCG-EH-NOT:     initresume.eh
// OGCG-EH:       init.ready:
// OGCG-EH-NEXT:    call void @_ZN14NoexceptInitAw12await_resumeEv(ptr {{.*}} %[[INIT:.*]], ptr
// OGCG-EH-NEXT:    call void @_ZN2NTD1Ev(ptr {{.*}} %[[INIT]])
// OGCG-EH-NOT:     initresume.eh
// OGCG-EH:       define

// With a destructor that may throw, a noexcept await_resume() call is still in
// the try/catch, and the result is destroyed before the flag is set.

struct TD {
  int x;
  ~TD() noexcept(false);
};

struct ThrowingDtorInitAw {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  TD await_resume() noexcept;
};

struct ThrowingDtorInitTask {
  struct promise_type {
    ThrowingDtorInitTask get_return_object() noexcept;
    ThrowingDtorInitAw initial_suspend() noexcept;
    std::suspend_never final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

ThrowingDtorInitTask throwing_dtor_initial_resume() {
  next();
  co_return;
}

// CIR-COMMON-LABEL: cir.func coroutine {{.*}} @_Z28throwing_dtor_initial_resumev(
// CIR-EH:      %[[RESUME_EH:.*]] = cir.alloca "resume.eh"
// CIR-COMMON:         cir.await(init, ready : {
// CIR-COMMON:         }, resume : {
// CIR-EH:          cir.try {
// CIR-EH-NEXT:       cir.call @_ZN18ThrowingDtorInitAw12await_resumeEv(%[[INIT:.*]], %{{.*}})
// CIR-EH-NEXT:       cir.cleanup.scope {
// CIR-EH-NEXT:         cir.yield
// CIR-EH-NEXT:       } cleanup all {
// CIR-EH-NEXT:         cir.call @_ZN2TDD1Ev(%[[INIT]])
// CIR-EH-NEXT:         cir.yield
// CIR-EH-NEXT:       }
// CIR-EH-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-EH-NEXT:       cir.store %[[TRUE]], %[[RESUME_EH]]
// CIR-EH-NEXT:       cir.yield
// CIR-EH-NEXT:     } catch all
// CIR-COMMON:         }, body : {
// CIR-EH-NEXT:   %[[RESUMED:.*]] = cir.load {{.*}} %[[RESUME_EH]]
// CIR-EH-NEXT:   cir.if %[[RESUMED]] {
// CIR-EH:          cir.call @_Z4nextv()

// OGCG-EH-LABEL: define {{.*}} void @_Z28throwing_dtor_initial_resumev(
// OGCG-EH:       init.ready:
// OGCG-EH-NEXT:    store i1 true, ptr %[[RESUME_EH:.*]], align 1
// OGCG-EH-NEXT:    call void @_ZN18ThrowingDtorInitAw12await_resumeEv(ptr {{.*}} %[[INIT:.*]], ptr
// OGCG-EH-NEXT:    invoke void @_ZN2TDD1Ev(ptr {{.*}} %[[INIT]])
// OGCG-EH-NEXT:      to label %[[CONT:.*]] unwind label
// OGCG-EH:       [[CONT]]:
// OGCG-EH-NEXT:    store i1 false, ptr %[[RESUME_EH]], align 1

// Resume-expression temporaries survive through the enclosing full-expression.
// Construction guards exclude paths that destroy the coroutine while suspended;
// array elements and ready/suspend calls retain their own cleanup boundaries.

struct S {
  S(int);
  ~S();
};

struct Agg {
  S s;
  int i;
};

extern "C" bool resume_flag();
extern "C" int resume_arg(const S &);
extern "C" int resume_next();
extern "C" void resume_use(int);
extern "C" void resume_use_pair(int, int);
extern "C" void resume_use_result(NT);
extern "C" void resume_use_aggregate(const Agg &, int);
extern "C" void mark();

struct DA {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  int await_resume(const S & = S(1));
};

struct DV {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  int await_resume(S = S(2));
};

struct DCC {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  int await_resume(int = resume_flag() ? resume_arg(S(1)) : 0);
};

struct RA {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  NT await_resume(const S & = S(1));
};

struct PL {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  int await_resume();
};

struct Task {
  struct promise_type {
    Task get_return_object();
    std::suspend_never initial_suspend();
    std::suspend_never final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
    DA yield_value(int);
  };
};

// RESUME-LABEL: cir.func coroutine {{.*}} @defarg_condtemp(
// RESUME: %[[TMP:[^,]+]] = cir.alloca "ref.tmp{{[0-9]+}}" {{.*}} : !cir.ptr<!rec_S>
// RESUME: %[[FLAG:.*]] = cir.alloca "cleanup.cond"
// RESUME: %[[FALSE:.*]] = cir.const #false
// RESUME-NEXT: cir.store %[[FALSE]], %[[FLAG]]
// RESUME: cir.await(user, ready : {
// RESUME: }, resume : {
// RESUME: cir.ternary(
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP]],
// RESUME-NEXT: %[[TRUE:.*]] = cir.const #true
// RESUME-NEXT: cir.store %[[TRUE]], %[[FLAG]]
// RESUME: cir.call @_ZN3DCC12await_resumeEi(
// RESUME: },)
// RESUME: cir.call @resume_use(
// RESUME-NOEH: } cleanup normal {
// RESUME-EH: } cleanup all {
// RESUME: %[[LIVE:.*]] = cir.load {{.*}} %[[FLAG]]
// RESUME-NEXT: cir.if %[[LIVE]] {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[TMP]])

extern "C" Task defarg_condtemp() {
  resume_use(co_await DCC{});
}

// Cover reference and value arguments across two suspension points. Destroy
// them in reverse order at full-expression end, before the next statement.
// Construction flags exclude paths that destroy the coroutine while suspended.
// RESUME-LABEL: cir.func coroutine {{.*}} @two_awaits(
// RESUME-DAG: %[[TMP1:[^,]+]] = cir.alloca "ref.tmp{{[0-9]+}}" {{.*}} : !cir.ptr<!rec_S>
// RESUME-DAG: %[[FLAG1:.*]] = cir.alloca "cleanup.cond"
// RESUME-DAG: %[[TMP2:[^,]+]] = cir.alloca "agg.tmp{{[0-9]+}}" {{.*}} : !cir.ptr<!rec_S>
// RESUME-DAG: %[[FLAG2:.*]] = cir.alloca "cleanup.cond"
// RESUME: %[[FALSE1:.*]] = cir.const #false
// RESUME-NEXT: cir.store %[[FALSE1]], %[[FLAG1]]
// RESUME: cir.await(user, ready : {
// RESUME: }, resume : {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP1]],
// RESUME-NEXT: %[[TRUE1:.*]] = cir.const #true
// RESUME-NEXT: cir.store %[[TRUE1]], %[[FLAG1]]
// RESUME: cir.call @_ZN2DA12await_resumeERK1S(
// RESUME: },)
// RESUME: %[[FALSE2:.*]] = cir.const #false
// RESUME-NEXT: cir.store %[[FALSE2]], %[[FLAG2]]
// RESUME: cir.await(user, ready : {
// RESUME: }, resume : {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP2]],
// RESUME-NEXT: %[[TRUE2:.*]] = cir.const #true
// RESUME-NEXT: cir.store %[[TRUE2]], %[[FLAG2]]
// RESUME: cir.call @_ZN2DV12await_resumeE1S(
// RESUME: },)
// RESUME: cir.call @resume_use_pair(
// RESUME-NOEH: } cleanup normal {
// RESUME-EH: } cleanup all {
// RESUME: %[[LIVE2:.*]] = cir.load {{.*}} %[[FLAG2]]
// RESUME-NEXT: cir.if %[[LIVE2]] {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[TMP2]])
// RESUME-NOEH: } cleanup normal {
// RESUME-EH: } cleanup all {
// RESUME: %[[LIVE1:.*]] = cir.load {{.*}} %[[FLAG1]]
// RESUME-NEXT: cir.if %[[LIVE1]] {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[TMP1]])
// RESUME: cir.call @mark()

// RESUME-IR-LABEL: define {{.*}} void @two_awaits(
// RESUME-IR: await.ready:
// RESUME-IR: {{(call|invoke)}} void @_ZN1SC1Ei(ptr {{.*}} %[[TMP1:[^,]+]], i32 {{.*}} 1)
// RESUME-IR: {{(call|invoke)}} {{.*}} i32 @_ZN2DA12await_resumeERK1S(
// RESUME-IR: {{(call|invoke)}} void @_ZN1SC1Ei(ptr {{.*}} %[[TMP2:[^,]+]], i32 {{.*}} 2)
// RESUME-IR: {{(call|invoke)}} void @resume_use_pair(
// RESUME-IR: call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP2]])
// RESUME-IR: call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP1]])
// RESUME-IR: {{(call|invoke)}} void @mark()

extern "C" Task two_awaits() {
  resume_use_pair(co_await DA{}, co_await DV{});
  mark();
}

// RESUME-LABEL: cir.func coroutine {{.*}} @class_result(
// RESUME: cir.await(user, ready : {
// RESUME: cir.call @_ZN2RA12await_resumeERK1S(
// RESUME: },)
// RESUME: cir.call @resume_use_result(
// RESUME: cir.call @_ZN2NTD1Ev(
// RESUME: cir.call @_ZN1SD1Ev(

// RESUME-IR-LABEL: define {{.*}} void @class_result(
// RESUME-IR: {{(call|invoke)}} void @_ZN2RA12await_resumeERK1S(
// RESUME-IR: {{(call|invoke)}} void @resume_use_result(
// RESUME-IR: call void @_ZN2NTD1Ev(
// RESUME-IR: call void @_ZN1SD1Ev(

extern "C" Task class_result() {
  resume_use_result(co_await RA{});
}

// RESUME-LABEL: cir.func coroutine {{.*}} @in_conditional(
// RESUME: %[[FLAG:.*]] = cir.alloca "cleanup.cond"
// RESUME: %[[FALSE:.*]] = cir.const #false
// RESUME-NEXT: cir.store %[[FALSE]], %[[FLAG]]
// RESUME: cir.ternary(
// RESUME: cir.await(user, ready : {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP:[^,]+]],
// RESUME: },)
// RESUME: cir.call @resume_use(
// RESUME: %[[LIVE:.*]] = cir.load {{.*}} %[[FLAG]]
// RESUME-NEXT: cir.if %[[LIVE]] {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[TMP]])

extern "C" Task in_conditional() {
  resume_use(resume_flag() ? co_await DA{} : 0);
}

// RESUME-LABEL: cir.func coroutine {{.*}} @agg_init(
// RESUME: cir.call @_ZN1SC1Ei(
// RESUME-NEXT: %[[TRUE:.*]] = cir.const #true
// RESUME-NEXT: cir.store %[[TRUE]], %[[ACTIVE:.*]]
// RESUME: cir.await(user, ready : {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP:[^,]+]],
// RESUME: cir.call @_ZN2DA12await_resumeERK1S(
// RESUME: },)
// RESUME: %[[FALSE:.*]] = cir.const #false
// RESUME-NEXT: cir.store %[[FALSE]], %[[ACTIVE]]
// RESUME-NOT: cir.call @_ZN1SD1Ev(%[[TMP]])
// RESUME: cir.call @resume_next()
// RESUME: cir.call @resume_use_aggregate(
// RESUME: cir.call @_ZN3AggD1Ev(
// RESUME: cir.call @_ZN1SD1Ev(%[[TMP]])

// RESUME-IR-LABEL: define {{.*}} void @agg_init(
// RESUME-IR: {{(call|invoke)}} {{.*}} i32 @_ZN2DA12await_resumeERK1S(
// RESUME-IR: {{(call|invoke)}} i32 @resume_next()
// RESUME-IR: {{(call|invoke)}} void @resume_use_aggregate(
// RESUME-IR: call void @_ZN3AggD1Ev(
// RESUME-IR: call void @_ZN1SD1Ev(

extern "C" Task agg_init() {
  resume_use_aggregate(Agg{S(0), co_await DA{}}, resume_next());
}

// An unrelated conditional must not create a resume cleanup guard.
// RESUME-LABEL: cir.func coroutine {{.*}} @agg_plain_resume(
// RESUME-NOT: cir.alloca "cleanup.cond"
// RESUME: cir.await(user, ready : {
// RESUME: cir.call @_ZN2PL12await_resumeEv(
// RESUME: },)
// RESUME: cir.call @resume_use_aggregate(
// RESUME: cir.call @_ZN3AggD1Ev(
// RESUME: cir.call @mark()

extern "C" Task agg_plain_resume() {
  resume_use_aggregate(Agg{S(0), co_await PL{}}, resume_flag() ? 1 : 2);
  mark();
}

struct E {
  E(const S & = S(3));
  ~E();
};

struct Holder {
  E e[2];
  int k;
};

extern "C" int take(E *);
extern "C" int take2(Holder *);

struct AN {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  int await_resume(int = take(new E[2]) + take2(new Holder{}));
};

// Array-element boundaries still apply inside a deferred resume expression.
// RESUME-LABEL: cir.func coroutine {{.*}} @array_elements(
// RESUME: cir.await(user, ready : {
// RESUME: }, resume : {
// RESUME: cir.do {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP1:[^,]+]],
// RESUME: cir.call @_ZN1EC1ERK1S(
// RESUME-NOEH: } cleanup normal {
// RESUME-EH: } cleanup all {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[TMP1]])
// RESUME: } while {
// RESUME: cir.call @take(
// RESUME: cir.do {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP2:[^,]+]],
// RESUME: cir.call @_ZN1EC1ERK1S(
// RESUME-NOEH: } cleanup normal {
// RESUME-EH: } cleanup all {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[TMP2]])
// RESUME: } while {
// RESUME: cir.call @take2(
// RESUME: cir.call @_ZN2AN12await_resumeEi(
// RESUME: },)
// RESUME: cir.call @resume_use(

extern "C" Task array_elements() {
  resume_use(co_await AN{});
  mark();
}

// RESUME-LABEL: cir.func coroutine {{.*}} @asm_operand(
// RESUME: cir.await(user, ready : {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP:[^,]+]],
// RESUME: },)
// RESUME: cir.asm(
// RESUME: cir.call @_ZN1SD1Ev(%[[TMP]])
// RESUME: cir.call @mark()

extern "C" Task asm_operand() {
  asm volatile("" : : "r"(co_await DA{}));
  mark();
}

// RESUME-LABEL: cir.func coroutine {{.*}} @coyield_arg(
// RESUME: cir.await(yield, ready : {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP:[^,]+]],
// RESUME: },)
// RESUME: cir.call @resume_use(
// RESUME: cir.call @_ZN1SD1Ev(%[[TMP]])

extern "C" Task coyield_arg() {
  resume_use(co_yield 1);
}

struct TransformTask {
  struct promise_type {
    TransformTask get_return_object();
    std::suspend_never initial_suspend();
    std::suspend_never final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
    PL await_transform(int, int = resume_flag() ? resume_arg(S(5)) : 0);
  };
};

// RESUME-LABEL: cir.func coroutine {{.*}} @transform_condtemp(
// RESUME: %[[FLAG:.*]] = cir.alloca "cleanup.cond"
// RESUME: cir.ternary(
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP:[^,]+]],
// RESUME: cir.call @_ZN13TransformTask12promise_type15await_transformEii(
// RESUME: cir.await(user, ready : {
// RESUME: },)
// RESUME: cir.call @resume_use(
// RESUME: %[[LIVE:.*]] = cir.load {{.*}} %[[FLAG]]
// RESUME-NEXT: cir.if %[[LIVE]] {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[TMP]])

extern "C" TransformTask transform_condtemp() {
  resume_use(co_await 1);
}

struct NoexceptInitAwaiter {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  void await_resume(const S & = S(1)) noexcept;
};

struct NoexceptFinalAwaiter {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume(const S & = S(2)) noexcept;
};

struct DefaultInitTask {
  struct promise_type {
    DefaultInitTask get_return_object();
    NoexceptInitAwaiter initial_suspend();
    NoexceptFinalAwaiter final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
};

// RESUME-LABEL: cir.func coroutine {{.*}} @init_final_noexcept(
// RESUME-NOT: resume.eh
// RESUME: cir.await(init, ready : {
// RESUME: cir.call @_ZN1SC1Ei(%[[INIT_TMP:[^,]+]],
// RESUME: cir.call @_ZN19NoexceptInitAwaiter12await_resumeERK1S(
// RESUME: },)
// RESUME: cir.call @_ZN1SD1Ev(%[[INIT_TMP]])
// RESUME: }, body : {
// RESUME: cir.call @mark()
// RESUME: }, finalSuspend : {
// RESUME: cir.call @_ZN1SC1Ei(%[[FINAL_TMP:[^,]+]],
// RESUME: cir.call @_ZN20NoexceptFinalAwaiter12await_resumeERK1S(
// RESUME: },)
// RESUME: cir.call @_ZN1SD1Ev(%[[FINAL_TMP]])

extern "C" DefaultInitTask init_final_noexcept() {
  mark();
  co_return;
}

struct InitAwaiter {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  void await_resume(const S & = S(1), int = resume_flag() ? resume_arg(S(2)) : 0);
};

struct InitTask {
  struct promise_type {
    InitTask get_return_object();
    InitAwaiter initial_suspend();
    std::suspend_never final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
};

// RESUME-LABEL: cir.func coroutine {{.*}} @init_try(
// RESUME-EH: %[[RESUMED:.*]] = cir.alloca "resume.eh"
// RESUME: cir.await(init, ready : {
// RESUME-EH: cir.try {
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP1:[^,]+]],
// RESUME: cir.ternary(
// RESUME: cir.call @_ZN1SC1Ei(%[[TMP2:[^,]+]],
// RESUME: cir.call @_ZN11InitAwaiter12await_resumeERK1Si(
// RESUME: cir.call @_ZN1SD1Ev(%[[TMP2]])
// RESUME: cir.call @_ZN1SD1Ev(%[[TMP1]])
// RESUME-EH-NEXT: cir.yield
// RESUME-EH-NEXT: }
// RESUME-EH-NEXT: %[[TRUE:.*]] = cir.const #true
// RESUME-EH-NEXT: cir.store %[[TRUE]], %[[RESUMED]]
// RESUME-EH: } catch all
// RESUME: }, body : {
// RESUME: cir.call @mark()

extern "C" InitTask init_try() {
  mark();
  co_return;
}

struct ReadySuspendArgs {
  bool await_ready(const S & = S(8));
  void await_suspend(std::coroutine_handle<>, const S & = S(9));
  int await_resume();
};

// Ready and suspend arguments end before the next coroutine phase.
// RESUME-LABEL: cir.func coroutine {{.*}} @ready_suspend_in_conditional(
// RESUME: cir.ternary(
// RESUME: cir.await(user, ready : {
// RESUME: cir.call @_ZN1SC1Ei(%[[READY:[^,]+]],
// RESUME: cir.call @_ZN16ReadySuspendArgs11await_readyERK1S(
// RESUME-NOEH: } cleanup normal {
// RESUME-EH: } cleanup all {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[READY]])
// RESUME: }, suspend : {
// RESUME: cir.call @_ZN1SC1Ei(%[[SUSPEND:[^,]+]],
// RESUME: cir.call @_ZN16ReadySuspendArgs13await_suspendESt16coroutine_handleIvERK1S(
// RESUME-NOEH: } cleanup normal {
// RESUME-EH: } cleanup all {
// RESUME-NEXT: cir.call @_ZN1SD1Ev(%[[SUSPEND]])
// RESUME: cir.coro.suspend_point
// RESUME: }, resume : {
// RESUME: cir.call @_ZN16ReadySuspendArgs12await_resumeEv(

extern "C" Task ready_suspend_in_conditional(bool b) { resume_use(b ? co_await ReadySuspendArgs{} : 0); }
