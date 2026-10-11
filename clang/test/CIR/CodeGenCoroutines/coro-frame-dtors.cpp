// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-NOEH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t-eh.cir
// RUN: FileCheck --input-file=%t-eh.cir %s -check-prefixes=CIR,CIR-EH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t-O1.cir
// RUN: FileCheck --input-file=%t-O1.cir %s -check-prefix=CIR-O1
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Ensure coroutine parameter copies, promises and the body are destroyed in the correct order.

#include "Inputs/coroutine.h"

struct Dtor {
  ~Dtor();
  int x;
};

struct Task {
  struct promise_type {
    ~promise_type();
    Task get_return_object() noexcept;
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

struct TrivialTask {
  struct promise_type {
    TrivialTask get_return_object() noexcept;
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

Task promise_dtor() { co_return; }

// The promise's destructor is not scoped to initial_suspend, where the promise
// is constructed, and runs only in destroy.

// CIR-LABEL: cir.func coroutine {{.*}} @_Z12promise_dtorv(
// CIR:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-NOT:     cir.cleanup.scope
// CIR:         cir.coroutine initialSuspend : {
// CIR-NOT:       {{cir.cleanup.scope|cir.call @_ZN4Task12promise_typeD1Ev}}
// CIR:           cir.await(init,
// CIR-NOT:       cir.call @_ZN4Task12promise_typeD1Ev
// CIR:         }, body : {
// CIR-NOT:       cir.call @_ZN4Task12promise_typeD1Ev
// CIR:         }, destroy : {
// CIR-NEXT:      cir.call @_ZN4Task12promise_typeD1Ev(%[[PROMISE]])
// CIR-NEXT:      %[[MEM:.*]] = cir.coro.intrinsic.free
// CIR-NOT:       cir.call @_ZN4Task12promise_typeD1Ev
// CIR:           cir.call @_ZdlPvm(%[[MEM]],
// CIR-NOT:       cir.call @_ZN4Task12promise_typeD1Ev
// CIR:         }, exit : {
// CIR-NOT:     cir.call @_ZN4Task12promise_typeD1Ev

// OGCG-LABEL: define {{.*}} @_Z12promise_dtorv(
// OGCG:       coro.cleanup:
// OGCG-NEXT:    call void @_ZN4Task12promise_typeD1Ev(ptr {{.*}} %__promise)
// OGCG:         call ptr @llvm.coro.free(

TrivialTask param_dtor(Dtor p) {
  co_await std::suspend_always{};
}

// The parameter copy's destructor is not scoped around the whole
// cir.coroutine, where it would run on every return to the caller.

// CIR-LABEL: cir.func coroutine {{.*}} @_Z10param_dtor4Dtor(
// CIR:         %[[P:.*]] = cir.alloca "p"
// CIR-NOT:     cir.cleanup.scope
// CIR:         cir.coroutine initialSuspend : {
// CIR-NOT:       cir.call @_ZN4DtorD1Ev
// CIR:         }, destroy : {
// CIR-NEXT:      cir.call @_ZN4DtorD1Ev(%[[P]])
// CIR-NEXT:      cir.coro.intrinsic.free
// CIR-NOT:       cir.call @_ZN4DtorD1Ev
// CIR:         }, exit : {
// CIR-NOT:       cir.call @_ZN4DtorD1Ev
// CIR:           cir.return
// CIR-NEXT:    }
// CIR-NOT:     cir.cleanup.scope
// CIR-NOT:     cir.call @_ZN4DtorD1Ev

// OGCG-LABEL: define {{.*}} @_Z10param_dtor4Dtor(
// OGCG:         %[[P_COPY:.+]] = alloca %struct.Dtor
// OGCG:       coro.cleanup:
// OGCG:         call void @_ZN4DtorD1Ev(ptr {{[^,]*}} %[[P_COPY]])
// OGCG:         call ptr @llvm.coro.free(

Task promise_and_params(Dtor a, int i, Dtor b) {
  Dtor local;
  co_await std::suspend_always{};
}

// The promise is destroyed first, then the parameter copies in reverse order
// of construction. A local variable of the body is still destroyed when
// control leaves the body, before the implicit co_return calls return_void().

// CIR-LABEL: cir.func coroutine {{.*}} @_Z18promise_and_params4DtoriS_(
// CIR-DAG:     %[[A:.*]] = cir.alloca "a"
// CIR-DAG:     %[[B:.*]] = cir.alloca "b"
// CIR-DAG:     %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-NOEH-DAG: %[[LOCAL:.*]] = cir.alloca "local"
// CIR-NOT:     cir.call {{@_ZN4DtorD1Ev|@_ZN4Task12promise_typeD1Ev}}
// CIR:         cir.coroutine initialSuspend : {
// CIR-NOT:       cir.call {{@_ZN4DtorD1Ev|@_ZN4Task12promise_typeD1Ev}}
// CIR:         }, body : {
// CIR-EH:        %[[LOCAL:.*]] = cir.alloca "local"
// CIR:             cir.await(user,
// CIR:             },)
// CIR-NEXT:        cir.yield
// CIR-NOEH-NEXT: } cleanup normal {
// CIR-EH-NEXT:   } cleanup all {
// CIR-NEXT:        cir.call @_ZN4DtorD1Ev(%[[LOCAL]])
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:      cir.call @_ZN4Task12promise_type11return_voidEv(
// CIR-NEXT:      cir.co_return
// CIR-NOT:       cir.call {{@_ZN4DtorD1Ev|@_ZN4Task12promise_typeD1Ev}}
// CIR:         }, destroy : {
// CIR-NEXT:      cir.call @_ZN4Task12promise_typeD1Ev(%[[PROMISE]])
// CIR-NEXT:      cir.call @_ZN4DtorD1Ev(%[[B]])
// CIR-NEXT:      cir.call @_ZN4DtorD1Ev(%[[A]])
// CIR-NEXT:      cir.coro.intrinsic.free
// CIR-NOT:       cir.call {{@_ZN4DtorD1Ev|@_ZN4Task12promise_typeD1Ev}}
// CIR:         }, exit : {
// CIR-NOT:     cir.call {{@_ZN4DtorD1Ev|@_ZN4Task12promise_typeD1Ev}}

// At -O1, the lifetime of the promise and of the parameter copies also ends in
// destroy, after each object is destroyed.

// CIR-O1-LABEL: cir.func coroutine {{.*}} @_Z18promise_and_params4DtoriS_(
// CIR-O1-DAG:     %[[A:.*]] = cir.alloca "a"
// CIR-O1-DAG:     %[[B:.*]] = cir.alloca "b"
// CIR-O1-DAG:     %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-O1:         cir.lifetime.start %[[A]]
// CIR-O1:         cir.lifetime.start %[[I:.*]] : !cir.ptr<!s32i>
// CIR-O1:         cir.lifetime.start %[[B]]
// CIR-O1:         cir.coroutine initialSuspend : {
// CIR-O1-NEXT:      cir.lifetime.start %[[PROMISE]]
// CIR-O1-NOT:       cir.lifetime.end %[[PROMISE]]
// CIR-O1-NOT:       cir.lifetime.end %[[B]]
// CIR-O1-NOT:       cir.lifetime.end %[[A]]
// CIR-O1:         }, destroy : {
// CIR-O1-NEXT:      cir.call @_ZN4Task12promise_typeD1Ev(%[[PROMISE]])
// CIR-O1-NEXT:      cir.lifetime.end %[[PROMISE]]
// CIR-O1-NEXT:      cir.call @_ZN4DtorD1Ev(%[[B]])
// CIR-O1-NEXT:      cir.lifetime.end %[[B]]
// CIR-O1-NEXT:      cir.lifetime.end %[[I]]
// CIR-O1-NEXT:      cir.call @_ZN4DtorD1Ev(%[[A]])
// CIR-O1-NEXT:      cir.lifetime.end %[[A]]
// CIR-O1-NEXT:      cir.coro.intrinsic.free

// OGCG-LABEL: define {{.*}} @_Z18promise_and_params4DtoriS_(
// OGCG:         %[[A_COPY:.+]] = alloca %struct.Dtor
// OGCG:         %[[B_COPY:.+]] = alloca %struct.Dtor
// OGCG:       coro.cleanup:
// OGCG-NEXT:    call void @_ZN4Task12promise_typeD1Ev(ptr {{[^,]*}} %__promise)
// OGCG:         call void @_ZN4DtorD1Ev(ptr {{[^,]*}} %[[B_COPY]])
// OGCG:         call void @_ZN4DtorD1Ev(ptr {{[^,]*}} %[[A_COPY]])
// OGCG:         call ptr @llvm.coro.free(

// The promise constructor sees the parameter copies.
struct PreviewTask {
  struct promise_type {
    promise_type(Dtor &);
    ~promise_type();
    PreviewTask get_return_object() noexcept;
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

PreviewTask promise_preview(Dtor d) { co_return; }

// CIR-LABEL: cir.func coroutine {{.*}} @_Z15promise_preview4Dtor(
// CIR:         %[[D:.*]] = cir.alloca "d"
// CIR:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR:         cir.coroutine initialSuspend : {
// CIR-NEXT:      cir.call @_ZN11PreviewTask12promise_typeC1ER4Dtor(%[[PROMISE]], %[[D]])
// CIR:         }, destroy : {
// CIR-NEXT:      cir.call @_ZN11PreviewTask12promise_typeD1Ev(%[[PROMISE]])
// CIR-NEXT:      cir.call @_ZN4DtorD1Ev(%[[D]])
// CIR-NEXT:      cir.coro.intrinsic.free

// A temporary in a parameter copy's or the promise's initializer is still
// destroyed at the end of its full-expression; it does not leak into destroy
// and does not leave initial_suspend unterminated.
struct Tmp {
  Tmp();
  ~Tmp();
};

struct TmpMove {
  TmpMove(TmpMove &&, Tmp = Tmp());
  ~TmpMove();
};

struct TmpTask {
  struct promise_type {
    promise_type(Tmp = Tmp());
    ~promise_type();
    TmpTask get_return_object() noexcept;
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

TmpTask init_temporaries(TmpMove m) { co_return; }

// CIR-LABEL: cir.func coroutine {{.*}} @_Z16init_temporaries7TmpMove(
// CIR:           %[[M:.*]] = cir.alloca "m"
// CIR:           %[[MTMP:.*]] = cir.alloca "agg.tmp0"
// CIR:           %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR:           %[[PTMP:.*]] = cir.alloca "agg.tmp1"
// CIR:           cir.call @_ZN3TmpC1Ev(%[[MTMP]])
// CIR-NEXT:      cir.cleanup.scope {
// CIR-NEXT:        cir.call @_ZN7TmpMoveC1EOS_3Tmp(%[[M]], %arg0, %[[MTMP]])
// CIR-NEXT:        cir.yield
// CIR-NOEH-NEXT: } cleanup normal {
// CIR-EH-NEXT:   } cleanup all {
// CIR-NEXT:        cir.call @_ZN3TmpD1Ev(%[[MTMP]])
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:      cir.coroutine initialSuspend : {
// CIR-NEXT:        cir.call @_ZN3TmpC1Ev(%[[PTMP]])
// CIR-NEXT:        cir.cleanup.scope {
// CIR-NEXT:          cir.call @_ZN7TmpTask12promise_typeC1E3Tmp(%[[PROMISE]], %[[PTMP]])
// CIR-NEXT:          cir.yield
// CIR-NOEH-NEXT:   } cleanup normal {
// CIR-EH-NEXT:     } cleanup all {
// CIR-NEXT:          cir.call @_ZN3TmpD1Ev(%[[PTMP]])
// CIR-NEXT:          cir.yield
// CIR-NEXT:        }
// CIR-NOT:         cir.cleanup.scope
// CIR:             cir.await(init,
// CIR:           }, destroy : {
// CIR-NEXT:        cir.call @_ZN7TmpTask12promise_typeD1Ev(%[[PROMISE]])
// CIR-NEXT:        cir.call @_ZN7TmpMoveD1Ev(%[[M]])
// CIR-NEXT:        cir.coro.intrinsic.free
// CIR-NOT:         cir.call @_ZN3TmpD1Ev
// CIR:           }, exit : {

// A temporary in the get_return_object() call is destroyed at the end of that
// full-expression, inside initial_suspend ([class.temporary]p4). Classic
// CodeGen does not end this full-expression: it leaves the temporary's cleanup
// on the EH stack and destroys it in coro.cleanup, with the coroutine state.
struct GROTask {
  struct promise_type {
    GROTask get_return_object(Tmp = Tmp()) noexcept;
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

GROTask gro_temporary() { co_return; }

// CIR-LABEL: cir.func coroutine {{.*}} @_Z13gro_temporaryv(
// CIR:         cir.coroutine initialSuspend : {
// CIR:           cir.cleanup.scope {
// CIR-NEXT:        cir.call @_ZN7GROTask12promise_type17get_return_objectE3Tmp(
// CIR-NOT:         cir.await
// CIR-NOEH:      } cleanup normal {
// CIR-EH:        } cleanup all {
// CIR-NEXT:        cir.call @_ZN3TmpD1Ev
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:      cir.call @_ZN7GROTask12promise_type15initial_suspendEv(
// CIR:           cir.await(init,
// CIR:         }, destroy : {
// CIR-NOT:       cir.call @_ZN3TmpD1Ev
// CIR:         }, exit : {

// OGCG-LABEL: define {{.*}} @_Z13gro_temporaryv(
// OGCG:         call void @_ZN7GROTask12promise_type17get_return_objectE3Tmp(
// OGCG-NOT:     call void @_ZN3TmpD1Ev
// OGCG:       coro.cleanup:
// OGCG-NEXT:    call void @_ZN3TmpD1Ev(

// A temporary that such a default argument creates conditionally is destroyed
// at the end of the full-expression too, guarded by its active flag. This also
// holds for a parameter copy's constructor.
bool cond();

struct CondTmp {
  CondTmp();
  ~CondTmp();
  bool ok();
};

struct CondGROTask {
  struct promise_type {
    CondGROTask get_return_object(bool = cond() && CondTmp().ok()) noexcept;
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

CondGROTask cond_gro_temporary() { co_return; }

// CIR-LABEL: cir.func coroutine {{.*}} @_Z18cond_gro_temporaryv(
// CIR:         %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_CondTmp>
// CIR:         %[[ACTIVE:.*]] = cir.alloca "cleanup.cond"
// CIR:         cir.coroutine initialSuspend : {
// CIR:           cir.cleanup.scope {
// CIR-NEXT:        %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:        cir.store %[[FALSE]], %[[ACTIVE]]
// CIR-NEXT:        %{{.*}} = cir.ternary(%{{.*}}, true {
// CIR-NEXT:          cir.call @_ZN7CondTmpC1Ev(%[[TMP]])
// CIR-NEXT:          %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:          cir.store %[[TRUE]], %[[ACTIVE]]
// CIR:             cir.call @_ZN11CondGROTask12promise_type17get_return_objectEb(
// CIR-NOEH:      } cleanup normal {
// CIR-EH:        } cleanup all {
// CIR-NEXT:        %[[IS_ACTIVE:.*]] = cir.load {{.*}} %[[ACTIVE]]
// CIR-NEXT:        cir.if %[[IS_ACTIVE]] {
// CIR-NEXT:          cir.call @_ZN7CondTmpD1Ev(%[[TMP]])
// CIR:           cir.await(init,
// CIR:         }, destroy : {
// CIR-NOT:       cir.call @_ZN7CondTmpD1Ev
// CIR:         }, exit : {

struct CondMove {
  CondMove(CondMove &&, bool = cond() && CondTmp().ok());
  ~CondMove();
};

TrivialTask cond_param_temporary(CondMove m) { co_return; }

// CIR-LABEL: cir.func coroutine {{.*}} @_Z20cond_param_temporary8CondMove(
// CIR:         %[[M:.*]] = cir.alloca "m"
// CIR:         %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_CondTmp>
// CIR:         %[[ACTIVE:.*]] = cir.alloca "cleanup.cond"
// CIR:         cir.cleanup.scope {
// CIR-NEXT:      %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[ACTIVE]]
// CIR-NEXT:      %{{.*}} = cir.ternary(%{{.*}}, true {
// CIR-NEXT:        cir.call @_ZN7CondTmpC1Ev(%[[TMP]])
// CIR-NEXT:        %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:        cir.store %[[TRUE]], %[[ACTIVE]]
// CIR:           cir.call @_ZN8CondMoveC1EOS_b(%[[M]],
// CIR-NOEH:    } cleanup normal {
// CIR-EH:      } cleanup all {
// CIR-NEXT:      %[[IS_ACTIVE:.*]] = cir.load {{.*}} %[[ACTIVE]]
// CIR-NEXT:      cir.if %[[IS_ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN7CondTmpD1Ev(%[[TMP]])
// CIR:         cir.coroutine initialSuspend : {
// CIR:         }, destroy : {
// CIR-NEXT:      cir.call @_ZN8CondMoveD1Ev(%[[M]])
// CIR-NOT:       cir.call @_ZN7CondTmpD1Ev
// CIR:         }, exit : {

// A coroutine that never reaches its end has no co_return and no fallthrough
// co_return, so final_suspend has no cir.await, but destroy still destroys
// the promise and the parameter copies.
struct ValueTask {
  struct promise_type {
    ~promise_type();
    ValueTask get_return_object() noexcept;
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_value(int) noexcept;
    void unhandled_exception() noexcept;
  };
};

ValueTask no_final_await(Dtor p) {
  for (;;)
    co_await std::suspend_always{};
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z14no_final_await4Dtor(
// CIR:         %[[P:.*]] = cir.alloca "p"
// CIR:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR:         }, finalSuspend : {
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }, destroy : {
// CIR-NEXT:      cir.call @_ZN9ValueTask12promise_typeD1Ev(%[[PROMISE]])
// CIR-NEXT:      cir.call @_ZN4DtorD1Ev(%[[P]])
// CIR-NEXT:      cir.coro.intrinsic.free

// A member coroutine copies its by-value parameters, but not *this.
struct S {
  Task member(Dtor d);
};
Task S::member(Dtor d) { co_return; }

// CIR-LABEL: cir.func coroutine {{.*}} @_ZN1S6memberE4Dtor(
// CIR:         %[[D:.*]] = cir.alloca "d"
// CIR:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR:         }, destroy : {
// CIR-NEXT:      cir.call @_ZN4Task12promise_typeD1Ev(%[[PROMISE]])
// CIR-NEXT:      cir.call @_ZN4DtorD1Ev(%[[D]])
// CIR-NEXT:      cir.coro.intrinsic.free

// An exception from get_return_object() propagates to the caller before the
// initial suspend point; classic CodeGen destroys the parameter copy and frees
// the frame on that path. FIXME: The unwind region is not implemented yet, so
// with -fexceptions CIR currently destroys the copy only in destroy, which the
// exception does not reach.
struct ThrowingGROTask {
  struct promise_type {
    ThrowingGROTask get_return_object();
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

ThrowingGROTask eh_before_body(Dtor p) { co_return; }

// CIR-LABEL: cir.func coroutine {{.*}} @_Z14eh_before_body4Dtor(
// CIR:         %[[P:.*]] = cir.alloca "p"
// CIR-NOT:     cir.cleanup.scope
// CIR:         cir.coroutine initialSuspend : {
// CIR-NEXT:      cir.call @_ZN15ThrowingGROTask12promise_type17get_return_objectEv(
// CIR-NOT:       cir.call @_ZN4DtorD1Ev
// CIR:         }, destroy : {
// CIR-NEXT:      cir.call @_ZN4DtorD1Ev(%[[P]])

// OGCG-LABEL: define {{.*}} @_Z14eh_before_body4Dtor(
// OGCG:         %[[P_COPY:.+]] = alloca %struct.Dtor
// OGCG:       coro.cleanup:
// OGCG:         call void @_ZN4DtorD1Ev(ptr {{[^,]*}} %[[P_COPY]])
