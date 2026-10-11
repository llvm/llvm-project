// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-NOEH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t-eh.cir
// RUN: FileCheck --input-file=%t-eh.cir %s -check-prefixes=CIR,CIR-EH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t-O1.cir
// RUN: FileCheck --input-file=%t-O1.cir %s -check-prefix=CIR-O1
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=OGCG,OGCG-NOEH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm -disable-llvm-passes %s -o %t-eh.ll
// RUN: FileCheck --input-file=%t-eh.ll %s -check-prefixes=OGCG,OGCG-EH

// When the body of a coroutine falls off its end, the implicit `co_return;`
// runs after the body: the body's local variables are destroyed first, then
// return_void() is called, then the coroutine goes to its final suspend point.
// An explicit co_return calls return_void() while the local variables are
// still alive and destroys them on its way to the final suspend point.

#include "Inputs/coroutine.h"

struct S {
  ~S();
};

struct Task {
  struct promise_type {
    Task get_return_object() noexcept;
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;
    // Not noexcept: with exceptions, the implicit co_return is still inside
    // the body's try/catch, so an exception from return_void() goes to
    // unhandled_exception().
    void return_void();
    void unhandled_exception();
  };
};

S make();
void use(int &);

Task falls_off_end() {
  co_await std::suspend_always{};
  S s;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z13falls_off_endv(
// CIR:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-NOEH:    %[[S:.*]] = cir.alloca "s"
// CIR:         }, body : {
// CIR-EH:        %[[S:.*]] = cir.alloca "s"
// CIR-EH:        cir.try {
// CIR:             cir.await(user,
// CIR:             cir.cleanup.scope {
// CIR-NEXT:          cir.yield
// CIR-NOEH-NEXT:   } cleanup normal {
// CIR-EH-NEXT:     } cleanup all {
// CIR-NEXT:          cir.call @_ZN1SD1Ev(%[[S]])
// CIR-NEXT:          cir.yield
// CIR-NEXT:        }
// CIR-NEXT:        cir.call @_ZN4Task12promise_type11return_voidEv(%[[PROMISE]])
// CIR-NEXT:        cir.co_return
// CIR-EH-NEXT:   } catch all
// CIR:         }, finalSuspend : {

// OGCG-LABEL: define {{.*}} void @_Z13falls_off_endv(
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S:.*]])
// OGCG-NEXT:    call void @llvm.lifetime.end.p0(ptr %[[S]])
// OGCG-NOEH-NEXT: call void @_ZN4Task12promise_type11return_voidEv(
// OGCG-NOEH-NEXT: br label %coro.final
// OGCG-EH-NEXT:   invoke void @_ZN4Task12promise_type11return_voidEv(
// OGCG-EH-NEXT:     to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG-EH:      [[CONT]]:
// OGCG-EH-NEXT:   br label %coro.final
// OGCG-EH:      [[LPAD]]:
// OGCG-EH-NEXT:   landingpad
// OGCG-EH-NEXT:     catch ptr null

// A temporary that a local variable at the top level of the body extends is
// destroyed before the implicit co_return as well.

Task falls_off_end_with_extended_temporary() {
  co_await std::suspend_always{};
  const S &r = make();
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z37falls_off_end_with_extended_temporaryv(
// CIR:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR:         }, body : {
// CIR:             cir.await(user,
// CIR:             cir.call @_Z4makev(%[[TMP:[0-9]+]])
// CIR-NEXT:        cir.cleanup.scope {
// CIR-NEXT:          cir.store {{.*}} %[[TMP]], %{{.+}} :
// CIR-NEXT:          cir.yield
// CIR-NOEH-NEXT:   } cleanup normal {
// CIR-EH-NEXT:     } cleanup all {
// CIR-NEXT:          cir.call @_ZN1SD1Ev(%[[TMP]])
// CIR-NEXT:          cir.yield
// CIR-NEXT:        }
// CIR-NEXT:        cir.call @_ZN4Task12promise_type11return_voidEv(%[[PROMISE]])
// CIR-NEXT:        cir.co_return
// CIR-EH-NEXT:   } catch all
// CIR:         }, finalSuspend : {

// OGCG-LABEL: define {{.*}} void @_Z37falls_off_end_with_extended_temporaryv(
// OGCG:         {{call|invoke}} void @_Z4makev(ptr {{.*}} %[[TMP:[a-z.0-9]+]])
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// OGCG-NOT:     return_void
// OGCG:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// OGCG-NEXT:    call void @llvm.lifetime.end.p0(
// OGCG-NOEH-NEXT: call void @_ZN4Task12promise_type11return_voidEv(
// OGCG-EH-NEXT:   invoke void @_ZN4Task12promise_type11return_voidEv(

// With optimizations, even a local variable without a destructor has a
// cleanup, its lifetime end.

Task falls_off_end_with_trivial_local() {
  co_await std::suspend_always{};
  int i = 0;
  use(i);
}

// CIR-O1-LABEL: cir.func coroutine {{.*}} @_Z32falls_off_end_with_trivial_localv(
// CIR-O1:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-O1:         %[[I:.*]] = cir.alloca "i"
// CIR-O1:         }, body : {
// CIR-O1:           cir.lifetime.start %[[I]]
// CIR-O1-NEXT:      cir.cleanup.scope {
// CIR-O1:             cir.call @_Z3useRi(%[[I]])
// CIR-O1-NEXT:        cir.yield
// CIR-O1-NEXT:      } cleanup normal {
// CIR-O1-NEXT:        cir.lifetime.end %[[I]]
// CIR-O1-NEXT:        cir.yield
// CIR-O1-NEXT:      }
// CIR-O1-NEXT:      cir.call @_ZN4Task12promise_type11return_voidEv(%[[PROMISE]])
// CIR-O1-NEXT:      cir.co_return
// CIR-O1-NEXT:    }, finalSuspend : {

// A co_return inside an if does not make the end of the body unreachable, so
// the fall-through handler is still emitted, after the local variables are
// destroyed.

Task co_return_in_if(bool b) {
  S s;
  if (b)
    co_return;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z15co_return_in_ifb(
// CIR:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-NOEH:    %[[S:.*]] = cir.alloca "s"
// CIR:         }, body : {
// CIR-EH:        %[[S:.*]] = cir.alloca "s"
// CIR-EH:        cir.try {
// CIR:             cir.cleanup.scope {
// CIR-NEXT:          cir.scope {
// CIR-NEXT:            %[[B:.*]] = cir.load
// CIR-NEXT:            cir.if %[[B]] {
// CIR-NEXT:              cir.call @_ZN4Task12promise_type11return_voidEv(%[[PROMISE]])
// CIR-NEXT:              cir.co_return
// CIR-NEXT:            }
// CIR-NEXT:          }
// CIR-NEXT:          cir.yield
// CIR-NOEH-NEXT:   } cleanup normal {
// CIR-EH-NEXT:     } cleanup all {
// CIR-NEXT:          cir.call @_ZN1SD1Ev(%[[S]])
// CIR-NEXT:          cir.yield
// CIR-NEXT:        }
// CIR-NEXT:        cir.call @_ZN4Task12promise_type11return_voidEv(%[[PROMISE]])
// CIR-NEXT:        cir.co_return
// CIR-EH-NEXT:   } catch all
// CIR:         }, finalSuspend : {

// OGCG-LABEL: define {{.*}} void @_Z15co_return_in_ifb(
// OGCG:       if.then:
// OGCG-NOEH-NEXT: call void @_ZN4Task12promise_type11return_voidEv(
// OGCG-EH-NEXT:   invoke void @_ZN4Task12promise_type11return_voidEv(
// OGCG:       cleanup{{[0-9]*}}:
// OGCG-NEXT:    %[[DEST:.*]] = phi i32
// OGCG-NEXT:    call void @_ZN1SD1Ev(
// OGCG-NEXT:    call void @llvm.lifetime.end.p0(
// OGCG-NEXT:    switch i32 %[[DEST]], label %unreachable [
// OGCG-NEXT:      i32 {{[0-9]+}}, label %[[FALLTHROUGH:.*]]
// OGCG-NEXT:      i32 {{[0-9]+}}, label %coro.final
// OGCG-NEXT:    ]
// OGCG:       [[FALLTHROUGH]]:
// OGCG-NOEH-NEXT: call void @_ZN4Task12promise_type11return_voidEv(
// OGCG-NOEH-NEXT: br label %coro.final
// OGCG-EH-NEXT:   invoke void @_ZN4Task12promise_type11return_voidEv(

Task co_return_at_end() {
  co_await std::suspend_always{};
  S s;
  co_return;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z16co_return_at_endv(
// CIR:         %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-NOEH:    %[[S:.*]] = cir.alloca "s"
// CIR:         }, body : {
// CIR-EH:        %[[S:.*]] = cir.alloca "s"
// CIR-EH:        cir.try {
// CIR:             cir.await(user,
// CIR:             cir.cleanup.scope {
// CIR-NEXT:          cir.call @_ZN4Task12promise_type11return_voidEv(%[[PROMISE]])
// CIR-NEXT:          cir.co_return
// CIR-NOEH-NEXT:   } cleanup normal {
// CIR-EH-NEXT:     } cleanup all {
// CIR-NEXT:          cir.call @_ZN1SD1Ev(%[[S]])
// CIR-NEXT:          cir.yield
// CIR-NEXT:        }
// CIR-NOT:         cir.call @_ZN4Task12promise_type11return_voidEv
// CIR:         }, finalSuspend : {

// OGCG-LABEL: define {{.*}} void @_Z16co_return_at_endv(
// OGCG-NOEH:    call void @_ZN4Task12promise_type11return_voidEv(
// OGCG-NOEH-NEXT: call void @_ZN1SD1Ev(
// OGCG-EH:      invoke void @_ZN4Task12promise_type11return_voidEv(
// OGCG-EH-NEXT:   to label %[[CONT:.*]] unwind label
// OGCG-EH:      [[CONT]]:
// OGCG-EH-NEXT:   call void @_ZN1SD1Ev(
// OGCG-NOT:     return_void
// OGCG:         br label %coro.final
