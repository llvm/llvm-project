// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt -cir-flatten-cfg %t.cir -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir %s -check-prefix=CIR-FLAT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

#include "Inputs/coroutine.h"

struct S {
  ~S();
};

TaskWithEH throw_in_body(bool b) {
  if (b)
    throw 42;
  co_return;
}

// CIR-LABEL: cir.func {{.*}} @_Z13throw_in_bodyb(
// CIR:         cir.coroutine
// CIR:         }, body : {
// CIR:           cir.try {
// CIR:             cir.if %{{.*}} {
// CIR:               cir.throw %{{.*}}, %{{.*}}, %{{.*}} : !cir.ptr<!s32i>, !cir.ptr<!u8i>, !cir.ptr<!void>
// CIR-NEXT:          cir.unreachable
// CIR:             cir.co_return
// CIR:           } catch all (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:             cir.call @_ZN10TaskWithEH12promise_type19unhandled_exceptionEv

// CIR-FLAT-LABEL: cir.func {{.*}} @_Z13throw_in_bodyb(
// CIR-FLAT:         cir.try_throw %{{.*}}, %{{.*}}, %{{.*}} : !cir.ptr<!s32i>, !cir.ptr<!u8i>, !cir.ptr<!void> ^[[UNREACH:bb[0-9]+]], ^[[LPAD:bb[0-9]+]]
// CIR-FLAT-NEXT:  ^[[UNREACH]]:
// CIR-FLAT-NEXT:    cir.unreachable
// CIR-FLAT:       ^[[LPAD]]:
// CIR-FLAT-NEXT:    cir.eh.initiate
// CIR-FLAT:         cir.eh.dispatch
// CIR-FLAT:         cir.begin_catch
// CIR-FLAT:         cir.try_call @_ZN10TaskWithEH12promise_type19unhandled_exceptionEv

// OGCG-LABEL: define dso_local void @_Z13throw_in_bodyb(
// OGCG:         invoke void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi, ptr null)
// OGCG-NEXT:            to label %[[UNREACH:[a-z0-9.]+]] unwind label %[[LPAD:[a-z0-9.]+]]
// OGCG:       [[LPAD]]:
// OGCG-NEXT:    landingpad { ptr, i32 }
// OGCG-NEXT:            catch ptr null
// OGCG:         call ptr @__cxa_begin_catch(
// OGCG:         invoke void @_ZN10TaskWithEH12promise_type19unhandled_exceptionEv(
// OGCG:       [[UNREACH]]:
// OGCG-NEXT:    unreachable

TaskWithEH throw_in_cleanup_in_body(bool b) {
  {
    S s;
    if (b)
      throw 42;
  }
  co_return;
}

// CIR-LABEL: cir.func {{.*}} @_Z24throw_in_cleanup_in_bodyb(
// CIR:         cir.coroutine
// CIR:         }, body : {
// CIR:           cir.try {
// CIR:             %[[S:.*]] = cir.alloca "s"
// CIR:             cir.cleanup.scope {
// CIR:               cir.if %{{.*}} {
// CIR:                 cir.throw %{{.*}}, %{{.*}}, %{{.*}} : !cir.ptr<!s32i>, !cir.ptr<!u8i>, !cir.ptr<!void>
// CIR-NEXT:            cir.unreachable
// CIR:             } cleanup all {
// CIR-NEXT:          cir.call @_ZN1SD1Ev(%[[S]])
// CIR:             cir.co_return
// CIR:           } catch all (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:             cir.call @_ZN10TaskWithEH12promise_type19unhandled_exceptionEv

// CIR-FLAT-LABEL: cir.func {{.*}} @_Z24throw_in_cleanup_in_bodyb(
// CIR-FLAT:         %[[S:.*]] = cir.alloca "s"
// CIR-FLAT:         cir.try_throw %{{.*}}, %{{.*}}, %{{.*}} : !cir.ptr<!s32i>, !cir.ptr<!u8i>, !cir.ptr<!void> ^[[UNREACH:bb[0-9]+]], ^[[LPAD:bb[0-9]+]]
// CIR-FLAT-NEXT:  ^[[UNREACH]]:
// CIR-FLAT-NEXT:    cir.unreachable
// CIR-FLAT:       ^[[LPAD]]:
// CIR-FLAT-NEXT:    %[[TOK:.*]] = cir.eh.initiate
// CIR-FLAT-NEXT:    cir.br ^[[CLEANUP:bb[0-9]+]](%[[TOK]] : !cir.eh_token)
// CIR-FLAT:       ^[[CLEANUP]](%[[C_TOK:.*]]: !cir.eh_token):
// CIR-FLAT-NEXT:    %[[CT:.*]] = cir.begin_cleanup %[[C_TOK]]
// CIR-FLAT-NEXT:    cir.call @_ZN1SD1Ev(%[[S]])
// CIR-FLAT:         cir.eh.dispatch
// CIR-FLAT:         cir.begin_catch
// CIR-FLAT:         cir.try_call @_ZN10TaskWithEH12promise_type19unhandled_exceptionEv

// OGCG-LABEL: define dso_local void @_Z24throw_in_cleanup_in_bodyb(
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         invoke void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi, ptr null)
// OGCG-NEXT:            to label %[[UNREACH:[a-z0-9.]+]] unwind label %[[LPAD:[a-z0-9.]+]]
// OGCG:       [[LPAD]]:
// OGCG-NEXT:    landingpad { ptr, i32 }
// OGCG-NEXT:            catch ptr null
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         call ptr @__cxa_begin_catch(
// OGCG:         invoke void @_ZN10TaskWithEH12promise_type19unhandled_exceptionEv(
// OGCG:       [[UNREACH]]:
// OGCG-NEXT:    unreachable
