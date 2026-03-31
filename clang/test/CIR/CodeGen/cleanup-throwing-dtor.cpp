// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t-flat.cir
// RUN: FileCheck --input-file=%t-flat.cir %s --check-prefix=CIR-FLAT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test that a struct with a potentially-throwing destructor (noexcept(false))
// produces the correct high-level CIR (cleanup region without nothrow on the
// dtor call) and correct flattened CIR (try_call in the EH cleanup path with
// an unwind-to-terminate block).

struct ThrowingDtor {
  ~ThrowingDtor() noexcept(false);
  void doSomething();
};

void test_throwing_dtor_cleanup() {
  ThrowingDtor c;
  c.doSomething();
}

// High-level: the cleanup region's dtor call does NOT have nothrow.
//
// CIR: cir.func{{.*}} @_Z26test_throwing_dtor_cleanupv()
// CIR:   %[[C:.*]] = cir.alloca !rec_ThrowingDtor, !cir.ptr<!rec_ThrowingDtor>, ["c"]
// CIR:   cir.cleanup.scope {
// CIR:     cir.call @_ZN12ThrowingDtor11doSomethingEv(%[[C]])
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN12ThrowingDtorD1Ev(%[[C]])
// CIR:     cir.yield
// CIR:   }

// Flattened: body call becomes try_call. In the EH cleanup path, the dtor
// becomes a try_call that unwinds to a terminate block.
//
// CIR-FLAT: cir.func{{.*}} @_Z26test_throwing_dtor_cleanupv()
// CIR-FLAT:   %[[C:.*]] = cir.alloca !rec_ThrowingDtor, !cir.ptr<!rec_ThrowingDtor>, ["c"]
// CIR-FLAT:   cir.br ^[[BODY:bb[0-9]+]]
//
// Body: doSomething becomes a try_call.
// CIR-FLAT: ^[[BODY]]:
// CIR-FLAT:   cir.try_call @_ZN12ThrowingDtor11doSomethingEv(%[[C]]) ^[[NORMAL_BODY:bb[0-9]+]], ^[[UNWIND:bb[0-9]+]]
//
// Normal path: dtor is a regular call (not during unwinding).
// CIR-FLAT: ^[[NORMAL_BODY]]:
// CIR-FLAT:   cir.call @_ZN12ThrowingDtorD1Ev(%[[C]])
//
// EH cleanup: dtor becomes try_call with unwind to terminate.
// CIR-FLAT:   cir.try_call @_ZN12ThrowingDtorD1Ev(%[[C]]) ^[[DTOR_OK:bb[0-9]+]], ^[[TERMINATE:bb[0-9]+]]
// CIR-FLAT: ^[[DTOR_OK]]:
// CIR-FLAT:   cir.end_cleanup
// CIR-FLAT:   cir.resume
//
// Terminate block.
// CIR-FLAT: ^[[TERMINATE]]:
// CIR-FLAT:   %[[TET:.*]] = cir.eh.initiate : !cir.eh_token
// CIR-FLAT:   cir.eh.terminate %[[TET]] : !cir.eh_token

// LLVM IR via the CIR pipeline. doSomething is invoked, dtor is called on
// normal path, invoked on EH path with unwind to terminate.
//
// LLVM: define dso_local void @_Z26test_throwing_dtor_cleanupv()
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:   %[[C:.*]] = alloca %struct.ThrowingDtor
// LLVM:   invoke void @_ZN12ThrowingDtor11doSomethingEv(ptr {{.*}} %[[C]])
// LLVM:           to label %[[NORMAL:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[NORMAL]]:
// LLVM:   call void @_ZN12ThrowingDtorD1Ev(ptr {{.*}} %[[C]])
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           cleanup
// LLVM:   invoke void @_ZN12ThrowingDtorD1Ev(ptr {{.*}} %[[C]])
// LLVM:           to label %[[RESUME:.*]] unwind label %[[TERMINATE:.*]]
// LLVM: [[RESUME]]:
// LLVM:   resume { ptr, i32 }
// LLVM: [[TERMINATE]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   call void @__clang_call_terminate(ptr
// LLVM:   unreachable
// LLVM:   ret void
//
// LLVM: define linkonce_odr hidden void @__clang_call_terminate(ptr %[[EXN:.*]])
// LLVM:   call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVM:   call void @_ZSt9terminatev()
// LLVM:   unreachable

// Same structural flow from original Clang CodeGen.
//
// OGCG: define dso_local void @_Z26test_throwing_dtor_cleanupv()
// OGCG-SAME: personality ptr @__gxx_personality_v0
// OGCG:   %[[C:.*]] = alloca %struct.ThrowingDtor
// OGCG:   invoke void @_ZN12ThrowingDtor11doSomethingEv(ptr {{.*}} %[[C]])
// OGCG:           to label %[[NORMAL:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[NORMAL]]:
// OGCG:   call void @_ZN12ThrowingDtorD1Ev(ptr {{.*}} %[[C]])
// OGCG:   ret void
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:           cleanup
// OGCG:   invoke void @_ZN12ThrowingDtorD1Ev(ptr {{.*}} %[[C]])
// OGCG:           to label %[[RESUME:.*]] unwind label %[[TERMINATE:.*]]
// OGCG: [[RESUME]]:
// OGCG:   resume { ptr, i32 }
// OGCG: [[TERMINATE]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:           catch ptr null
// OGCG:   call void @__clang_call_terminate(ptr
// OGCG:   unreachable
//
// OGCG: define linkonce_odr hidden void @__clang_call_terminate(ptr {{.*}} %[[EXN:.*]])
// OGCG:   call ptr @__cxa_begin_catch(ptr %[[EXN]])
// OGCG:   call void @_ZSt9terminatev()
// OGCG:   unreachable
