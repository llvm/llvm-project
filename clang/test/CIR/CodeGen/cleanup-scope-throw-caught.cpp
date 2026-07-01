// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt -cir-hoist-allocas -cir-flatten-cfg -cir-eh-abi-lowering %t.cir -o %t.eh.cir
// RUN: FileCheck --input-file=%t.eh.cir %s -check-prefix=CIR-EHABI
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct C {
  C();
  ~C();
};

// Two EH cleanups -- destroying the local `a` and freeing the thrown
// `new C()` if its constructor throws -- funnel into the same catch dispatch,
// so the throw's landing pad must carry both the cleanup and the catch clause.
int testCaughtThrowSharedDispatch() {
  try {
    C a;
    throw new C();
  } catch (C *p) {
    delete p;
    return 0;
  }
  return 1;
}

// CIR: cir.func{{.*}} @_Z29testCaughtThrowSharedDispatchv()
// CIR:   %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_C>
// CIR:   cir.try {
// CIR:     cir.call @_ZN1CC1Ev(%[[A]])
// CIR:     cir.cleanup.scope {
// CIR:       cir.throw %{{.*}} : !cir.ptr<!cir.ptr<!rec_C>>, @_ZTIP1C
// CIR:       cir.unreachable
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN1CD1Ev(%[[A]]) nothrow
// CIR:       cir.yield
// CIR:     }
// CIR:   } catch [type #cir.global_view<@_ZTIP1C> : !cir.ptr<!u8i>]

// After EHABI lowering the throw's in-flight exception carries BOTH the
// `cleanup` clause (to run `a`'s destructor) and the `catch @_ZTIP1C` clause
// (to reach the handler).  The missing catch clause was the bug: the exception
// would resume past the handler and terminate.

// CIR-EHABI-LABEL: cir.func{{.*}} @_Z29testCaughtThrowSharedDispatchv()
// CIR-EHABI:   cir.eh.inflight_exception cleanup [@_ZTIP1C]

// The throw lowers to an `invoke @__cxa_throw` whose unwind landing pad has
// BOTH a `cleanup` clause and the `catch ptr @_ZTIP1C` clause; OGCG produces
// the equivalent landing pad.

// LLVM: define {{.*}}@_Z29testCaughtThrowSharedDispatchv()
// LLVM:   invoke void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIP1C, ptr null)
// LLVM-NEXT: to label %{{.*}} unwind label %[[LPAD:.*]]
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM-NEXT: cleanup
// LLVM-NEXT: catch ptr @_ZTIP1C

struct S {
  S();
  ~S();
};
void may_throw();

// A throw in nested try/catch reaches more than one dispatch; its landing pad
// lists every reachable catch clause (innermost first) plus the cleanup.
void testNestedTry() {
  try {
    S a;
    try {
      may_throw();
    } catch (int) {
    }
  } catch (double) {
  }
}

// CIR: cir.func{{.*}} @_Z13testNestedTryv()
// CIR:   cir.try {
// CIR:     cir.try {
// CIR:       cir.call @_Z9may_throwv()
// CIR:     } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>]
// CIR:   } catch [type #cir.global_view<@_ZTId> : !cir.ptr<!u8i>]

// The may_throw landing pad reaches both the inner (int) and outer (double)
// dispatches plus the cleanup for `a`, so its in-flight exception lists both
// catch clauses innermost first alongside the cleanup.

// CIR-EHABI-LABEL: cir.func{{.*}} @_Z13testNestedTryv()
// CIR-EHABI:   cir.eh.inflight_exception cleanup [@_ZTIi, @_ZTId]

// LLVM: define {{.*}}@_Z13testNestedTryv()
// LLVM:   invoke void @_Z9may_throwv()
// LLVM-NEXT: to label %{{.*}} unwind label %[[LP:.*]]
// LLVM: [[LP]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM-NEXT: cleanup
// LLVM-NEXT: catch ptr @_ZTIi
// LLVM-NEXT: catch ptr @_ZTId
