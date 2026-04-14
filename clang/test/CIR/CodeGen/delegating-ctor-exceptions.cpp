// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fcxx-exceptions -fexceptions \
// RUN:     -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck -check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fcxx-exceptions -fexceptions \
// RUN:     -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck -check-prefixes=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fcxx-exceptions -fexceptions \
// RUN:     -emit-llvm %s -o %t.ll
// RUN: FileCheck -check-prefixes=OGCG --input-file=%t.ll %s

// Delegating constructor with a non-trivial destructor and C++ exceptions enabled
// exercises CIRGenFunction::emitDelegatingCXXConstructorCall's NYI for EH cleanup
// around the delegated subobject initialization.

void mayThrow();

struct HasDtor {
  ~HasDtor() {}
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: emitDelegatingCXXConstructorCall: exception}}
  HasDtor() : HasDtor(0) { mayThrow(); }
  HasDtor(int);
};

HasDtor::HasDtor(int) {}

void force_default_ctor() { HasDtor x; }

// CIR: cir.func {{.*}} @_ZN7HasDtorC1Ev(%[[ARG0:.*]]: !cir.ptr<!rec_HasDtor>
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_HasDtor>, !cir.ptr<!cir.ptr<!rec_HasDtor>>, ["this", init]
// CIR:   cir.store %[[ARG0]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_HasDtor>>, !cir.ptr<!rec_HasDtor>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.call @_ZN7HasDtorC1Ei(%[[THIS]], %[[ZERO]]) : (!cir.ptr<!rec_HasDtor> {{.*}}, !s32i {{.*}}) -> ()
// CIR:   cir.cleanup.scope {
// CIR:     cir.call @_Z8mayThrowv() : () -> ()
// CIR:     cir.yield
// CIR:   } cleanup eh {
// CIR:     cir.call @_ZN7HasDtorD1Ev(%[[THIS]]) nothrow : (!cir.ptr<!rec_HasDtor> {{.*}}) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define {{.*}} void @_ZN7HasDtorC1Ev(ptr {{.*}} %[[ARG0:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   call void @_ZN7HasDtorC1Ei(ptr{{.*}} %[[THIS]], i32{{.*}} 0)
// LLVM:   br label %[[CLEANUP_SCOPE:.*]]
// LLVM: [[CLEANUP_SCOPE]]:
// LLVM:   invoke void @_Z8mayThrowv()
// LLVM:     to label %[[INVOKE_CONT:.*]] unwind label %[[INVOKE_UNWIND:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[EXIT_CLEANUP_SCOPE:.*]]
// LLVM: [[INVOKE_UNWIND]]:
// LLVM:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// LLVM:   %[[EXN_SEL:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// LLVM:   br label %[[EH_CLEANUP:.*]]
// LLVM: [[EH_CLEANUP]]:
// LLVM:   %[[EXN_PTR_PHI:.*]] = phi ptr [ %[[EXN_PTR]], %[[INVOKE_UNWIND]] ]
// LLVM:   %[[EXN_SEL_PHI:.*]] = phi i32 [ %[[EXN_SEL]], %[[INVOKE_UNWIND]] ]
// LLVM:   call void @_ZN7HasDtorD1Ev(ptr{{.*}} %[[THIS]])
// LLVM:   %[[EXN_PARTIAL:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR_PHI]], 0
// LLVM:   %[[EXN:.*]] = insertvalue { ptr, i32 } %[[EXN_PARTIAL]], i32 %[[EXN_SEL_PHI]], 1
// LLVM:   resume { ptr, i32 } %[[EXN]]
// LLVM: [[EXIT_CLEANUP_SCOPE]]:
// LLVM:   ret void
// LLVM: }

// OGCG: define {{.*}} void @_ZN7HasDtorC1Ev(ptr {{.*}} %[[ARG0:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSELECTOR_SLOT:.*]] = alloca i32
// OGCG:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   call void @_ZN7HasDtorC1Ei(ptr{{.*}} %[[THIS]], i32{{.*}} 0)
// OGCG:   invoke void @_Z8mayThrowv()
// OGCG:     to label %[[INVOKE_CONT:.*]] unwind label %[[INVOKE_UNWIND:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   ret void
// OGCG: [[INVOKE_UNWIND]]:
// OGCG:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// OGCG:   store ptr %[[EXN_PTR]], ptr %[[EXN_SLOT]]
// OGCG:   %[[EXN_INT:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// OGCG:   store i32 %[[EXN_INT]], ptr %[[EHSELECTOR_SLOT]]
// OGCG:   call void @_ZN7HasDtorD1Ev(ptr{{.*}} %[[THIS]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[EXN_PTR2:.*]] = load ptr, ptr %[[EXN_SLOT]]
// OGCG:   %[[EHSELECTOR:.*]] = load i32, ptr %[[EHSELECTOR_SLOT]]
// OGCG:   %[[EXN_PARTIAL:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR2]], 0
// OGCG:   %[[EXN:.*]] = insertvalue { ptr, i32 } %[[EXN_PARTIAL]], i32 %[[EHSELECTOR]], 1
// OGCG:   resume { ptr, i32 } %[[EXN]]
// OGCG: }
