// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t-flat.cir
// RUN: FileCheck --input-file=%t-flat.cir %s --check-prefix=CIR-FLAT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct Local { ~Local(); };
void testSwitchWithCleanup(int n) {
  Local x;
  throw 42;
}

// In CIR, the throw is emitted inside a `cir.cleanup.scope` whose cleanup
// region runs the destructor for `x` on the EH unwind path.

// CIR: cir.func{{.*}} @_Z21testSwitchWithCleanupi(%[[ARG:.*]]: !s32i
// CIR:   %[[N_ADDR:.*]] = cir.alloca "n" {{.*}} init : !cir.ptr<!s32i>
// CIR:   %[[X:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!rec_Local>
// CIR:   cir.store %[[ARG]], %[[N_ADDR]] : !s32i
// CIR:   cir.cleanup.scope {
// CIR:     %[[EXN:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR:     %[[VAL:.*]] = cir.const #cir.int<42> : !s32i
// CIR:     cir.store{{.*}} %[[VAL]], %[[EXN]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXN]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN5LocalD1Ev(%[[X]]) nothrow
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return

// After CFG flattening the cleanup scope is gone: the `cir.throw` becomes a
// `cir.try_throw` whose normal destination is a literally-unreachable block
// at the end of the function and whose unwind destination is the EH
// cleanup chain that runs the destructor and then resumes.

// CIR-FLAT: cir.func{{.*}} @_Z21testSwitchWithCleanupi(%[[ARG:.*]]: !s32i
// CIR-FLAT:   %[[N_ADDR:.*]] = cir.alloca "n" {{.*}} init : !cir.ptr<!s32i>
// CIR-FLAT:   %[[X:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!rec_Local>
// CIR-FLAT:   cir.store %[[ARG]], %[[N_ADDR]]
// CIR-FLAT:   cir.br ^[[BODY:.+]]
// CIR-FLAT: ^[[BODY]]:
// CIR-FLAT:   %[[EXN:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR-FLAT:   %[[VAL:.*]] = cir.const #cir.int<42> : !s32i
// CIR-FLAT:   cir.store{{.*}} %[[VAL]], %[[EXN]]
// CIR-FLAT:   cir.try_throw %[[EXN]] : !cir.ptr<!s32i>, @_ZTIi ^[[UNREACH:.+]], ^[[UNWIND:.+]]
// CIR-FLAT: ^[[UNWIND]]:
// CIR-FLAT:   %[[ET:.*]] = cir.eh.initiate cleanup : !cir.eh_token
// CIR-FLAT:   cir.br ^[[CLEANUP:.+]](%[[ET]] : !cir.eh_token)
// CIR-FLAT: ^[[CLEANUP]](%[[ET2:.*]]: !cir.eh_token):
// CIR-FLAT:   %[[CT:.*]] = cir.begin_cleanup %[[ET2]]
// CIR-FLAT:   cir.call @_ZN5LocalD1Ev(%[[X]]) nothrow
// CIR-FLAT:   cir.end_cleanup %[[CT]]
// CIR-FLAT:   cir.resume %[[ET2]]
// CIR-FLAT:   cir.return
// CIR-FLAT: ^[[UNREACH]]:
// CIR-FLAT:   cir.unreachable

// In LLVM IR the throw becomes an `invoke @__cxa_throw` whose unwind
// destination is a landingpad with a `cleanup` clause, runs the destructor,
// and resumes. The "normal" destination of the invoke is a block containing
// just `unreachable`.

// LLVM: define dso_local void @_Z21testSwitchWithCleanupi(i32 noundef %{{.*}}) {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[X:.*]] = alloca %struct.Local
// LLVM:   %[[EXN:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// LLVM:   store i32 42, ptr %[[EXN]]
// LLVM:   invoke void @__cxa_throw(ptr %[[EXN]], ptr @_ZTIi, ptr null)
// LLVM-NEXT: to label %[[NORMAL:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[LPAD]]:
// LLVM:   %{{.*}} = landingpad { ptr, i32 }
// LLVM-NEXT: cleanup
// LLVM:   call void @_ZN5LocalD1Ev(ptr {{.*}} %[[X]])
// LLVM:   resume { ptr, i32 } %{{.*}}
// LLVM: [[NORMAL]]:
// LLVM:   unreachable

// OGCG produces equivalent IR: an `invoke __cxa_throw` whose unwind path
// is a `cleanup` landingpad that calls the destructor and resumes.

// OGCG: define dso_local void @_Z21testSwitchWithCleanupi(i32 noundef %n) {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[X:.*]] = alloca %struct.Local
// OGCG:   %[[EXN:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// OGCG:   store i32 42, ptr %[[EXN]]
// OGCG:   invoke void @__cxa_throw(ptr %[[EXN]], ptr @_ZTIi, ptr null)
// OGCG-NEXT: to label %[[NORMAL:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[LPAD]]:
// OGCG:   %{{.*}} = landingpad { ptr, i32 }
// OGCG-NEXT: cleanup
// OGCG:   call void @_ZN5LocalD1Ev(ptr {{.*}} %[[X]])
// OGCG:   resume { ptr, i32 } %{{.*}}
// OGCG: [[NORMAL]]:
// OGCG:   unreachable
