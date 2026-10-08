// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

void external();
void inner();

void target() throw(int) {
  external();
}

// CIR-LABEL: cir.func{{.*}} @_Z6targetv()
// CIR-SAME: personality(@__gxx_personality_v0)
// CIR:         cir.try {
// CIR:           cir.call @_Z8externalv() : () -> ()
// CIR:           cir.yield
// CIR:         } filter [#cir.global_view<@_ZTIi> : !cir.ptr<!u8i>]
// CIR-SAME:        (%[[TOK:.*]]: !cir.eh_token
// CIR:           cir.resume %[[TOK]] : !cir.eh_token
// CIR:         } unexpected (%[[UTOK:.*]]: !cir.eh_token
// CIR:           cir.eh.unexpected %[[UTOK]] : !cir.eh_token
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define{{.*}} void @_Z6targetv()
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         invoke void @_Z8externalv()
// LLVM-NEXT:            to label %[[CONT:[^ ,]+]] unwind label %[[LPAD:[^ ,]+]]
// LLVM:       {{^}}[[CONT]]:
// LLVM:       {{^}}[[LPAD]]:
// LLVM-NEXT:    %{{.*}} = landingpad { ptr, i32 }
// LLVM-NEXT:            filter [1 x ptr] [ptr @_ZTIi]
//      A negative selector means the personality routine rejected the
//      exception against the filter clause, which violates the specification.
// LLVM:         %[[FAILS:.*]] = icmp slt i32 %{{.*}}, 0
// LLVM-NEXT:    br i1 %[[FAILS]], label %[[UNEXPECTED:[^ ,]+]], label %[[RESUME:[^ ,]+]]

// The two pipelines emit the handler blocks in opposite orders.
// LLVMCIR:    {{^}}[[RESUME]]:
// LLVMCIR:      resume { ptr, i32 } %{{.*}}
// LLVMCIR:    {{^}}[[UNEXPECTED]]:
// LLVMCIR:      call void @__cxa_call_unexpected(ptr %{{.*}})
// LLVMCIR-NEXT: unreachable

// OGCG:       {{^}}[[UNEXPECTED]]:
// OGCG:         call void @__cxa_call_unexpected(ptr %{{.*}})
// OGCG-NEXT:    unreachable
// OGCG:       {{^}}[[RESUME]]:
// OGCG:         resume { ptr, i32 } %{{.*}}

void target2() throw() {
  external();
}

// CIR-LABEL: cir.func{{.*}} @_Z7target2v()
// CIR-SAME: personality(@__gxx_personality_v0)
// CIR:         cir.try {
// CIR:           cir.call @_Z8externalv() : () -> ()
// CIR:           cir.yield
// CIR:         } filter []
// CIR:           cir.unreachable
// CIR:         } unexpected (%[[UTOK:.*]]: !cir.eh_token
// CIR:           cir.eh.unexpected %[[UTOK]] : !cir.eh_token
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define{{.*}} void @_Z7target2v()
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         invoke void @_Z8externalv()
// LLVM-NEXT:            to label %[[CONT:[^ ,]+]] unwind label %[[LPAD:[^ ,]+]]
// LLVM:       {{^}}[[CONT]]:
// LLVM:       {{^}}[[LPAD]]:
// LLVM-NEXT:    %{{.*}} = landingpad { ptr, i32 }
// LLVM-NEXT:            filter [0 x ptr] zeroinitializer
//      An empty filter list permits nothing, so every exception violates the
//      specification. There is no selector comparison and no resume path.
// LLVM-NOT:     icmp
// LLVM-NOT:     resume
// LLVM:         call void @__cxa_call_unexpected(ptr %{{.*}})
// LLVM-NEXT:    unreachable
// LLVM-NOT:     resume

void outer() throw() {
  try {
    inner();
  } catch (int) {
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z5outerv()
// CIR-SAME: personality(@__gxx_personality_v0)
// CIR:         cir.try {
// CIR:           cir.scope {
// CIR:             cir.try {
// CIR:               cir.call @_Z5innerv() : () -> ()
// CIR:               cir.yield
// CIR:             } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>]
// CIR:             } unwind
// CIR:           }
// CIR:           cir.yield
// CIR:         } filter []
// CIR:           cir.unreachable
// CIR:         } unexpected
// CIR:           cir.eh.unexpected

// LLVM-LABEL: define{{.*}} void @_Z5outerv()
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         invoke void @_Z5innerv()
// LLVM-NEXT:            to label %[[CONT:[^ ,]+]] unwind label %[[LPAD:[^ ,]+]]
// LLVM:       {{^}}[[CONT]]:
// LLVM-NEXT:    br label %[[TRY_CONT:[^ ,]+]]

// The inner catch and the enclosing exception specification share a single
// landing pad, whose clauses are the catch clause followed by the filter.
// LLVM:       {{^}}[[LPAD]]:
// LLVM-NEXT:    %{{.*}} = landingpad { ptr, i32 }
// LLVM-NEXT:            catch ptr @_ZTIi
// LLVM-NEXT:            filter [0 x ptr] zeroinitializer

// The catch clause is tested first; a non-matching exception falls through to
// the exception specification.
// LLVM:         %[[TID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM-NEXT:    %[[MATCHES:.*]] = icmp eq i32 %{{.*}}, %[[TID]]
// LLVM-NEXT:    br i1 %[[MATCHES]], label %[[CATCH:[^ ,]+]], label %[[NOMATCH:[^ ,]+]]

// Classic codegen folds the empty filter into the non-matching edge.
// OGCG:       {{^}}[[NOMATCH]]:
// OGCG:         call void @__cxa_call_unexpected(ptr %{{.*}})
// OGCG-NEXT:    unreachable

// LLVM:       {{^}}[[CATCH]]:
// LLVM:         call ptr @__cxa_begin_catch(ptr %{{.*}})
// LLVM:         call void @__cxa_end_catch()
//      The handled exception falls through to the join rather than being
//      rethrown; `outer` is `throw()`, so it has no resume path.
// LLVM-NOT:     resume
// LLVM:         br label %[[TRY_CONT]]

// The CIR pipeline routes the non-matching exception through the
// specification's own dispatch block before reaching the unexpected handler.
// LLVMCIR:    {{^}}[[NOMATCH]]:
// LLVMCIR:      br label %[[FILTER_DISPATCH:[0-9]+]]
// LLVMCIR:    {{^}}[[TRY_CONT]]:
// LLVMCIR:    {{^}}[[FILTER_DISPATCH]]:
// LLVMCIR:      br label %[[UNEXPECTED:[0-9]+]]
// LLVMCIR:    {{^}}[[UNEXPECTED]]:
// LLVMCIR:      call void @__cxa_call_unexpected(ptr %{{.*}})
// LLVMCIR-NEXT: unreachable
// LLVMCIR:      ret void

// OGCG:       {{^}}[[TRY_CONT]]:
// OGCG-NEXT:    ret void
