// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

void external();
void inner();
void harmless() noexcept;

void target() noexcept {
  external();
}

// A function that cannot throw is wrapped in a try whose only handler
// terminates the program. The handler catches everything, so there is no
// permitted type list to check.
// CIR-LABEL: cir.func{{.*}} @_Z6targetv()
// CIR:         cir.try {
// CIR:           cir.call @_Z8externalv() : () -> ()
// CIR:           cir.yield
// CIR:         } catch all (%[[TOK:.*]]: !cir.eh_token
// CIR:           cir.eh.terminate %[[TOK]] : !cir.eh_token
// CIR:         }
// CIR:         cir.return

// LLVM-LABEL: define{{.*}} void @_Z6targetv()
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         invoke void @_Z8externalv()
// LLVM-NEXT:            to label %[[CONT:[^ ,]+]] unwind label %[[LPAD:[^ ,]+]]
// LLVM:       {{^}}[[CONT]]:
// LLVM:       {{^}}[[LPAD]]:
// LLVM-NEXT:    %{{.*}} = landingpad { ptr, i32 }
//      The catch-all clause accepts every exception, so there is no selector
//      comparison and no resume path.
// LLVM-NEXT:            catch ptr null
// LLVM-NOT:     icmp
// LLVM-NOT:     resume
// LLVM:         call void @__clang_call_terminate(ptr %{{.*}})
// LLVM-NEXT:    unreachable
// LLVM-NOT:     resume

// In C++17, 'throw()' is a terminate scope rather than an empty dynamic
// exception specification.
void target2() throw() {
  external();
}

// CIR-LABEL: cir.func{{.*}} @_Z7target2v()
// CIR:         cir.try {
// CIR:           cir.call @_Z8externalv() : () -> ()
// CIR:           cir.yield
// CIR:         } catch all (%[[TOK:.*]]: !cir.eh_token
// CIR:           cir.eh.terminate %[[TOK]] : !cir.eh_token
// CIR-NOT:     cir.eh.unexpected

// LLVM-LABEL: define{{.*}} void @_Z7target2v()
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         invoke void @_Z8externalv()
// LLVM-NEXT:            to label %{{[^ ,]+}} unwind label %[[LPAD:[^ ,]+]]
// LLVM:       {{^}}[[LPAD]]:
// LLVM-NEXT:    %{{.*}} = landingpad { ptr, i32 }
// LLVM-NEXT:            catch ptr null
// LLVM-NOT:     __cxa_call_unexpected
// LLVM:         call void @__clang_call_terminate(ptr %{{.*}})
// LLVM-NEXT:    unreachable

void permissive() noexcept {
  harmless();
}

// The wrapper try is emitted for every function that cannot throw, but when
// nothing in the body can throw there is no unwind edge for it to catch.
// CIR-LABEL: cir.func{{.*}} @_Z10permissivev()
// CIR-SAME: personality(@__gxx_personality_v0)
// CIR:         cir.try {
// CIR:           cir.call @_Z8harmlessv() nothrow : () -> ()
// CIR:         } catch all (%[[TOK:.*]]: !cir.eh_token
// CIR:           cir.eh.terminate %[[TOK]] : !cir.eh_token

// So it leaves no landing pad behind. It does leave the personality function
// that CIRGen set for the wrapper, which classic codegen omits because it only
// sets one where it emits a landing pad. The same difference shows up for a
// try statement whose body cannot throw, so it is not specific to exception
// specifications.
// LLVM-LABEL: define{{.*}} void @_Z10permissivev()
// LLVMCIR-SAME: personality ptr @__gxx_personality_v0
// OGCG-NOT:   personality
// LLVM:         call void @_Z8harmlessv()
// LLVM-NOT:     landingpad
// LLVM-NOT:     __clang_call_terminate
// LLVM:         ret void

void tailToNothrow() noexcept {
  [[clang::musttail]] return harmless();
}

// A musttail call replaces this function's frame, and with it the terminate
// handler, which is why the callee has to be non-throwing. A throwing callee
// is diagnosed, in attr-musttail-noexcept-mismatch.cpp.
// CIR-LABEL: cir.func{{.*}} @_Z13tailToNothrowv()
// CIR:         cir.try {
// CIR:           cir.call @_Z8harmlessv() musttail nothrow : () -> ()
// CIR-NEXT:      cir.return

// LLVM-LABEL: define{{.*}} void @_Z13tailToNothrowv()
// LLVM:         musttail call void @_Z8harmlessv()
// LLVM-NEXT:    ret void
// LLVM-NOT:     landingpad
// LLVM-NOT:     __clang_call_terminate

void outer() noexcept {
  try {
    inner();
  } catch (int) {
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z5outerv()
// CIR:         cir.try {
// CIR:           cir.scope {
// CIR:             cir.try {
// CIR:               cir.call @_Z5innerv() : () -> ()
// CIR:               cir.yield
// CIR:             } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>]
// CIR:             } unwind
// CIR:           }
// CIR:           cir.yield
// CIR:         } catch all (%[[TOK:.*]]: !cir.eh_token
// CIR:           cir.eh.terminate %[[TOK]] : !cir.eh_token

// LLVM-LABEL: define{{.*}} void @_Z5outerv()
// LLVM-SAME: personality ptr @__gxx_personality_v0
// LLVM:         invoke void @_Z5innerv()
// LLVM-NEXT:            to label %[[CONT:[^ ,]+]] unwind label %[[LPAD:[^ ,]+]]

// The normal path continues to the join that follows the try statement.
// LLVM:       {{^}}[[CONT]]:
// LLVM-NEXT:    br label %[[TRY_CONT:[^ ,]+]]

// The inner catch and the enclosing terminate scope share a single landing
// pad, whose clauses are the catch clause followed by the catch-all.
// LLVM:       {{^}}[[LPAD]]:
// LLVM-NEXT:    %{{.*}} = landingpad { ptr, i32 }
// LLVM-NEXT:            catch ptr @_ZTIi
// LLVM-NEXT:            catch ptr null

// The catch clause is tested first; an exception of any other type terminates.
// LLVM:         %[[TID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:         %[[MATCHES:.*]] = icmp eq i32 %{{.*}}, %[[TID]]
// LLVM-NEXT:    br i1 %[[MATCHES]], label %[[CATCH:[^ ,]+]], label %[[NOMATCH:[^ ,]+]]

// LLVM:       {{^}}[[CATCH]]:
// LLVM:         call ptr @__cxa_begin_catch(ptr %{{.*}})
// LLVM:         call void @__cxa_end_catch()

//      A caught int leaves the function normally, through the same join the
//      normal path uses. It is neither rethrown, which `outer` has no resume
//      path for, nor routed to the terminate handler.
// LLVM-NOT:     resume
// LLVM-NOT:     __clang_call_terminate
// LLVM:         br label %[[TRY_CONT]]

// The pipelines lay the join and the terminate handler out in opposite orders,
// and the CIR pipeline reaches each of them through a chain of empty
// forwarding blocks, so these checks follow the branches. Interleaving the two
// chains is what the layout requires, not what the control flow does.

// An exception of any other type is on its way to the terminate handler.
// LLVMCIR:    {{^}}[[NOMATCH]]:
// LLVMCIR:      br label %[[TERM_CHAIN:[^ ,]+]]

// The join leads to the return.
// LLVMCIR:    {{^}}[[TRY_CONT]]:
// LLVMCIR-NEXT: br label %[[RET_CHAIN:[^ ,]+]]
// LLVMCIR:    {{^}}[[RET_CHAIN]]:
// LLVMCIR-NEXT: br label %[[RET:[^ ,]+]]

// And the block the unmatched exception branched to terminates.
// LLVMCIR:    {{^}}[[TERM_CHAIN]]:
// LLVMCIR:      br label %[[TERM:[^ ,]+]]
// LLVMCIR:    {{^}}[[TERM]]:
// LLVMCIR:      call void @__clang_call_terminate(ptr %{{.*}})
// LLVMCIR-NEXT: unreachable

// LLVMCIR:    {{^}}[[RET]]:
// LLVMCIR-NEXT: ret void

// OGCG:       {{^}}[[TRY_CONT]]:
// OGCG-NEXT:    ret void
// OGCG:       {{^}}[[NOMATCH]]:
// OGCG:         call void @__clang_call_terminate(ptr %{{.*}})
// OGCG-NEXT:    unreachable

// LLVM-NOT:     resume
