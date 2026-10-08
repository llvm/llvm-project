// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM,OGCG

void mayThrow();

struct S {
  S(int) {}
  S();
  ~S() {}
};

S::S() try : S(1) {
  mayThrow();
} catch (...) {
}

// CIR: cir.func {{.*}} @_ZN1SC2Ev(
// CIR:   cir.scope {
// CIR:     cir.try {
// CIR:       cir.call @_ZN1SC2Ei(
// CIR:       cir.cleanup.scope {
// CIR:         cir.call @_Z8mayThrowv() : () -> ()
// CIR:         cir.yield
// CIR:       } cleanup eh {
// CIR:         cir.call @_ZN1SD2Ev({{.*}}) nothrow
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } catch all ({{.*}}: !cir.eh_token{{.*}}) {
// CIR:       %[[CATCH_TOK:.*]], %[[EXN_PTR:.*]] = cir.begin_catch {{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:       cir.cleanup.scope {
// CIR:         cir.throw
// CIR:         cir.unreachable
// CIR:       ^bb1:
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch %[[CATCH_TOK]] : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define {{.*}} void @_ZN1SC2Ev(
// LLVM:   invoke void @_ZN1SC2Ei(
// LLVM:   invoke void @_Z8mayThrowv()
// LLVM:   call void @_ZN1SD2Ev({{.*}})
// LLVM:   call ptr @__cxa_begin_catch(
// LLVM:   invoke void @__cxa_rethrow()
// LLVM:     to label %[[UNREACHABLE_DEST:.*]] unwind label %[[CLEANUP_LPAD:.*]]
// LLVM: [[CLEANUP_LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVMCIR:   call void @__cxa_end_catch()
// OGCG:   invoke void @__cxa_end_catch()
// LLVM:   resume { ptr, i32 }
// LLVM: [[UNREACHABLE_DEST]]:
// LLVM:   unreachable

struct Throws {
  Throws(int) {}
  Throws();
  ~Throws(){}
};

Throws::Throws() try : Throws(1) {
  mayThrow();
} catch (...) {
  throw 5;
}

// CIR: cir.func {{.*}} @_ZN6ThrowsC2Ev(
// CIR:   cir.scope {
// CIR:     cir.try {
// CIR:       cir.call @_ZN6ThrowsC2Ei(
// CIR:       cir.cleanup.scope {
// CIR:         cir.call @_Z8mayThrowv() : () -> ()
// CIR:         cir.yield
// CIR:       } cleanup eh {
// CIR:         cir.call @_ZN6ThrowsD2Ev({{.*}}) nothrow
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } catch all ({{.*}}: !cir.eh_token{{.*}}) {
// CIR:       %[[CATCH_TOK:.*]], %[[EXN_PTR:.*]] = cir.begin_catch {{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:       cir.cleanup.scope {
// CIR:         %[[EXCEPT:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR:         %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR:         cir.store align(16) %[[FIVE]], %[[EXCEPT]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.throw %[[EXCEPT]], %{{.*}}, %{{.*}} : !cir.ptr<!s32i>, !cir.ptr<!u8i>, !cir.ptr<!void>
// CIR:         cir.unreachable
// CIR:       ^bb1:
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch %[[CATCH_TOK]] : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define {{.*}} void @_ZN6ThrowsC2Ev(
// LLVM:   invoke void @_ZN6ThrowsC2Ei(
// LLVM:   invoke void @_Z8mayThrowv()
// LLVM:   call void @_ZN6ThrowsD2Ev({{.*}})
// LLVM:   call ptr @__cxa_begin_catch(
// LLVM:   call ptr @__cxa_allocate_exception(i64 4)
// LLVM:   store i32 5, ptr %[[EXCEPT:.*]], align 16
// LLVM:   invoke void @__cxa_throw(ptr %[[EXCEPT]], ptr @_ZTIi, ptr null)
// LLVM:     to label %[[UNREACHABLE_DEST:.*]] unwind label %[[CLEANUP_LPAD:.*]]
// LLVM: [[CLEANUP_LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVMCIR:   call void @__cxa_end_catch()
// OGCG:   invoke void @__cxa_end_catch()
// LLVM:   resume { ptr, i32 }
// LLVM: [[UNREACHABLE_DEST]]:
// LLVM:   unreachable

struct Ctor {
  Ctor();
};

struct FromCtor {
  FromCtor(const Ctor&);
};

void side_effect();
void side_effect2();

struct Base {
  Base();
};

struct HasThings : Base {
  FromCtor ct;

  HasThings(const Ctor &c)
    try : ct(c) {
    side_effect();
  } catch (...) {
    side_effect2();
  }

// CIR: cir.func {{.*}}@_ZN9HasThingsC2ERK4Ctor(%[[THIS_ARG:.*]]: !cir.ptr<!rec_HasThings> {{.*}}, %[[C_ARG:.*]]: !cir.ptr<!rec_Ctor> {{.*}}) {{.*}}func_info<#cir.cxx_ctor<!rec_HasThings, custom>>{{.*}} {
// CIR-NEXT:  %[[THIS_ALLOC:.*]] = cir.alloca "this" {{.*}} init : !cir.ptr<!cir.ptr<!rec_HasThings>>
// CIR-NEXT:  %[[C_ALLOC:.*]] = cir.alloca "c" {{.*}} init const : !cir.ptr<!cir.ptr<!rec_Ctor>>
// CIR-NEXT:  cir.store %[[THIS_ARG]], %[[THIS_ALLOC]] : !cir.ptr<!rec_HasThings>, !cir.ptr<!cir.ptr<!rec_HasThings>>
// CIR-NEXT:  cir.store %[[C_ARG]], %[[C_ALLOC]] : !cir.ptr<!rec_Ctor>, !cir.ptr<!cir.ptr<!rec_Ctor>>
// CIR-NEXT:  %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOC]] : !cir.ptr<!cir.ptr<!rec_HasThings>>, !cir.ptr<!rec_HasThings>
// CIR-NEXT:  cir.scope {
// CIR-NEXT:    cir.try {
// CIR-NEXT:      %[[BASE_ADDR:.*]] = cir.base_class_addr nonnull %[[THIS_LOAD]] [0] : !cir.ptr<!rec_HasThings> -> !cir.ptr<!rec_Base>
// CIR-NEXT:      cir.call @_ZN4BaseC2Ev(%[[BASE_ADDR]]) : (!cir.ptr<!rec_Base>{{.*}}) -> ()
// CIR-NEXT:      %[[FROMCTOR_ADDR:.*]] = cir.cast bitcast %[[THIS_LOAD]] : !cir.ptr<!rec_HasThings> -> !cir.ptr<!rec_FromCtor>
// CIR-NEXT:      %[[C_LOAD:.*]] = cir.load %[[C_ALLOC]] : !cir.ptr<!cir.ptr<!rec_Ctor>>, !cir.ptr<!rec_Ctor>
// CIR-NEXT:      cir.call @_ZN8FromCtorC1ERK4Ctor(%[[FROMCTOR_ADDR]], %[[C_LOAD]]) : (!cir.ptr<!rec_FromCtor> {{.*}}, !cir.ptr<!rec_Ctor> {{.*}}) -> ()
// CIR-NEXT:      cir.call @_Z11side_effectv() : () -> ()
// CIR-NEXT:      cir.yield
// CIR-NEXT:    } catch all (%[[CATCH_ARG:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT:      %[[CATCH_TOK:.*]], %[[EX_PTR:.*]] = cir.begin_catch %[[CATCH_ARG]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR-NEXT:      cir.cleanup.scope {
// CIR-NEXT:        cir.call @_Z12side_effect2v() : () -> ()
// CIR-NEXT:        cir.throw
// CIR-NEXT:        cir.unreachable
// CIR-NEXT:      ^bb1:
// CIR-NEXT:        cir.yield
// CIR-NEXT:      } cleanup all {
// CIR-NEXT:        cir.end_catch %[[CATCH_TOK]] : !cir.catch_token
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }
// CIR-NEXT:  }
// CIR-NEXT:  cir.return
// CIR-NEXT:}

// Note: This skips a LOT of lines, but otherwise dives into an absolutely
// 'normal' try/catch/etc block, which both differs between LLVM and OGCG, but
// isn't particularly relevant to the fact that we generate the base,
// initializers, and body all in a try block.
// LLVM: define linkonce_odr void @_ZN9HasThingsC2ERK4Ctor(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[C_ARG:.*]])
// LLVM:   invoke void @_ZN4BaseC2Ev(ptr {{.*}}%{{.*}})
// LLVM:   invoke void @_ZN8FromCtorC1ERK4Ctor(ptr {{.*}}%{{.*}}, ptr {{.*}}%{{.*}})
// LLVM:   invoke void @_Z11side_effectv()
// LLVM:   call ptr @__cxa_begin_catch(ptr %{{.*}})
// LLVM:   invoke void @_Z12side_effect2v()
// LLVM:   invoke void @__cxa_rethrow()
// LLVM:     unwind label %[[CLEANUP_LPAD:.*]]
// LLVM: [[CLEANUP_LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVMCIR:   call void @__cxa_end_catch()
// OGCG:   invoke void @__cxa_end_catch()
};

void foo() {
  Ctor ct;
  HasThings ht(ct);
}
