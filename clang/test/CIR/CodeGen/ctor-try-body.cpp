// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM,OGCG

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

// CIR: cir.func {{.*}}@_ZN9HasThingsC2ERK4Ctor(%[[THIS_ARG:.*]]: !cir.ptr<!rec_HasThings> {{.*}}, %[[C_ARG:.*]]: !cir.ptr<!rec_Ctor> {{.*}}) {{.*}}special_member<#cir.cxx_ctor<!rec_HasThings, custom>>{{.*}} {
// CIR-NEXT:  %[[THIS_ALLOC:.*]] = cir.alloca !cir.ptr<!rec_HasThings>, !cir.ptr<!cir.ptr<!rec_HasThings>>, ["this", init]
// CIR-NEXT:  %[[C_ALLOC:.*]] = cir.alloca !cir.ptr<!rec_Ctor>, !cir.ptr<!cir.ptr<!rec_Ctor>>, ["c", init, const]
// CIR-NEXT:  cir.store %[[THIS_ARG]], %[[THIS_ALLOC]] : !cir.ptr<!rec_HasThings>, !cir.ptr<!cir.ptr<!rec_HasThings>>
// CIR-NEXT:  cir.store %[[C_ARG]], %[[C_ALLOC]] : !cir.ptr<!rec_Ctor>, !cir.ptr<!cir.ptr<!rec_Ctor>>
// CIR-NEXT:  %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOC]] : !cir.ptr<!cir.ptr<!rec_HasThings>>, !cir.ptr<!rec_HasThings>
// CIR-NEXT:  cir.scope {
// CIR-NEXT:    cir.try {
// CIR-NEXT:      %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS_LOAD]] : !cir.ptr<!rec_HasThings> nonnull [0] -> !cir.ptr<!rec_Base>
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
// LLVMCIR:   call void @__cxa_end_catch()
// OGCG:   invoke void @__cxa_end_catch()
};

void foo() {
  Ctor ct;
  HasThings ht(ct);
}
