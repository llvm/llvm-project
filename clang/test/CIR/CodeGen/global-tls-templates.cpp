// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP

int get_i();
struct CtorDtor {
  constexpr CtorDtor(int i) : i(i){}
  ~CtorDtor(){}
    int i;
};

template<typename T>
thread_local T tls_templ = {get_i()};

// CIR-BEFORE-LPP-LABEL:  cir.global linkonce_odr comdat tls_dyn dyn_tls_refs = <"_ZTW9tls_templIiE", "_ZTH9tls_templIiE", "_ZGV9tls_templIiE"> @_Z9tls_templIiE = ctor : !s32i {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templIiE : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:    %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:    cir.store{{.*}} %[[CALL]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP:  }
//
// CIR-BEFORE-LPP-LABEL:  cir.global linkonce_odr comdat tls_dyn dyn_tls_refs = <"_ZTW9tls_templI8CtorDtorE", "_ZTH9tls_templI8CtorDtorE", "_ZGV9tls_templI8CtorDtorE"> @_Z9tls_templI8CtorDtorE = ctor : !rec_CtorDtor {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:    %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:    cir.call @_ZN8CtorDtorC1Ei(%[[GET_GLOB]], %[[CALL]]) : (!cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:  } dtor {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:    cir.call @_ZN8CtorDtorD1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP:  }

// CIR-BEFORE-LPP-LABEL: cir.func{{.*}}@_Z4usesv
void uses() {
  auto x = tls_templ<int>;
// CIR-BEFORE-LPP: cir.get_global thread_local @_Z9tls_templIiE : !cir.ptr<!s32i>
  auto y = tls_templ<CtorDtor>;
// CIR-BEFORE-LPP: cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
}
