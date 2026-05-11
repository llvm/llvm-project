// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP

int get_i();
struct CtorDtor {
  constexpr CtorDtor(int i) : i(i){}
  ~CtorDtor(){}
    int i;
};

thread_local CtorDtor tls_cd = 5;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW6tls_cd", "_ZTH6tls_cd"> @tls_cd = #cir.const_record<{#cir.int<5> : !s32i}> : !rec_CtorDtor dtor {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorD1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP: }

thread_local CtorDtor tls_cd_dyn = get_i();
// CIR-BEFORE-LPP:  cir.global external tls_dyn dyn_tls_refs = <"_ZTW10tls_cd_dyn", "_ZTH10tls_cd_dyn"> @tls_cd_dyn = ctor : !rec_CtorDtor {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:    %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:    cir.call @_ZN8CtorDtorC1Ei(%[[GET_GLOB]], %[[CALL]]) : (!cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:  } dtor {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:    cir.call @_ZN8CtorDtorD1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP:  }

thread_local CtorDtor &tls_cd_ref = tls_cd_dyn;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW10tls_cd_ref", "_ZTH10tls_cd_ref"> @tls_cd_ref = ctor : !cir.ptr<!rec_CtorDtor> {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_ref : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR-BEFORE-LPP: }

thread_local CtorDtor tls_cd_dyn_not_used = get_i();
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW19tls_cd_dyn_not_used", "_ZTH19tls_cd_dyn_not_used"> @tls_cd_dyn_not_used = ctor : !rec_CtorDtor {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn_not_used : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorC1Ei(%[[GET_GLOB]], %[[CALL]])
// CIR-BEFORE-LPP: } dtor {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn_not_used : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorD1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP: }

void uses() {
  auto a = tls_cd;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_cd : !cir.ptr<!rec_CtorDtor>
  auto b = tls_cd_dyn;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
  auto c = tls_cd_ref;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_cd_ref : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
}
