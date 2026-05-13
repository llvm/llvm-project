// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP

int get_i();
struct CtorDtor {
  constexpr CtorDtor(int i) : i(i){}
  ~CtorDtor(){}
    int i;
};

thread_local int tls_int = 5;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW7tls_int", "_ZTH7tls_int"> @tls_int = #cir.int<5> : !s32i

thread_local int tls_int_dyn = get_i();
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW11tls_int_dyn", "_ZTH11tls_int_dyn"> @tls_int_dyn = ctor : !s32i {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_dyn : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP: }

thread_local int &tls_int_ref = tls_int_dyn;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW11tls_int_ref", "_ZTH11tls_int_ref"> @tls_int_ref = ctor : !cir.ptr<!s32i> {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_ref : !cir.ptr<!cir.ptr<!s32i>>
// CIR-BEFORE-LPP:   %[[GET_OTHER:.*]] = cir.get_global thread_local @tls_int_dyn : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[GET_OTHER]], %[[GET_GLOB]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-BEFORE-LPP: }

thread_local int tls_int_self_init = tls_int_self_init + get_i();
// CIR-BEFORE-LPP:  cir.global external tls_dyn dyn_tls_refs = <"_ZTW17tls_int_self_init", "_ZTH17tls_int_self_init"> @tls_int_self_init = ctor : !s32i {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_int_self_init : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:    %[[GET_SELF:.*]] = cir.get_global thread_local @tls_int_self_init : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:    %[[LOAD_SELF:.*]] = cir.load {{.*}}%[[GET_SELF]] : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE-LPP:    %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:    %[[ADD:.*]] = cir.add nsw %[[LOAD_SELF]], %[[CALL]] : !s32i
// CIR-BEFORE-LPP:    cir.store {{.*}}%[[ADD]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP:  }

extern thread_local int definitely_inited = 5;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW17definitely_inited", "_ZTH17definitely_inited"> @definitely_inited = #cir.int<5> : !s32i

extern thread_local int definitely_inited_dyn = get_i();
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW21definitely_inited_dyn", "_ZTH21definitely_inited_dyn"> @definitely_inited_dyn = ctor : !s32i {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @definitely_inited_dyn : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP: }

extern thread_local int maybe_inited;
// CIR-BEFORE-LPP: cir.global "private" external tls_dyn dyn_tls_refs = <"_ZTW12maybe_inited", "_ZTH12maybe_inited"> @maybe_inited : !s32i

void uses() {
  auto a = tls_int;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_int : !cir.ptr<!s32i>
  auto b = tls_int_dyn;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_int_dyn : !cir.ptr<!s32i>
  auto c = tls_int_ref;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_int_ref : !cir.ptr<!cir.ptr<!s32i>>
  auto d = tls_int_self_init;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_int_self_init : !cir.ptr<!s32i>
  auto e = maybe_inited;
// CIR-BEFORE-LPP: cir.get_global thread_local @maybe_inited : !cir.ptr<!s32i>
  auto f = definitely_inited;
// CIR-BEFORE-LPP: cir.get_global thread_local @definitely_inited : !cir.ptr<!s32i>
  auto g = definitely_inited_dyn;
// CIR-BEFORE-LPP: cir.get_global thread_local @definitely_inited_dyn : !cir.ptr<!s32i>
}
