// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM,LLVM-BOTH
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG,LLVM-BOTH

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

// Wrapper: Ctor/Dtor
// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW9tls_templI8CtorDtorE() -> !cir.ptr<!rec_CtorDtor> {
// CIR-NOT:  cir.call @_ZTH9tls_templI8CtorDtorE() : () -> ()
// CIR:  %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
// CIR:  cir.return %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>
// CIR:}

// Wrapper: int
// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW9tls_templIiE() -> !cir.ptr<!s32i>
// CIR-NOT:   cir.call @_ZTH9tls_templIiE() : () -> () 
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templIiE : !cir.ptr<!s32i>
// CIR:   cir.return %[[GET_GLOB]] : !cir.ptr<!s32i>
// CIR: }

// Global: int
// CIR: cir.global linkonce_odr comdat tls_dyn dyn_tls_refs = <"_ZTW9tls_templIiE", "_ZTH9tls_templIiE", "_ZGV9tls_templIiE"> @_Z9tls_templIiE = #cir.int<0> : !s32i
// Global: Ctor/Dotr:
// CIR: cir.global linkonce_odr comdat tls_dyn dyn_tls_refs = <"_ZTW9tls_templI8CtorDtorE", "_ZTH9tls_templI8CtorDtorE", "_ZGV9tls_templI8CtorDtorE"> @_Z9tls_templI8CtorDtorE = #cir.zero : !rec_CtorDtor

// Globals:
// LLVM-BOTH-DAG: @_Z9tls_templIiE = linkonce_odr thread_local global i32 0, comdat, align 4
// LLVM-BOTH-DAG: @_Z9tls_templI8CtorDtorE = linkonce_odr thread_local global %struct.CtorDtor zeroinitializer, comdat, align 4

// Wrappers: Just opposite ordering, same check lines as LLVM.
// FIXME: OGCG has these set as 'comdat'. However, CIR doesn't lower comdat to
// LLVM, so it doesn't show up in the IR here.
// LLVM-LABEL: define weak_odr hidden {{.*}}ptr @_ZTW9tls_templI8CtorDtorE() {
// LLVM-NOT:   call void @_ZTH9tls_templI8CtorDtorE()
// LLVM:   call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templI8CtorDtorE)
// LLVM: }

// LLVM-LABEL: define weak_odr hidden {{.*}}ptr @_ZTW9tls_templIiE() {
// LLVM-NOT:   call void @_ZTH9tls_templIiE()
// LLVM:   call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templIiE)
// LLVM: }

// CIR-BEFORE-LPP-LABEL: cir.func{{.*}}@_Z4usesv
// CIR-LABEL: cir.func{{.*}}@_Z4usesv
// LLVM-BOTH-LABEL: define dso_local void @_Z4usesv()
void uses() {
  auto x = tls_templ<int>;
// CIR-BEFORE-LPP: cir.get_global thread_local @_Z9tls_templIiE : !cir.ptr<!s32i>
// CIR: cir.call @_ZTW9tls_templIiE() : () -> !cir.ptr<!s32i>
// LLVM-BOTH: call ptr @_ZTW9tls_templIiE()
  auto y = tls_templ<CtorDtor>;
// CIR-BEFORE-LPP: cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
// CIR: cir.call @_ZTW9tls_templI8CtorDtorE() : () -> !cir.ptr<!rec_CtorDtor>
// LLVM-BOTH: call ptr @_ZTW9tls_templI8CtorDtorE()
}

// OGCG-LABEL: define weak_odr hidden {{.*}}ptr @_ZTW9tls_templIiE() {{.*}} comdat {
// OGCG:   call void @_ZTH9tls_templIiE()
// OGCG:   call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templIiE)
// OGCG: }

// OGCG-LABEL: define weak_odr hidden {{.*}}ptr @_ZTW9tls_templI8CtorDtorE(){{.*}} comdat {
// OGCG:   call void @_ZTH9tls_templI8CtorDtorE()
// OGCG:   call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templI8CtorDtorE)
// OGCG: }
