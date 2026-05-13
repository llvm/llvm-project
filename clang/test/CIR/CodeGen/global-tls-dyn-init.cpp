// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-BOTH,LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-BOTH,OGCG

int get_i();
struct CtorDtor {
  constexpr CtorDtor(int i) : i(i){}
  ~CtorDtor(){}
    int i;
};

// Wrappers:
// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW19tls_cd_dyn_not_used() -> !cir.ptr<!rec_CtorDtor> {
// CIR-NOT: cir.call @_ZTH19tls_cd_dyn_not_used() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn_not_used : !cir.ptr<!rec_CtorDtor>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW10tls_cd_ref() -> !cir.ptr<!cir.ptr<!rec_CtorDtor>> {
// CIR-NOT: cir.call @_ZTH10tls_cd_ref() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_ref : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW10tls_cd_dyn() -> !cir.ptr<!rec_CtorDtor> {
// CIR-NOT: cir.call @_ZTH10tls_cd_dyn() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor> 

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW6tls_cd() -> !cir.ptr<!rec_CtorDtor> {
// CIR-NOT: cir.call @_ZTH6tls_cd() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd : !cir.ptr<!rec_CtorDtor>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>

// LLVM: define weak_odr hidden ptr @_ZTW19tls_cd_dyn_not_used() {
// LLVM-NOT:   call void @_ZTH19tls_cd_dyn_not_used()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_dyn_not_used)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//
// LLVM: define weak_odr hidden ptr @_ZTW10tls_cd_ref() {
// LLVM-NOT:   call void @_ZTH10tls_cd_ref()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_ref)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//
// LLVM: define weak_odr hidden ptr @_ZTW10tls_cd_dyn() {
// LLVM-NOT:   call void @_ZTH10tls_cd_dyn()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_dyn)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//
// LLVM: define weak_odr hidden ptr @_ZTW6tls_cd() {
// LLVM-NOT:   call void @_ZTH6tls_cd()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//

thread_local CtorDtor tls_cd = 5;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW6tls_cd", "_ZTH6tls_cd"> @tls_cd = #cir.const_record<{#cir.int<5> : !s32i}> : !rec_CtorDtor dtor {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorD1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP: }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW6tls_cd", "_ZTH6tls_cd"> @tls_cd = #cir.const_record<{#cir.int<5> : !s32i}> : !rec_CtorDtor

// OGCG: define internal void @[[TLS_CD_INIT:.*]]() {{.*}}{
// OGCG:   call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @tls_cd, ptr @__dso_handle)
// OGCG:   ret void

thread_local CtorDtor tls_cd_dyn = get_i();
// CIR-BEFORE-LPP:  cir.global external tls_dyn dyn_tls_refs = <"_ZTW10tls_cd_dyn", "_ZTH10tls_cd_dyn"> @tls_cd_dyn = ctor : !rec_CtorDtor {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:    %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:    cir.call @_ZN8CtorDtorC1Ei(%[[GET_GLOB]], %[[CALL]]) : (!cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:  } dtor {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:    cir.call @_ZN8CtorDtorD1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP:  }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW10tls_cd_dyn", "_ZTH10tls_cd_dyn"> @tls_cd_dyn = #cir.zero : !rec_CtorDtor

// OGCG: define internal void @[[TLS_CD_DYN_INIT:.*]]() {{.*}} {
// OGCG:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// OGCG:   call void @_ZN8CtorDtorC1Ei(ptr {{.*}}@tls_cd_dyn, i32 {{.*}}%[[CALL]])
// OGCG:   call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @tls_cd_dyn, ptr @__dso_handle)
// OGCG:   ret void

thread_local CtorDtor &tls_cd_ref = tls_cd_dyn;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW10tls_cd_ref", "_ZTH10tls_cd_ref"> @tls_cd_ref = ctor : !cir.ptr<!rec_CtorDtor> {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_ref : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR-BEFORE-LPP: }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW10tls_cd_ref", "_ZTH10tls_cd_ref"> @tls_cd_ref = #cir.ptr<null> : !cir.ptr<!rec_CtorDtor>

// OGCG: define internal void @[[TLS_CD_REF_INIT:.*]]() {{.*}} {
// OGCG:   %[[CALL:.*]] = call ptr @_ZTW10tls_cd_dyn()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_cd_ref)
// OGCG:   store ptr %[[CALL]], ptr %[[GET_GLOB]], align 8
// OGCG:   ret void

// OGCG: define weak_odr hidden noundef ptr @_ZTW10tls_cd_dyn() {{.*}} comdat {
// OGCG:   call void @_ZTH10tls_cd_dyn()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_cd_dyn)
// OGCG:   ret ptr %[[GET_GLOB]]
// OGCG: }

thread_local CtorDtor tls_cd_dyn_not_used = get_i();
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW19tls_cd_dyn_not_used", "_ZTH19tls_cd_dyn_not_used"> @tls_cd_dyn_not_used = ctor : !rec_CtorDtor {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn_not_used : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorC1Ei(%[[GET_GLOB]], %[[CALL]])
// CIR-BEFORE-LPP: } dtor {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn_not_used : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorD1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP: }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW19tls_cd_dyn_not_used", "_ZTH19tls_cd_dyn_not_used"> @tls_cd_dyn_not_used = #cir.zero : !rec_CtorDtor

// OGCG: define internal void @[[TLS_CD_DYN_NOT_USED_INIT:.*]]() {{.*}} {
// OGCG:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// OGCG:   call void @_ZN8CtorDtorC1Ei(ptr {{.*}}@tls_cd_dyn_not_used, i32 {{.*}}%[[CALL]])
// OGCG:   call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @tls_cd_dyn_not_used, ptr @__dso_handle)
// OGCG:   ret void

void uses() {
  auto a = tls_cd;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_cd : !cir.ptr<!rec_CtorDtor>
// CIR: cir.call @_ZTW6tls_cd() : () -> !cir.ptr<!rec_CtorDtor>
// LLVM-BOTH: call ptr @_ZTW6tls_cd()
  auto b = tls_cd_dyn;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR: cir.call @_ZTW10tls_cd_dyn() : () -> !cir.ptr<!rec_CtorDtor>
// LLVM-BOTH: call ptr @_ZTW10tls_cd_dyn()
  auto c = tls_cd_ref;
// CIR-BEFORE-LPP: cir.get_global thread_local @tls_cd_ref : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR: cir.call @_ZTW10tls_cd_ref() : () -> !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// LLVM-BOTH: call ptr @_ZTW10tls_cd_ref()
}

// OGCG: define weak_odr hidden noundef ptr @_ZTW6tls_cd() {{.*}} comdat {
// OGCG:   call void @_ZTH6tls_cd()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_cd)
// OGCG:   ret ptr %[[GET_GLOB]]
// OGCG: }
//
// OGCG: define weak_odr hidden noundef ptr @_ZTW10tls_cd_ref() {{.*}} comdat {
// OGCG:   call void @_ZTH10tls_cd_ref()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_cd_ref)
// OGCG:   %[[LOAD_GLOB:.*]] = load ptr, ptr %[[GET_GLOB]], align 8
// OGCG:   ret ptr %[[LOAD_GLOB]]
// OGCG: }
//
// OGCG: define internal void @__tls_init() {{.*}} {
// OGCG:   %[[GET_GUARD:.*]] = load i8, ptr @__tls_guard, align 1
// OGCG:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GET_GUARD]], 0
// OGCG:   br i1 %[[IS_UNINIT]]
// OGCG
// OGCG:   store i8 1, ptr @__tls_guard, align 1
// OGCG:   call void @[[TLS_CD_INIT]]()
// OGCG:   call void @[[TLS_CD_DYN_INIT]]()
// OGCG:   call void @[[TLS_CD_REF_INIT]]()
// OGCG:   call void @[[TLS_CD_DYN_NOT_USED_INIT]]()
//
// OGCG: define weak_odr hidden noundef ptr @_ZTW19tls_cd_dyn_not_used() {{.*}} comdat {
// OGCG:   call void @_ZTH19tls_cd_dyn_not_used()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}} ptr @llvm.threadlocal.address.p0(ptr {{.*}} @tls_cd_dyn_not_used)
// OGCG:   ret ptr %[[GET_GLOB]]
// OGCG: }
