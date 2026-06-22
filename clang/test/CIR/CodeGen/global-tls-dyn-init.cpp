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

// LLVM-BOTH-DAG: @__tls_guard = internal thread_local global i8 0, align 1
// LLVM-BOTH-DAG: @__dso_handle = external hidden global i8
// LLVM-BOTH-DAG: @tls_cd = thread_local global %struct.CtorDtor { i32 5 }, align 4
// LLVM-BOTH-DAG: @tls_cd_dyn = thread_local global %struct.CtorDtor zeroinitializer, align 4
// LLVM-BOTH-DAG: @tls_cd_ref = thread_local global ptr null, align 8
// LLVM-BOTH-DAG: @tls_cd_dyn_not_used = thread_local global %struct.CtorDtor zeroinitializer, align 4
//
// LLVM-BOTH-DAG: @_ZTH19tls_cd_dyn_not_used = alias void (), ptr @__tls_init
// LLVM-BOTH-DAG: @_ZTH10tls_cd_ref = alias void (), ptr @__tls_init
// LLVM-BOTH-DAG: @_ZTH10tls_cd_dyn = alias void (), ptr @__tls_init
// LLVM-BOTH-DAG: @_ZTH6tls_cd = alias void (), ptr @__tls_init

// Wrappers & aliases.
// CIR:       cir.global internal tls_dyn @__tls_guard = #cir.int<0> : !s8i {alignment = 1 : i64}
// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW19tls_cd_dyn_not_used() -> !cir.ptr<!rec_CtorDtor> {
// CIR: cir.call @_ZTH19tls_cd_dyn_not_used() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn_not_used : !cir.ptr<!rec_CtorDtor>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>
// CIR: cir.func @_ZTH19tls_cd_dyn_not_used() alias(@__tls_init)

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW10tls_cd_ref() -> !cir.ptr<!cir.ptr<!rec_CtorDtor>> {
// CIR: cir.call @_ZTH10tls_cd_ref() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_ref : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR: cir.func @_ZTH10tls_cd_ref() alias(@__tls_init)

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW10tls_cd_dyn() -> !cir.ptr<!rec_CtorDtor> {
// CIR: cir.call @_ZTH10tls_cd_dyn() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor> 
// CIR: cir.func @_ZTH10tls_cd_dyn() alias(@__tls_init)

// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW6tls_cd() -> !cir.ptr<!rec_CtorDtor> {
// CIR: cir.call @_ZTH6tls_cd() : () -> ()
// CIR: %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd : !cir.ptr<!rec_CtorDtor>
// CIR: cir.return %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>
// CIR: cir.func @_ZTH6tls_cd() alias(@__tls_init)

// CIR-LABEL: cir.func internal private @__tls_init() {
// CIR: %[[GET_GUARD:.*]] = cir.get_global thread_local @__tls_guard : !cir.ptr<!s8i>
// CIR: %[[LOAD_GUARD:.*]] = cir.load align(1) %[[GET_GUARD]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[CMP:.*]] = cir.cmp eq %[[LOAD_GUARD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[CMP]] {
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR:   cir.store %[[ONE]], %[[GET_GUARD]] : !s8i, !cir.ptr<!s8i>
// CIR:   cir.call @[[TLS_CD_INIT:.*]]() : () -> ()
// CIR:   cir.call @[[TLS_CD_DYN_INIT:.*]]() : () -> ()
// CIR:   cir.call @[[TLS_CD_REF_INIT:.*]]() : () -> ()
// CIR:   cir.call @[[TLS_CD_DYN_NOT_USED_INIT:.*]]() : () -> ()
// CIR:  }
// CIR:  cir.return

// LLVM: define weak_odr hidden ptr @_ZTW19tls_cd_dyn_not_used() {
// LLVM:   call void @_ZTH19tls_cd_dyn_not_used()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_dyn_not_used)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//
// LLVM: define weak_odr hidden ptr @_ZTW10tls_cd_ref() {
// LLVM:   call void @_ZTH10tls_cd_ref()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_ref)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//
// LLVM: define weak_odr hidden ptr @_ZTW10tls_cd_dyn() {
// LLVM:   call void @_ZTH10tls_cd_dyn()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_dyn)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//
// LLVM: define weak_odr hidden ptr @_ZTW6tls_cd() {
// LLVM:   call void @_ZTH6tls_cd()
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd)
// LLVM:   ret ptr %[[GET_GLOB]]
// LLVM: }
//
// LLVM: define internal void @__tls_init() {
// LLVM:   %[[GET_GUARD:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @__tls_guard)
// LLVM:   %[[LOAD_GUARD:.*]] = load i8, ptr %[[GET_GUARD]], align 1
// LLVM:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[LOAD_GUARD]], 0
// LLVM:   br i1 %[[IS_UNINIT]]
// LLVM
// LLVM:   store i8 1, ptr %[[GET_GUARD]], align 1
// LLVM:   call void @[[TLS_CD_INIT:.*]]()
// LLVM:   call void @[[TLS_CD_DYN_INIT:.*]]()
// LLVM:   call void @[[TLS_CD_REF_INIT:.*]]()
// LLVM:   call void @[[TLS_CD_DYN_NOT_USED_INIT:.*]]()

thread_local CtorDtor tls_cd = 5;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW6tls_cd", "_ZTH6tls_cd"> @tls_cd = #cir.const_record<{#cir.int<5> : !s32i}> : !rec_CtorDtor dtor {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorD1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP: }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW6tls_cd", "_ZTH6tls_cd"> @tls_cd = #cir.const_record<{#cir.int<5> : !s32i}> : !rec_CtorDtor
// CIR: cir.func internal private @[[TLS_CD_INIT]]() {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[GET_DTOR:.*]] = cir.get_global @_ZN8CtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>>
// CIR:   %[[DTOR_DECAY:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[GLOB_DECAY:.*]] = cir.cast bitcast %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor> -> !cir.ptr<!void>
// CIR:   %[[DSOHANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_thread_atexit(%[[DTOR_DECAY]], %[[GLOB_DECAY]], %[[DSOHANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:   cir.return
//
// LLVM: define internal void @[[TLS_CD_INIT]]() {
// OGCG: define internal void @[[TLS_CD_INIT:.*]]() {{.*}}{
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd)
// LLVM:   call void @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr %[[GET_GLOB]], ptr @__dso_handle)
// OGCG:   call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @tls_cd, ptr @__dso_handle)
// LLVM-BOTH:   ret void

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
// CIR: cir.func internal private @[[TLS_CD_DYN_INIT]]() {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR:   cir.call @_ZN8CtorDtorC1Ei(%[[GET_GLOB]], %[[CALL]])
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[GET_DTOR:.*]] = cir.get_global @_ZN8CtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>>
// CIR:   %[[DTOR_DECAY:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[GLOB_DECAY:.*]] = cir.cast bitcast %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor> -> !cir.ptr<!void>
// CIR:   %[[DSOHANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_thread_atexit(%[[DTOR_DECAY]], %[[GLOB_DECAY]], %[[DSOHANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:   cir.return
//
// LLVM: define internal void @[[TLS_CD_DYN_INIT]]() {
// OGCG: define internal void @[[TLS_CD_DYN_INIT:.*]]() {{.*}} {
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_dyn)
// LLVM-BOTH:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM:   call void @_ZN8CtorDtorC1Ei(ptr {{.*}}%[[GET_GLOB]], i32 {{.*}}%[[CALL]])
// OGCG:   call void @_ZN8CtorDtorC1Ei(ptr {{.*}}@tls_cd_dyn, i32 {{.*}}%[[CALL]])
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_dyn)
// LLVM:   call void @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr %[[GET_GLOB]], ptr @__dso_handle)
// OGCG:   call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @tls_cd_dyn, ptr @__dso_handle)
// LLVM-BOTH:   ret void

thread_local CtorDtor &tls_cd_ref = tls_cd_dyn;
// CIR-BEFORE-LPP: cir.global external tls_dyn dyn_tls_refs = <"_ZTW10tls_cd_ref", "_ZTH10tls_cd_ref"> @tls_cd_ref = ctor : !cir.ptr<!rec_CtorDtor> {
// CIR-BEFORE-LPP:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_ref : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.get_global thread_local @tls_cd_dyn : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR-BEFORE-LPP: }
// CIR: cir.global external tls_dyn dyn_tls_refs = <"_ZTW10tls_cd_ref", "_ZTH10tls_cd_ref"> @tls_cd_ref = #cir.ptr<null> : !cir.ptr<!rec_CtorDtor>
// CIR: cir.func internal private @[[TLS_CD_REF_INIT]]() {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_ref : !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR:   %[[GET_DYN:.*]] = cir.call @_ZTW10tls_cd_dyn() : () -> !cir.ptr<!rec_CtorDtor>
// CIR:   cir.store align(8) %[[GET_DYN]], %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CIR:   cir.return
//
// LLVM: define internal void @[[TLS_CD_REF_INIT]]() {
// OGCG: define internal void @[[TLS_CD_REF_INIT:.*]]() {{.*}} {
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_ref)
// LLVM-BOTH:   %[[CALL:.*]] = call ptr @_ZTW10tls_cd_dyn()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tls_cd_ref)
// LLVM-BOTH:   store ptr %[[CALL]], ptr %[[GET_GLOB]], align 8
// LLVM-BOTH:   ret void
//
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
// CIR: cir.func internal private @[[TLS_CD_DYN_NOT_USED_INIT]]() {
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn_not_used : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR:   cir.call @_ZN8CtorDtorC1Ei(%[[GET_GLOB]], %[[CALL]])
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @tls_cd_dyn_not_used : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[GET_DTOR:.*]] = cir.get_global @_ZN8CtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>>
// CIR:   %[[DTOR_DECAY:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[GLOB_DECAY:.*]] = cir.cast bitcast %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor> -> !cir.ptr<!void>
// CIR:   %[[DSOHANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_thread_atexit(%[[DTOR_DECAY]], %[[GLOB_DECAY]], %[[DSOHANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:   cir.return
//
// LLVM: define internal void @[[TLS_CD_DYN_NOT_USED_INIT]]() {
// OGCG: define internal void @[[TLS_CD_DYN_NOT_USED_INIT:.*]]() {{.*}} {
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_dyn_not_used)
// LLVM-BOTH:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM:   call void @_ZN8CtorDtorC1Ei(ptr {{.*}}%[[GET_GLOB]], i32 {{.*}}%[[CALL]])
// OGCG:   call void @_ZN8CtorDtorC1Ei(ptr {{.*}}@tls_cd_dyn_not_used, i32 {{.*}}%[[CALL]])
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @tls_cd_dyn_not_used)
// LLVM:   call void @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr %[[GET_GLOB]], ptr @__dso_handle)
// OGCG:   call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @tls_cd_dyn_not_used, ptr @__dso_handle)
// LLVM-BOTH:   ret void

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
//
