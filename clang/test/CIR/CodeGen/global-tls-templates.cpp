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
// CIR:  cir.call @_ZTH9tls_templI8CtorDtorE() : () -> ()
// CIR:  %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
// CIR:  cir.return %[[GET_GLOB]] : !cir.ptr<!rec_CtorDtor>
// CIR:}

// Alias: Ctor/Dtor: 
// CIR: cir.func linkonce_odr @_ZTH9tls_templI8CtorDtorE() alias(@[[CTOR_DTOR_INIT:[^)]*]])
// TLS Guard: Ctor/Dtor:
// CIR: cir.global "private" linkonce_odr comdat tls_dyn @_ZGV9tls_templI8CtorDtorE = #cir.int<0> : !s64i

// Wrapper: int
// CIR-LABEL: cir.func comdat weak_odr private hidden @_ZTW9tls_templIiE() -> !cir.ptr<!s32i>
// CIR:   cir.call @_ZTH9tls_templIiE() : () -> () 
// CIR:   %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templIiE : !cir.ptr<!s32i>
// CIR:   cir.return %[[GET_GLOB]] : !cir.ptr<!s32i>
// CIR: }

// Alias: int
// CIR: cir.func linkonce_odr @_ZTH9tls_templIiE() alias(@[[INT_INIT:[^)]*]])
// TLS Guard: int
// CIR: cir.global "private" linkonce_odr comdat tls_dyn @_ZGV9tls_templIiE = #cir.int<0> : !s64i

// Global: int
// CIR: cir.global linkonce_odr comdat tls_dyn dyn_tls_refs = <"_ZTW9tls_templIiE", "_ZTH9tls_templIiE", "_ZGV9tls_templIiE"> @_Z9tls_templIiE = #cir.int<0> : !s32i

// Init Func: int
// CIR:  cir.func internal private @[[INT_INIT]]() {
// CIR:    %[[GET_GUARD:.*]] = cir.get_global thread_local @_ZGV9tls_templIiE : !cir.ptr<!s64i>
// CIR:    %[[GUARD_CAST:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR:    %[[LOAD_GUARD:.*]] = cir.load align(8) %[[GUARD_CAST]] : !cir.ptr<!s8i>, !s8i
// CIR:    %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR:    %[[ISUNINIT:.*]] = cir.cmp eq %[[LOAD_GUARD]], %[[ZERO]] : !s8i
// CIR:    cir.if %[[ISUNINIT]] {
// CIR:      %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:      cir.store %[[ONE]], %[[GET_GUARD]] : !s64i, !cir.ptr<!s64i>
// CIR:      %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templIiE : !cir.ptr<!s32i>
// CIR:      %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR:      cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !s32i, !cir.ptr<!s32i>
// CIR:    }
// CIR:    cir.return
// CIR:  }


// Global: Ctor/Dotr:
// CIR: cir.global linkonce_odr comdat tls_dyn dyn_tls_refs = <"_ZTW9tls_templI8CtorDtorE", "_ZTH9tls_templI8CtorDtorE", "_ZGV9tls_templI8CtorDtorE"> @_Z9tls_templI8CtorDtorE = #cir.zero : !rec_CtorDtor

// Init Func: Ctor/Dtor:
// CIR: cir.func internal private @[[CTOR_DTOR_INIT]]() {
// CIR:   %[[GET_GUARD:.*]] = cir.get_global thread_local @_ZGV9tls_templI8CtorDtorE : !cir.ptr<!s64i>
// CIR:    %[[GUARD_CAST:.*]] = cir.cast bitcast %[[GET_GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR:    %[[LOAD_GUARD:.*]] = cir.load align(8) %[[GUARD_CAST]] : !cir.ptr<!s8i>, !s8i
// CIR:    %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR:    %[[ISUNINIT:.*]] = cir.cmp eq %[[LOAD_GUARD]], %[[ZERO]] : !s8i
// CIR:    cir.if %[[ISUNINIT]] {
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:     cir.store %[[ONE]], %[[GET_GUARD]] : !s64i, !cir.ptr<!s64i>
// CIR:     %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
// CIR:     %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {llvm.noundef})
// CIR:     cir.call @_ZN8CtorDtorC1Ei(%[[GET_GLOB]], %[[CALL]]) : (!cir.ptr<!rec_CtorDtor> {{.*}}, !s32i {llvm.noundef}) -> ()
// CIR:     %[[GET_GLOB:.*]] = cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
// CIR:     %[[GET_DTOR:.*]] = cir.get_global @_ZN8CtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>>
// CIR:     %[[DTOR_FPTR:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:     %[[GLOB_DECAY:.*]] = cir.cast bitcast %[[GET_GLOB:.*]] : !cir.ptr<!rec_CtorDtor> -> !cir.ptr<!void>
// CIR:     %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:     cir.call @__cxa_thread_atexit(%[[DTOR_FPTR]], %[[GLOB_DECAY]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:   }
// CIR:   cir.return
// CIR: }

// FIXME: These have inconsistent COMDAT with classic codegen, but we don't
// currently specify 'comdat' with a name.
// Guards:
// LLVM-BOTH-DAG: @_ZGV9tls_templI8CtorDtorE = linkonce_odr thread_local global i64 0, comdat{{.*}}, align 8
// LLVM-BOTH-DAG: @_ZGV9tls_templIiE = linkonce_odr thread_local global i64 0, comdat{{.*}}, align 8
// Globals:
// LLVM-BOTH-DAG: @_Z9tls_templIiE = linkonce_odr thread_local global i32 0, comdat, align 4
// LLVM-BOTH-DAG: @_Z9tls_templI8CtorDtorE = linkonce_odr thread_local global %struct.CtorDtor zeroinitializer, comdat, align 4

// Aliases:
// LLVM-BOTH-DAG: @_ZTH9tls_templI8CtorDtorE = linkonce_odr alias void (), ptr @[[CTOR_DTOR_INIT:.*]]
// LLVM-BOTH-DAG: @_ZTH9tls_templIiE = linkonce_odr alias void (), ptr @[[INT_INIT:.*]]

// OGCG Has this first, same check lines as LLVM.
// OGCG-LABEL: define dso_local void @_Z4usesv()
// OGCG: call ptr @_ZTW9tls_templIiE()
// OGCG: call ptr @_ZTW9tls_templI8CtorDtorE()

// Wrappers: Just opposite ordering, same check lines as LLVM.
// FIXME: OGCG has these set as 'comdat'. However, CIR doesn't lower comdat to
// LLVM, so it doesn't show up in the IR here.
// LLVM-LABEL: define weak_odr hidden {{.*}}ptr @_ZTW9tls_templI8CtorDtorE() {
// LLVM:   call void @_ZTH9tls_templI8CtorDtorE()
// LLVM:   call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templI8CtorDtorE)
// LLVM: }

// LLVM-LABEL: define weak_odr hidden {{.*}}ptr @_ZTW9tls_templIiE() {
// LLVM:   call void @_ZTH9tls_templIiE()
// LLVM:   call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templIiE)
// LLVM: }

// OGCG-LABEL: define weak_odr hidden {{.*}}ptr @_ZTW9tls_templIiE() {{.*}} comdat {
// OGCG:   call void @_ZTH9tls_templIiE()
// OGCG:   call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templIiE)
// OGCG: }

// OGCG-LABEL: define weak_odr hidden {{.*}}ptr @_ZTW9tls_templI8CtorDtorE(){{.*}} comdat {
// OGCG:   call void @_ZTH9tls_templI8CtorDtorE()
// OGCG:   call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templI8CtorDtorE)
// OGCG: }


// Inits: 
// Note: the differences here are mostly ordering, however there are a few diferences:
// 1- OGCG skipps the llvm.threadlocal call.  We've seen this elsewhere with thread local,
//    so I don't think it is problematic, as it just seems like an early opt.
// 2- For some reason OGCG generates guards as i64/i8 depending on platform (like with static-local),
//    but ALWAYS treats the load/stores as i8.  This is likely a 'bug' in OGCG, but one that
//    doesn't really matter at all.
// LLVM-BOTH: define internal void @[[INT_INIT]]()
// LLVM:   %[[GET_GUARD:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZGV9tls_templIiE)
// LLVM:   %[[LOAD_GUARD:.*]] = load i8, ptr %[[GET_GUARD]], align 8
// OGCG:   %[[LOAD_GUARD:.*]] = load i8, ptr @_ZGV9tls_templIiE, align 8
// LLVM-BOTH:   %[[ISUNINIT:.*]] = icmp eq i{{.*}} %[[LOAD_GUARD]], 0
// LLVM-BOTH:   br i1 %[[ISUNINIT]]
//
// LLVM:   store i64 1, ptr %[[GET_GUARD]], align 8
// OGCG:   store i8 1, ptr @_ZGV9tls_templIiE, align 8
// LLVM:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templIiE)
// LLVM:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// OGCG:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// OGCG:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templIiE)
// LLVM-BOTH:   store i32 %[[CALL]], ptr %[[GET_GLOB]]

// LLVM-BOTH: define internal void @[[CTOR_DTOR_INIT]]()
// LLVM:   %[[GET_GUARD:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZGV9tls_templI8CtorDtorE)
// LLVM:   %[[LOAD_GUARD:.*]] = load i8, ptr %[[GET_GUARD]], align 8
// OGCG:   %[[LOAD_GUARD:.*]] = load i8, ptr @_ZGV9tls_templI8CtorDtorE, align 8
// LLVM-BOTH:   %[[ISUNINIT:.*]] = icmp eq i{{.*}} %[[LOAD_GUARD]], 0
// LLVM-BOTH:   br i1 %[[ISUNINIT]]
//
// LLVM:   store i64 1, ptr %[[GET_GUARD]], align 8
// OGCG:  store i8 1, ptr @_ZGV9tls_templI8CtorDtorE, align 8
//
// LLVM:   %[[GET_GLOB:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_Z9tls_templI8CtorDtorE)
// LLVM-BOTH:   %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM:   call void @_ZN8CtorDtorC1Ei(ptr {{.*}}%[[GET_GLOB]], i32 {{.*}}%[[CALL]])
// OGCG:   call void @_ZN8CtorDtorC1Ei(ptr {{.*}}@_Z9tls_templI8CtorDtorE, i32 {{.*}}%[[CALL]])
//
// LLVM:   %[[GET_GLOB:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_Z9tls_templI8CtorDtorE)
// LLVM:   call void @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr %[[GET_GLOB]], ptr @__dso_handle)
// OGCG:   call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @_Z9tls_templI8CtorDtorE, ptr @__dso_handle)

// CIR-BEFORE-LPP-LABEL: cir.func{{.*}}@_Z4usesv
// CIR-LABEL: cir.func{{.*}}@_Z4usesv
// LLVM-LABEL: define dso_local void @_Z4usesv()
void uses() {
  auto x = tls_templ<int>;
// CIR-BEFORE-LPP: cir.get_global thread_local @_Z9tls_templIiE : !cir.ptr<!s32i>
// CIR: cir.call @_ZTW9tls_templIiE() : () -> !cir.ptr<!s32i>
// LLVM: call ptr @_ZTW9tls_templIiE()
  auto y = tls_templ<CtorDtor>;
// CIR-BEFORE-LPP: cir.get_global thread_local @_Z9tls_templI8CtorDtorE : !cir.ptr<!rec_CtorDtor>
// CIR: cir.call @_ZTW9tls_templI8CtorDtorE() : () -> !cir.ptr<!rec_CtorDtor>
// LLVM: call ptr @_ZTW9tls_templI8CtorDtorE()
}
