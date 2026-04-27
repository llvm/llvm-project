// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP,CIR-BOTH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR,CIR-BOTH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM,LLVM-BOTH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG,LLVM-BOTH

// Guard variables.
// CIR-DAG: cir.global "private" internal tls_dyn dso_local @_ZGVZ13test_ctordtoriE4init = #cir.int<0> : !s8i
// LLVM-BOTH-DAG: @_ZGVZ13test_ctordtoriE4init = internal thread_local global i8 0
// CIR-DAG: cir.global "private" internal tls_dyn dso_local @_ZGVZ13test_ctordtoriE10const_init = #cir.int<0> : !s8i
// LLVM-BOTH-DAG: @_ZGVZ13test_ctordtoriE10const_init = internal thread_local global i8 0
// CIR-DAG: cir.global "private" internal tls_dyn dso_local @_ZGVZ9test_dtoriE10const_init = #cir.int<0> : !s8i
// LLVM-BOTH-DAG: @_ZGVZ9test_dtoriE10const_init = internal thread_local global i8 0
// CIR-DAG: cir.global "private" internal tls_dyn dso_local @_ZGVZ9test_ctoriE4init = #cir.int<0> : !s8i
// LLVM-BOTH-DAG: @_ZGVZ9test_ctoriE4init = internal thread_local global i8 0
// CIR-DAG: cir.global "private" internal tls_dyn dso_local @_ZGVZ9test_ctoriE10const_init = #cir.int<0> : !s8i
// LLVM-BOTH-DAG: @_ZGVZ9test_ctoriE10const_init = internal thread_local global i8 0
// CIR-DAG: cir.global "private" internal tls_dyn dso_local @_ZGVZ8test_intiE4init = #cir.int<0> : !s8i
// LLVM-BOTH-DAG: @_ZGVZ8test_intiE4init = internal thread_local global i8 0

// CIR-BOTH-DAG: cir.global "private" internal tls_dyn dso_local static_local_guard<"_ZGVZ13test_ctordtoriE4init"> @_ZZ13test_ctordtoriE4init = #cir.zero : !rec_CtorDtor
// LLVM-BOTH-DAG: @_ZZ13test_ctordtoriE4init = internal thread_local global %struct.CtorDtor zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal tls_dyn dso_local static_local_guard<"_ZGVZ13test_ctordtoriE10const_init"> @_ZZ13test_ctordtoriE10const_init = #cir.zero : !rec_CtorDtor
// LLVM-BOTH-DAG: @_ZZ13test_ctordtoriE10const_init = internal thread_local global %struct.CtorDtor zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal tls_dyn dso_local static_local_guard<"_ZGVZ9test_dtoriE10const_init"> @_ZZ9test_dtoriE10const_init = #cir.zero : !rec_Dtor
// LLVM-BOTH-DAG: @_ZZ9test_dtoriE10const_init = internal thread_local global %struct.Dtor zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal tls_dyn dso_local static_local_guard<"_ZGVZ9test_ctoriE4init"> @_ZZ9test_ctoriE4init = #cir.zero : !rec_Ctor
// LLVM-BOTH-DAG: @_ZZ9test_ctoriE4init = internal thread_local global %struct.Ctor zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal tls_dyn dso_local static_local_guard<"_ZGVZ9test_ctoriE10const_init"> @_ZZ9test_ctoriE10const_init = #cir.zero : !rec_Ctor
// LLVM-BOTH-DAG: @_ZZ9test_ctoriE10const_init = internal thread_local global %struct.Ctor zeroinitializer
// CIR-BOTH-DAG: cir.global "private" internal tls_dyn dso_local static_local_guard<"_ZGVZ8test_intiE4init"> @_ZZ8test_intiE4init = #cir.int<0> : !s32i
// LLVM-BOTH-DAG: @_ZZ8test_intiE4init = internal thread_local global i32 0
// CIR-BOTH-DAG: cir.global "private" internal tls_dyn dso_local @_ZZ8test_intiE10const_init = #cir.int<5> : !s32i
// LLVM-BOTH-DAG: @_ZZ8test_intiE10const_init = internal thread_local global i32 5
int get_i();

void test_int(int param) {
  int local;
  thread_local int const_init = 5;
  thread_local int init = param + local + get_i();

// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z8test_inti(
// CIR-BOTH: %[[PARAM_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["param", init]
// CIR-BOTH: [[LOCAL_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["local"]
// CIR-BOTH: %[[GET_CONST_TLS:.*]] = cir.get_global thread_local @_ZZ8test_intiE10const_init : !cir.ptr<!s32i>
// CIR-BOTH: %[[GET_TLS:.*]] = cir.get_global thread_local static_local @_ZZ8test_intiE4init : !cir.ptr<!s32i>
//
// CIR-BEFORE-LPP: cir.local_init thread_local @_ZZ8test_intiE4init ctor {
//
// CIR: %[[GUARD:.*]] = cir.get_global thread_local @_ZGVZ8test_intiE4init : !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}%[[GUARD]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:   %[[GET_TLS_INIT:.*]] = cir.get_global thread_local static_local @_ZZ8test_intiE4init : !cir.ptr<!s32i>
// CIR-BOTH:   %[[LOAD_PARAM:.*]] = cir.load {{.*}}[[PARAM_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   %[[LOAD_LOCAL:.*]] = cir.load {{.*}}[[LOCAL_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   %[[ADD1:.*]] = cir.add nsw %[[LOAD_PARAM]], %[[LOAD_LOCAL]] : !s32i
// CIR-BOTH:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {{.*}})
// CIR-BOTH:   %[[ADD2:.*]] = cir.add nsw %[[ADD1]], %[[CALL]] : !s32i
// CIR-BOTH:   cir.store {{.*}}%[[ADD2]], %[[GET_TLS_INIT]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   cir.yield

// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR: cir.store %[[ONE]], %[[GUARD]] : !s8i, !cir.ptr<!s8i>
//
// CIR-BOTH: }

// LLVM-BOTH-LABEL: define dso_local void @_Z8test_inti(
// LLVM-BOTH: %[[PARAM_ALLOCA:.*]] = alloca i32
// LLVM-BOTH: %[[LOCAL_ALLOCA:.*]] = alloca i32
//
// OG just loads this without the builtin, but I don't believe it is meaningful.
// LLVM: %[[GUARD_ADDR:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZGVZ8test_intiE4init)
// LLVM: %[[GUARD_LOAD:.*]] = load i8, ptr %[[GUARD_ADDR]]
// OGCG: %[[GUARD_LOAD:.*]] = load i8, ptr @_ZGVZ8test_intiE4init
//
// LLVM-BOTH: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD_LOAD]], 0
// LLVM-BOTH: br i1 %[[IS_UNINIT]]
// 
// OG loads the TLS after the init, just before the store, again irrelevant.
// Also, the store to the guard variable is also without the call to
// llvm.threadlocal.address.
// LLVM: %[[GET_TLS_INIT:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZZ8test_intiE4init)
// LLVM-BOTH: %[[LOAD_PARAM:.*]] = load i32, ptr %[[PARAM_ALLOCA]]
// LLVM-BOTH: %[[LOAD_LOCAL:.*]] = load i32, ptr %[[LOCAL_ALLOCA]]
// LLVM-BOTH: %[[ADD1:.*]] = add nsw i32 %[[LOAD_PARAM]], %[[LOAD_LOCAL]]
// LLVM-BOTH: %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM-BOTH: %[[ADD2:.*]] = add nsw i32 %[[ADD1]], %[[CALL]]
// OGCG: %[[GET_TLS_INIT:.*]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_ZZ8test_intiE4init)
// LLVM-BOTH: store i32 %[[ADD2]], ptr %[[GET_TLS_INIT]]
// LLVM: store i8 1, ptr %[[GUARD_ADDR]]
// OGCG: store i8 1, ptr @_ZGVZ8test_intiE4init
}

struct Ctor {
  Ctor(int i);
    int i;
};

void test_ctor(int param) {
  int local;
  thread_local Ctor const_init = 5;
  thread_local Ctor init = param + local + get_i();
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z9test_ctori(
// CIR-BOTH: %[[PARAM_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["param", init]
// CIR-BOTH: [[LOCAL_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["local"]
// CIR-BOTH: %[[GET_CONST_TLS:.*]] = cir.get_global thread_local static_local @_ZZ9test_ctoriE10const_init : !cir.ptr<!rec_Ctor>
//
// CIR-BEFORE-LPP: cir.local_init thread_local @_ZZ9test_ctoriE10const_init ctor {
//
// CIR: %[[GUARD:.*]] = cir.get_global thread_local @_ZGVZ9test_ctoriE10const_init : !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}%[[GUARD]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:   %[[GET_CONST_TLS_INIT:.*]] = cir.get_global thread_local static_local @_ZZ9test_ctoriE10const_init : !cir.ptr<!rec_Ctor>
// CIR-BOTH:   %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR-BOTH:   cir.call @_ZN4CtorC1Ei(%[[GET_CONST_TLS_INIT]], %[[FIVE]]) : (!cir.ptr<!rec_Ctor> {{.*}}) -> ()
// CIR-BEFORE-LPP:   cir.yield
//
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR: cir.store %[[ONE]], %[[GUARD]] : !s8i, !cir.ptr<!s8i>
//
// CIR-BOTH: }
// CIR-BOTH: %[[GET_TLS:.*]] = cir.get_global thread_local static_local @_ZZ9test_ctoriE4init : !cir.ptr<!rec_Ctor>
//
// CIR-BEFORE-LPP: cir.local_init thread_local @_ZZ9test_ctoriE4init ctor {
//
// CIR: %[[GUARD:.*]] = cir.get_global thread_local @_ZGVZ9test_ctoriE4init : !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}%[[GUARD]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
//
// CIR-BOTH:   %[[GET_TLS_INIT:.*]] = cir.get_global thread_local static_local @_ZZ9test_ctoriE4init : !cir.ptr<!rec_Ctor>
// CIR-BOTH:   %[[LOAD_PARAM:.*]] = cir.load {{.*}}[[PARAM_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   %[[LOAD_LOCAL:.*]] = cir.load {{.*}}[[LOCAL_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BOTH:   %[[ADD1:.*]] = cir.add nsw %[[LOAD_PARAM]], %[[LOAD_LOCAL]] : !s32i
// CIR-BOTH:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {{.*}})
// CIR-BOTH:   %[[ADD2:.*]] = cir.add nsw %[[ADD1]], %[[CALL]] : !s32i
// CIR-BOTH:   cir.call @_ZN4CtorC1Ei(%[[GET_TLS_INIT]], %[[ADD2]]) : (!cir.ptr<!rec_Ctor> {{.*}}) -> ()
// CIR-BEFORE-LPP:   cir.yield
//
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR: cir.store %[[ONE]], %[[GUARD]] : !s8i, !cir.ptr<!s8i>
//
// CIR-BOTH: }

// LLVM-BOTH-LABEL: define dso_local void @_Z9test_ctori(
// LLVM-BOTH: %[[PARAM_ALLOCA:.*]] = alloca i32
// LLVM-BOTH: %[[LOCAL_ALLOCA:.*]] = alloca i32
// LLVM: %[[GUARD_ADDR:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZGVZ9test_ctoriE10const_init)
// LLVM: %[[GUARD_LOAD:.*]] = load i8, ptr %[[GUARD_ADDR]]
// OGCG: %[[GUARD_LOAD:.*]] = load i8, ptr @_ZGVZ9test_ctoriE10const_init
// LLVM-BOTH: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD_LOAD]], 0
// LLVM-BOTH: br i1 %[[IS_UNINIT]]
//
// LLVM: %[[GET_TLS_INIT:.*]] = call ptr @llvm.threadlocal.address.p0(ptr {{.*}}@_ZZ9test_ctoriE10const_init)
// LLVM: call void @_ZN4CtorC1Ei(ptr {{.*}}%[[GET_TLS_INIT]], i32 noundef 5)
// OGCG: call void @_ZN4CtorC1Ei(ptr {{.*}}@_ZZ9test_ctoriE10const_init, i32 noundef 5)
//
// LLVM: store i8 1, ptr %[[GUARD_ADDR]]
// OGCG: store i8 1, ptr @_ZGVZ9test_ctoriE10const_init
// LLVM-BOTH: br
//
// LLVM: %[[GUARD_ADDR:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZGVZ9test_ctoriE4init)
// LLVM: %[[GUARD_LOAD:.*]] = load i8, ptr %[[GUARD_ADDR]]
// OGCG: %[[GUARD_LOAD:.*]] = load i8, ptr @_ZGVZ9test_ctoriE4init
// LLVM-BOTH: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD_LOAD]], 0
// LLVM-BOTH: br i1 %[[IS_UNINIT]]
//
// LLVM: %[[GET_TLS_INIT:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZZ9test_ctoriE4init)
// LLVM-BOTH: %[[LOAD_PARAM:.*]] = load i32, ptr %[[PARAM_ALLOCA]]
// LLVM-BOTH: %[[LOAD_LOCAL:.*]] = load i32, ptr %[[LOCAL_ALLOCA]]
// LLVM-BOTH: %[[ADD1:.*]] = add nsw i32 %[[LOAD_PARAM]], %[[LOAD_LOCAL]]
// LLVM-BOTH: %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM-BOTH: %[[ADD2:.*]] = add nsw i32 %[[ADD1]], %[[CALL]]
// LLVM: call void @_ZN4CtorC1Ei(ptr {{.*}}%[[GET_TLS_INIT]], i32 noundef %[[ADD2]])
// OGCG: call void @_ZN4CtorC1Ei(ptr {{.*}}@_ZZ9test_ctoriE4init, i32 noundef %[[ADD2]])
// LLVM: store i8 1, ptr %[[GUARD_ADDR]]
// OGCG: store i8 1, ptr @_ZGVZ9test_ctoriE4init

}

struct Dtor {
  ~Dtor();
    int i;
};

void test_dtor(int param) {
  int local;
  thread_local Dtor const_init;
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z9test_dtori(
// CIR-BOTH: %[[PARAM_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["param", init]
// CIR-BOTH: [[LOCAL_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["local"]
// CIR-BOTH: %[[GET_TLS:.*]] = cir.get_global thread_local static_local @_ZZ9test_dtoriE10const_init : !cir.ptr<!rec_Dtor>
//
// CIR-BEFORE-LPP: cir.local_init thread_local @_ZZ9test_dtoriE10const_init dtor {
// CIR-BEFORE-LPP:   %[[GET_TLS_DEL:.*]] = cir.get_global thread_local static_local @_ZZ9test_dtoriE10const_init : !cir.ptr<!rec_Dtor>
// CIR-BEFORE-LPP:   cir.call @_ZN4DtorD1Ev(%[[GET_TLS_DEL]]) : (!cir.ptr<!rec_Dtor>) -> ()
// CIR-BEFORE-LPP:   cir.yield
// CIR-BEFORE_LLP: }
//
// CIR: %[[GUARD:.*]] = cir.get_global thread_local @_ZGVZ9test_dtoriE10const_init : !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}%[[GUARD]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
//
// CIR:   %[[GET_TLS_DEL:.*]] = cir.get_global thread_local static_local @_ZZ9test_dtoriE10const_init : !cir.ptr<!rec_Dtor>
// CIR:   %[[GET_DEL_FUNC:.*]] = cir.get_global @_ZN4DtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_Dtor>)>>
// CIR:   %[[DEL_FUNC_DECAY:.*]] = cir.cast bitcast %[[GET_DEL_FUNC]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_Dtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[TLS_DECAY:.*]] = cir.cast bitcast %[[GET_TLS_DEL]] : !cir.ptr<!rec_Dtor> -> !cir.ptr<!void>
// CIR:   %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_thread_atexit(%[[DEL_FUNC_DECAY]], %[[TLS_DECAY]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR:   cir.store %[[ONE]], %[[GUARD]] : !s8i, !cir.ptr<!s8i>
// CIR: }

// LLVM-BOTH-LABEL: define dso_local void @_Z9test_dtori(
// LLVM-BOTH: %[[PARAM_ALLOCA:.*]] = alloca i32
// LLVM-BOTH: %[[LOCAL_ALLOCA:.*]] = alloca i32
// LLVM: %[[GUARD_ADDR:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZGVZ9test_dtoriE10const_init)
// LLVM: %[[GUARD_LOAD:.*]] = load i8, ptr %[[GUARD_ADDR]]
// OGCG: %[[GUARD_LOAD:.*]] = load i8, ptr @_ZGVZ9test_dtoriE10const_init
// LLVM-BOTH: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD_LOAD]], 0
// LLVM-BOTH: br i1 %[[IS_UNINIT]]
//
// LLVM: %[[GET_TLS_DEL:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZZ9test_dtoriE10const_init)
// LLVM: call void @__cxa_thread_atexit(ptr @_ZN4DtorD1Ev, ptr %[[GET_TLS_DEL]], ptr @__dso_handle)
// OGCG: call i32 @__cxa_thread_atexit(ptr @_ZN4DtorD1Ev, ptr @_ZZ9test_dtoriE10const_init, ptr @__dso_handle)
// LLVM: store i8 1, ptr %[[GUARD_ADDR]]
// OGCG:store i8 1, ptr @_ZGVZ9test_dtoriE10const_init
}

struct CtorDtor {
  CtorDtor(int i);
  ~CtorDtor();
    int i;
};

void test_ctordtor(int param) {
  int local;
  thread_local CtorDtor const_init = 5;
  thread_local CtorDtor init = param + local + get_i();
// CIR-BOTH-LABEL: cir.func no_inline dso_local @_Z13test_ctordtori(
// CIR-BOTH: %[[PARAM_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["param", init]
// CIR-BOTH: [[LOCAL_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["local"]
// CIR-BOTH: %[[GET_CONST_TLS:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE10const_init : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LLP: cir.local_init thread_local @_ZZ13test_ctordtoriE10const_init ctor {
// CIR-BEFORE-LPP:   %[[GET_CONST_TLS_INIT:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE10const_init : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorC1Ei(%[[GET_CONST_TLS_INIT]], %[[FIVE:.*]]) : (!cir.ptr<!rec_CtorDtor> {{.*}}) -> ()
// CIR-BEFORE-LPP:   cir.yield
// CIR-BEFORE-LPP: } dtor 
// CIR-BEFORE-LPP:   %[[GET_TLS_DEL:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE10const_init : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorD1Ev(%[[GET_TLS_DEL]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP:   cir.yield
// CIR-BEFORE-LPP: }
//
// CIR: %[[GUARD:.*]] = cir.get_global thread_local @_ZGVZ13test_ctordtoriE10const_init : !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}%[[GUARD]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[GET_CONST_TLS_INIT:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE10const_init : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR:   cir.call @_ZN8CtorDtorC1Ei(%[[GET_CONST_TLS_INIT]], %[[FIVE:.*]]) : (!cir.ptr<!rec_CtorDtor> {{.*}}) -> ()
// CIR:   %[[GET_CONST_TLS_DEL:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE10const_init : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[GET_DEL_FUNC:.*]] = cir.get_global @_ZN8CtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>>
// CIR:   %[[DEL_FUNC_DECAY:.*]] = cir.cast bitcast %[[GET_DEL_FUNC]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[TLS_DECAY:.*]] = cir.cast bitcast %[[GET_CONST_TLS_DEL]] : !cir.ptr<!rec_CtorDtor> -> !cir.ptr<!void>
// CIR:   %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_thread_atexit(%[[DEL_FUNC_DECAY]], %[[TLS_DECAY]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR:   cir.store %[[ONE]], %[[GUARD]] : !s8i, !cir.ptr<!s8i>
// CIR: }
//
// CIR-BOTH: %[[GET_TLS:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE4init : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP: cir.local_init thread_local @_ZZ13test_ctordtoriE4init ctor {
// CIR-BEFORE-LPP:   %[[GET_TLS_INIT:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE4init : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   %[[LOAD_PARAM:.*]] = cir.load {{.*}}[[PARAM_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE-LPP:   %[[LOAD_LOCAL:.*]] = cir.load {{.*}}[[LOCAL_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE-LPP:   %[[ADD1:.*]] = cir.add nsw %[[LOAD_PARAM]], %[[LOAD_LOCAL]] : !s32i
// CIR-BEFORE-LPP:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {{.*}})
// CIR-BEFORE-LPP:   %[[ADD2:.*]] = cir.add nsw %[[ADD1]], %[[CALL]] : !s32i
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorC1Ei(%[[GET_TLS_INIT]], %[[ADD2]]) : (!cir.ptr<!rec_CtorDtor> {{.*}}) -> ()
// CIR-BEFORE-LPP:   cir.yield
// CIR-BEFORE-LPP: } dtor {
// CIR-BEFORE-LPP:   %[[GET_TLS_DEL:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE4init : !cir.ptr<!rec_CtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN8CtorDtorD1Ev(%[[GET_TLS_DEL]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CIR-BEFORE-LPP:   cir.yield
// CIR-BEFORE-LPP: }
//
// CIR: %[[GUARD:.*]] = cir.get_global thread_local @_ZGVZ13test_ctordtoriE4init : !cir.ptr<!s8i>
// CIR: %[[GUARD_LOAD:.*]] = cir.load {{.*}}%[[GUARD]] : !cir.ptr<!s8i>, !s8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]] : !s8i
// CIR: cir.if %[[IS_UNINIT]] {
// CIR:   %[[GET_TLS_INIT:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE4init : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[LOAD_PARAM:.*]] = cir.load {{.*}}[[PARAM_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[LOAD_LOCAL:.*]] = cir.load {{.*}}[[LOCAL_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[ADD1:.*]] = cir.add nsw %[[LOAD_PARAM]], %[[LOAD_LOCAL]] : !s32i
// CIR:   %[[CALL:.*]] = cir.call @_Z5get_iv() : () -> (!s32i {{.*}})
// CIR:   %[[ADD2:.*]] = cir.add nsw %[[ADD1]], %[[CALL]] : !s32i
// CIR:   cir.call @_ZN8CtorDtorC1Ei(%[[GET_TLS_INIT]], %[[ADD2]]) : (!cir.ptr<!rec_CtorDtor> {{.*}}) -> ()
// CIR:   %[[GET_TLS_DEL:.*]] = cir.get_global thread_local static_local @_ZZ13test_ctordtoriE4init : !cir.ptr<!rec_CtorDtor>
// CIR:   %[[GET_DEL_FUNC:.*]] = cir.get_global @_ZN8CtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>>
// CIR:   %[[DEL_FUNC_DECAY:.*]] = cir.cast bitcast %[[GET_DEL_FUNC]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_CtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[TLS_DECAY:.*]] = cir.cast bitcast %[[GET_TLS_DEL]] : !cir.ptr<!rec_CtorDtor> -> !cir.ptr<!void>
// CIR:   %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_thread_atexit(%[[DEL_FUNC_DECAY]], %[[TLS_DECAY]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR:   cir.store %[[ONE]], %[[GUARD]] : !s8i, !cir.ptr<!s8i>
// CIR: }

// LLVM-BOTH-LABEL: define dso_local void @_Z13test_ctordtori(
// LLVM-BOTH: %[[PARAM_ALLOCA:.*]] = alloca i32
// LLVM-BOTH: %[[LOCAL_ALLOCA:.*]] = alloca i32
// LLVM: %[[GUARD_ADDR:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZGVZ13test_ctordtoriE10const_init)
// LLVM: %[[GUARD_LOAD:.*]] = load i8, ptr %[[GUARD_ADDR]]
// OGCG: %[[GUARD_LOAD:.*]] = load i8, ptr @_ZGVZ13test_ctordtoriE10const_init
// LLVM-BOTH: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD_LOAD]], 0
// LLVM-BOTH: br i1 %[[IS_UNINIT]]
//
// LLVM: %[[GET_TLS_INIT:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZZ13test_ctordtoriE10const_init)
// LLVM: call void @_ZN8CtorDtorC1Ei(ptr {{.*}}%[[GET_TLS_INIT]], i32 noundef 5)
// OGCG: call void @_ZN8CtorDtorC1Ei(ptr {{.*}}@_ZZ13test_ctordtoriE10const_init, i32 noundef 5)
// LLVM: %[[GET_TLS_DEL:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZZ13test_ctordtoriE10const_init)
// LLVM: call void @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr %[[GET_TLS_DEL]], ptr @__dso_handle)
// OGCG: call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @_ZZ13test_ctordtoriE10const_init, ptr @__dso_handle)
// LLVM: store i8 1, ptr %[[GUARD_ADDR]]
// OGCG: store i8 1, ptr @_ZGVZ13test_ctordtoriE10const_init
// LLVM-BOTH: br label
//
// LLVM: %[[GUARD_ADDR:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZGVZ13test_ctordtoriE4init)
// LLVM: %[[GUARD_LOAD:.*]] = load i8, ptr %[[GUARD_ADDR]]
// OGCG: %[[GUARD_LOAD:.*]] = load i8, ptr @_ZGVZ13test_ctordtoriE4init, align 1
// LLVM-BOTH: %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD_LOAD]], 0
// LLVM-BOTH: br i1 %[[IS_UNINIT]]
//
// LLVM: %[[GET_TLS_INIT:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZZ13test_ctordtoriE4init)
// LLVM-BOTH: %[[LOAD_PARAM:.*]] = load i32, ptr %[[PARAM_ALLOCA]]
// LLVM-BOTH: %[[LOAD_LOCAL:.*]] = load i32, ptr %[[LOCAL_ALLOCA]]
// LLVM-BOTH: %[[ADD1:.*]] = add nsw i32 %[[LOAD_PARAM]], %[[LOAD_LOCAL]]
// LLVM-BOTH: %[[CALL:.*]] = call noundef i32 @_Z5get_iv()
// LLVM-BOTH: %[[ADD2:.*]] = add nsw i32 %[[ADD1]], %[[CALL]]
// LLVM: call void @_ZN8CtorDtorC1Ei(ptr {{.*}}%[[GET_TLS_INIT]], i32 noundef %[[ADD2]])
// OGCG: call void @_ZN8CtorDtorC1Ei(ptr {{.*}}@_ZZ13test_ctordtoriE4init, i32 noundef %[[ADD2]])
//
// LLVM: %[[GET_TLS_DEL:.*]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZZ13test_ctordtoriE4init)
// LLVM: call void @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr %[[GET_TLS_DEL]], ptr @__dso_handle)
// OGCG: call i32 @__cxa_thread_atexit(ptr @_ZN8CtorDtorD1Ev, ptr @_ZZ13test_ctordtoriE4init, ptr @__dso_handle)
// LLVM: store i8 1, ptr %[[GUARD_ADDR]]
// OGCG: store i8 1, ptr @_ZGVZ13test_ctordtoriE4init
// LLVM-BOTH: br label
}

