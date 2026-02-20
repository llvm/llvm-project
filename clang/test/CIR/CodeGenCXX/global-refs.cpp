// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM

struct DefCtor{};
struct WithCtor{
  WithCtor();
  WithCtor(int);
};

struct WithCtorDtor{
  WithCtorDtor();
  WithCtorDtor(int);
  ~WithCtorDtor();
};

int globalInt;
// CIR: cir.global external @globalInt = #cir.int<0> : !s32i {alignment = 4 : i64}
// LLVM: @globalInt = global i32 0, align 4

int &globalIntRef = globalInt;
// CIR: cir.global constant external @globalIntRef = #cir.global_view<@globalInt> : !cir.ptr<!s32i> {alignment = 8 : i64}
// LLVM: @globalIntRef = constant ptr @globalInt, align 8

const int &constGlobalIntRef = 5;
// CIR: cir.global "private" external @_ZGR17constGlobalIntRef_ = #cir.int<5> : !s32i {alignment = 4 : i64}
// CIR: cir.global constant external @constGlobalIntRef = #cir.global_view<@_ZGR17constGlobalIntRef_> : !cir.ptr<!s32i> {alignment = 8 : i64}
// LLVM: @_ZGR17constGlobalIntRef_ = {{.*}}global i32 5, align 4
// LLVM: @constGlobalIntRef = constant ptr @_ZGR17constGlobalIntRef_, align 8

DefCtor defCtor{};
// CIR: cir.global external @defCtor = #cir.undef : !rec_DefCtor {alignment = 1 : i64}
// LLVM: @defCtor = global %struct.DefCtor undef, align 1

DefCtor &defCtorRef = defCtor;
// CIR: cir.global constant external @defCtorRef = #cir.global_view<@defCtor> : !cir.ptr<!rec_DefCtor> {alignment = 8 : i64}
// LLVM: @defCtorRef = constant ptr @defCtor, align 8

const DefCtor &constDefCtorRef{};
// CIR: cir.global "private" constant external @_ZGR15constDefCtorRef_ = #cir.undef : !rec_DefCtor {alignment = 1 : i64}
// CIR: cir.global constant external @constDefCtorRef = #cir.global_view<@_ZGR15constDefCtorRef_> : !cir.ptr<!rec_DefCtor> {alignment = 8 : i64}
// LLVM: @_ZGR15constDefCtorRef_ = {{.*}}constant %struct.DefCtor undef, align 1
// LLVM: @constDefCtorRef = constant ptr @_ZGR15constDefCtorRef_, align 8

WithCtor withCtor{};
// CIR: cir.global external @withCtor = #cir.zero : !rec_WithCtor {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-NEXT: cir.func internal private @__cxx_global_var_init{{.*}}() {
// CIR-NEXT:   %[[GET_GLOB:.*]] = cir.get_global @withCtor : !cir.ptr<!rec_WithCtor>
// CIR-NEXT:   cir.call @_ZN8WithCtorC1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_WithCtor>{{.*}}) -> ()
// CIR-NEXT:   cir.return
// CIR-NEXT: }
// LLVM: @withCtor = global %struct.WithCtor zeroinitializer, align 1

WithCtor &withCtorRef = withCtor;
// CIR: cir.global constant external @withCtorRef = #cir.global_view<@withCtor> : !cir.ptr<!rec_WithCtor> {alignment = 8 : i64}
// LLVM: @withCtorRef = constant ptr @withCtor, align 8

const WithCtor &constWithCtorRef{};
// CIR: cir.global external @constWithCtorRef = #cir.ptr<null> : !cir.ptr<!rec_WithCtor> {alignment = 8 : i64, ast = #cir.var.decl.ast}
// CIR-NEXT: cir.func internal private @__cxx_global_var_init{{.*}}() {
// CIR-NEXT:   %[[GET_GLOB:.*]] = cir.get_global @constWithCtorRef : !cir.ptr<!cir.ptr<!rec_WithCtor>>
// CIR-NEXT:   %[[GET_GLOB_OBJ:.*]] = cir.get_global @_ZGR16constWithCtorRef_ : !cir.ptr<!rec_WithCtor>
// CIR-NEXT:   cir.call @_ZN8WithCtorC1Ev(%[[GET_GLOB_OBJ]]) : (!cir.ptr<!rec_WithCtor>{{.*}}) -> ()
// CIR-NEXT:   cir.store align(8) %[[GET_GLOB_OBJ]], %[[GET_GLOB]] : !cir.ptr<!rec_WithCtor>, !cir.ptr<!cir.ptr<!rec_WithCtor>>
// CIR-NEXT:   cir.return
// CIR-NEXT: }
// LLVM: @constWithCtorRef = global ptr null, align 8

const WithCtor &constWithCtorRef2{5};
// CIR: cir.global external @constWithCtorRef2 = #cir.ptr<null> : !cir.ptr<!rec_WithCtor> {alignment = 8 : i64, ast = #cir.var.decl.ast}
// CIR-NEXT: cir.func internal private @__cxx_global_var_init{{.*}}() {
// CIR-NEXT:   %[[GET_GLOB:.*]] = cir.get_global @constWithCtorRef2 : !cir.ptr<!cir.ptr<!rec_WithCtor>>
// CIR-NEXT:   %[[GET_GLOB_OBJ:.*]] = cir.get_global @_ZGR17constWithCtorRef2_ : !cir.ptr<!rec_WithCtor>
// CIR-NEXT:   %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR-NEXT:   cir.call @_ZN8WithCtorC1Ei(%[[GET_GLOB_OBJ]], %[[FIVE]]) : (!cir.ptr<!rec_WithCtor>{{.*}}, !s32i{{.*}}) -> ()
// CIR-NEXT:   cir.store align(8) %[[GET_GLOB_OBJ]], %[[GET_GLOB]] : !cir.ptr<!rec_WithCtor>, !cir.ptr<!cir.ptr<!rec_WithCtor>>
// CIR-NEXT:   cir.return
// CIR-NEXT: }
// LLVM: @constWithCtorRef2 = global ptr null, align 8

WithCtorDtor withCtorDtor{};
// CIR: cir.global external @withCtorDtor = #cir.zero : !rec_WithCtorDtor {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR: cir.func internal private @__cxx_global_var_init{{.*}}() {
// CIR-NEXT:   %[[GET_GLOB:.*]] = cir.get_global @withCtorDtor : !cir.ptr<!rec_WithCtorDtor>
// CIR-NEXT:   cir.call @_ZN12WithCtorDtorC1Ev(%[[GET_GLOB]]) : (!cir.ptr<!rec_WithCtorDtor>{{.*}}) -> ()
// CIR-NEXT:   %[[GET_GLOB:.*]] = cir.get_global @withCtorDtor : !cir.ptr<!rec_WithCtorDtor> 
// CIR-NEXT:   %[[GET_DTOR:.*]] = cir.get_global @_ZN12WithCtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_WithCtorDtor>)>>
// CIR-NEXT:   %[[VOID_FN_PTR:.*]] = cir.cast bitcast %[[GET_DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_WithCtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR-NEXT:   %[[GLOB_TO_VOID:.*]] = cir.cast bitcast %[[GET_GLOB]] : !cir.ptr<!rec_WithCtorDtor> -> !cir.ptr<!void>
// CIR-NEXT:   %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR-NEXT:   cir.call @__cxa_atexit(%[[VOID_FN_PTR]], %[[GLOB_TO_VOID]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>{{.*}}) -> ()
// CIR-NEXT:   cir.return
// CIR-NEXT: }
// LLVM: @withCtorDtor = global %struct.WithCtorDtor zeroinitializer, align 1

WithCtorDtor &withCtorDtorRef = withCtorDtor;
// CIR: cir.global constant external @withCtorDtorRef = #cir.global_view<@withCtorDtor> : !cir.ptr<!rec_WithCtorDtor> {alignment = 8 : i64}
// LLVM: @withCtorDtorRef = constant ptr @withCtorDtor, align 8

// LLVM: define internal void @__cxx_global_var_init{{.*}}()
// LLVM:   call void @_ZN8WithCtorC1Ev(ptr {{.*}}@withCtor)
// LLVM-NEXT:   ret void

// LLVM: define internal void @__cxx_global_var_init{{.*}}()
// LLVM:   call void @_ZN8WithCtorC1Ev(ptr {{.*}}@_ZGR16constWithCtorRef_)
// LLVM-NEXT:   store ptr @_ZGR16constWithCtorRef_, ptr @constWithCtorRef, align 8
// LLVM-NEXT:   ret void

// LLVM: define internal void @__cxx_global_var_init{{.*}}()
// LLVM:   call void @_ZN8WithCtorC1Ei(ptr {{.*}}@_ZGR17constWithCtorRef2_, i32 {{.*}}5)
// LLVM-NEXT:   store ptr @_ZGR17constWithCtorRef2_, ptr @constWithCtorRef2, align 8
// LLVM-NEXT:   ret void

// LLVM: define internal void @__cxx_global_var_init{{.*}}()
// LLVM:   call void @_ZN12WithCtorDtorC1Ev(ptr {{.*}}@withCtorDtor)
// LLVM-NEXT:   call {{.*}}@__cxa_atexit(ptr {{.*}}@_ZN12WithCtorDtorD1Ev, ptr {{.*}}@withCtorDtor, ptr {{.*}}@__dso_handle)
// LLVM-NEXT:   ret void

// TODO(cir): Once we get destructors for temporaries done, we should test them
// here, same as the 'const-WithCtor' examples, except with the 'withCtorDtor'
// versions.
