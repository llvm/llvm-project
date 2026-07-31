// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR,CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR,CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

// 'S' doesn't end up in the 'after' IR because it loses its uses.
// CIR-BEFORE-DAG: !rec_S = !cir.struct<"S" {!s32i}>

// CIR-BEFORE-DAG: !rec_inner_aggregate = !cir.union<"inner_aggregate" {!cir.data_member<!s32i in !rec_S>, !s32i}>
// CIR-AFTER-DAG:  !rec_inner_aggregate = !cir.union<"inner_aggregate" {!s64i, !s32i}>
// LLVM-DAG: %union.inner_aggregate = type { i64 }

// CIR-BEFORE-DAG: !rec_inner_aggregate2 = !cir.union<"inner_aggregate2" {!s32i, !cir.data_member<!s32i in !rec_S>}>
// CIR-AFTER-DAG:  !rec_inner_aggregate2 = !cir.union<"inner_aggregate2" {!s32i, !s64i}>
// In LLVM, inner_aggregate2 was lowered to a literal, so the type went away.

// CIR-BEFORE-DAG: !rec_outer_aggregate = !cir.union<"outer_aggregate" {!cir.data_member<!s32i in !rec_S>, !s32i}>
// CIR-AFTER-DAG:  !rec_outer_aggregate = !cir.union<"outer_aggregate" {!s64i, !s32i}>
// LLVM-DAG: %union.outer_aggregate = type { i64 }

// CIR-BEFORE-DAG: !rec_outer_aggregate2 = !cir.union<"outer_aggregate2" {!s32i, !cir.data_member<!s32i in !rec_S>}>
// CIR-AFTER-DAG:  !rec_outer_aggregate2 = !cir.union<"outer_aggregate2" {!s32i, !s64i}>
// LLVM-DAG: %union.inner_aggregate2 = type { i64 }

// CIR-BEFORE-DAG: !rec_outer_aggregate3 = !cir.union<"outer_aggregate3" {!cir.data_member<!s32i in !rec_S>, !s32i}>
// CIR-AFTER-DAG:  !rec_outer_aggregate3 = !cir.union<"outer_aggregate3" {!s64i, !s32i}>
// LLVM-DAG: %union.outer_aggregate3 = type { i64 }

// CIR-BEFORE-DAG: !rec_HasPtrToMember = !cir.struct<"HasPtrToMember" {!cir.data_member<!s32i in !rec_S>}>
// CIR-AFTER-DAG: !rec_HasPtrToMember = !cir.struct<"HasPtrToMember" {!s64i}>
// LLVM-DAG: %struct.HasPtrToMember = type { i64 }

// CIR-BEFORE-DAG: !rec_U = !cir.union<"U" {!s32i, !rec_HasPtrToMember}>
// CIR-AFTER-DAG: !rec_U = !cir.union<"U" {!s32i, !rec_HasPtrToMember}>
// LLVM-DAG: %union.U = type { %struct.HasPtrToMember }

// This gets promoted to a constant, so it is up here.
// CIR-AFTER-DAG: cir.global "private" constant cir_private @__const._Z1fv.inner_a2 = #cir.const_record<{#cir.int<12> : !s32i}> : !rec_inner_aggregate2
// LLVM-DAG: @__const._Z1fv.inner_a2 = private {{.*}}constant { i32, [4 x i8] } { i32 12, [4 x i8] undef }

struct S { int x; };
int S::* p = nullptr;
// CIR-BEFORE-LABEL:   cir.global external @p = #cir.data_member<null> : !cir.data_member<!s32i in !rec_S>
// CIR-AFTER-LABEL: cir.global external @p = #cir.int<-1> : !s64i
// LLVM-DAG: @p = global i64 -1, align 8

// LLVMCIR gets this different because by the time we see how to do a 'zero'
// field, we've already lost the member-pointer type, because LowerToLLVM is
// doing the 'zeroing'.  We could be more clever here, but this is only in cases
// where it gets initialized anyway.
// LLVMCIR-DAG: @outer_a1 = global %union.outer_aggregate zeroinitializer
// OGCG-DAG:    @outer_a1 = global %union.outer_aggregate { i64 -1 }

// LLVM-DAG: @outer_a2 = global { i32, [4 x i8] } { i32 32, [4 x i8] undef }
// LLVM-DAG: @outer_a3 = global %union.outer_aggregate3 { i64 -1 }

struct HasPtrToMember { int S::*p; };  // not zero-initializable
union U {
  int a;              // first named member: zero-initializable
  HasPtrToMember b;   // later member: not zero-initializable
};
U u{};
// CIR-DAG: cir.global external @u = #cir.zero : !rec_U
// LLVMCIR-DAG: @u = global %union.U zeroinitializer
// OGCG-DAG: @u = global { i32, [4 x i8] } { i32 0, [4 x i8] undef }
auto use() {
  return u;
}

union outer_aggregate{int S::*m; int i; } outer_a1 = { p };
// CIR-BEFORE-LABEL:   cir.global external @outer_a1 = ctor : !rec_outer_aggregate {
// CIR-BEFORE:     %[[GET_GLOB:.*]] = cir.get_global @outer_a1 : !cir.ptr<!rec_outer_aggregate>
// CIR-BEFORE:     %[[GET_MEM:.*]] = cir.get_member %[[GET_GLOB]][0] {name = "m"} : !cir.ptr<!rec_outer_aggregate> -> !cir.ptr<!cir.data_member<!s32i in !rec_S>>
// CIR-BEFORE:     %[[GET_P:.*]] = cir.get_global @p : !cir.ptr<!cir.data_member<!s32i in !rec_S>>
// CIR-BEFORE:     %[[LOAD_P:.*]] = cir.load align(8) %[[GET_P]] : !cir.ptr<!cir.data_member<!s32i in !rec_S>>, !cir.data_member<!s32i in !rec_S>
// CIR-BEFORE:     cir.store {{.*}}%[[LOAD_P]], %[[GET_MEM]] : !cir.data_member<!s32i in !rec_S>, !cir.ptr<!cir.data_member<!s32i in !rec_S>>
// CIR-BEFORE:   }

// CIR-AFTER-LABEL: cir.global external @outer_a1 = #cir.zero : !rec_outer_aggregate
// CIR-AFTER-LABEL: cir.func internal private @__cxx_global_var_init() {
// CIR-AFTER:   %[[GET_GLOB:.*]] = cir.get_global @outer_a1 : !cir.ptr<!rec_outer_aggregate>
// CIR-AFTER:   %[[GET_MEM:.*]] = cir.get_member %[[GET_GLOB]][0] {name = "m"} : !cir.ptr<!rec_outer_aggregate> -> !cir.ptr<!s64i>
// CIR-AFTER:   %[[GET_P:.*]] = cir.get_global @p : !cir.ptr<!s64i>
// CIR-AFTER:   %[[LOAD_P:.*]] = cir.load {{.*}}%[[GET_P]] : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   cir.store {{.*}}%[[LOAD_P]], %[[GET_MEM]] : !s64i, !cir.ptr<!s64i>
// CIR-AFTER:   cir.return
// CIR-AFTER: }

// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   %[[LOAD_P:.*]] = load i64, ptr @p
// LLVM:   store i64 %[[LOAD_P]], ptr @outer_a1
// LLVM:   ret void
// LLVM: }

union outer_aggregate2{int i; int S::*m; } outer_a2 = { 32 };
// CIR-LABEL: cir.global external @outer_a2 = #cir.const_record<{#cir.int<32> : !s32i}> : !rec_outer_aggregate2
// LLVM version is above, because LLVM-IR orders vars before functions.

union outer_aggregate3{int S::*m; int i; } outer_a3;
// CIR-BEFORE-LABEL: cir.global external @outer_a3 = #cir.const_record<{#cir.data_member<null> : !cir.data_member<!s32i in !rec_S>}> : !rec_outer_aggregate3
// CIR-AFTER-LABEL:  cir.global external @outer_a3 = #cir.const_record<{#cir.int<-1> : !s64i}> : !rec_outer_aggregate3
// LLVM version is above, because LLVM-IR orders vars before functions.

void f() {
  union inner_aggregate{int S::*m; int i; } inner_a1 = { p };
  union inner_aggregate2{int i; int S::*m; } inner_a2 = { 12 };
}
// CIR-LABEL: cir.func {{.*}}@_Z1fv()
// CIR:          %[[A1_ALLOCA:.*]] = cir.alloca "inner_a1" {{.*}}init : !cir.ptr<!rec_inner_aggregate>
// CIR:          %[[A2_ALLOCA:.*]] = cir.alloca "inner_a2" {{.*}}init : !cir.ptr<!rec_inner_aggregate2>
// CIR-BEFORE:   %[[GET_A1_M:.*]] = cir.get_member %[[A1_ALLOCA]][0] {name = "m"} : !cir.ptr<!rec_inner_aggregate> -> !cir.ptr<!cir.data_member<!s32i in !rec_S>>
// CIR-AFTER:    %[[GET_A1_M:.*]] = cir.get_member %[[A1_ALLOCA]][0] {name = "m"} : !cir.ptr<!rec_inner_aggregate> -> !cir.ptr<!s64i>
// CIR-BEFORE:   %[[GET_P:.*]] = cir.get_global @p : !cir.ptr<!cir.data_member<!s32i in !rec_S>>
// CIR-AFTER:    %[[GET_P:.*]] = cir.get_global @p : !cir.ptr<!s64i>
// CIR-BEFORE:   %[[LOAD_P:.*]] = cir.load {{.*}}%[[GET_P]] : !cir.ptr<!cir.data_member<!s32i in !rec_S>>, !cir.data_member<!s32i in !rec_S>
// CIR-AFTER:    %[[LOAD_P:.*]] = cir.load {{.*}}%[[GET_P]] : !cir.ptr<!s64i>, !s64i
// CIR-BEFORE:   cir.store {{.*}}%[[LOAD_P]], %[[GET_A1_M]] : !cir.data_member<!s32i in !rec_S>, !cir.ptr<!cir.data_member<!s32i in !rec_S>>
// CIR-AFTER:    cir.store {{.*}}%[[LOAD_P]], %[[GET_A1_M]] : !s64i, !cir.ptr<!s64i>
// A2 is converted to a global constant, so there is slightly different behaviors here.
// CIR-BEFORE:   %[[TWELVE:.*]] = cir.const #cir.const_record<{#cir.int<12> : !s32i}> : !rec_inner_aggregate2
// CIR-BEFORE:   cir.store {{.*}}%[[TWELVE]], %[[A2_ALLOCA]] : !rec_inner_aggregate2, !cir.ptr<!rec_inner_aggregate2>
// CIR-AFTER:    %[[GET_A2_GLOB:.*]] = cir.get_global @__const._Z1fv.inner_a2 : !cir.ptr<!rec_inner_aggregate2>
// CIR-AFTER:    cir.copy %[[GET_A2_GLOB]] to %[[A2_ALLOCA]] : !cir.ptr<!rec_inner_aggregate2>
// CIR:          cir.return
// CIR: }

// LLVM-LABEL: define dso_local void @_Z1fv()
// LLVM:   %[[A1_ALLOCA:.*]] = alloca %union.inner_aggregate
// LLVM:   %[[A2_ALLOCA:.*]] = alloca %union.inner_aggregate2
// LLVM:   %[[LOAD_P:.*]] = load i64, ptr @p
// LLVM:   store i64 %[[LOAD_P]], ptr %[[A1_ALLOCA]]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[A2_ALLOCA]], ptr {{.*}}@__const._Z1fv.inner_a2, i64 8, i1 false)
// LLVM:   ret void
// LLVM: }
