// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fclangir -emit-cir -fcxx-exceptions -fexceptions -mmlir --mlir-print-ir-before=cir-cxxabi-lowering -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR,CIR-BEFORE
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR,CIR-AFTER
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fclangir -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefixes=LLVM

struct Other {
    int x;
    void func(int, float);
};

struct WithMemPtr {
    int x;
    double y;
    void (Other::*mpt)(int, float);
};

struct Trivial {
    int x;
    double y;
    decltype(&Other::x) ptr;
};

// CIR-BEFORE-DAG: !rec_Other = !cir.record<struct "Other" {!s32i}>
// CIR-BEFORE-DAG: !rec_Trivial = !cir.record<struct "Trivial" {!s32i, !cir.double, !cir.data_member<!s32i in !rec_Other>}>
// CIR-BEFORE-DAG: !rec_WithMemPtr = !cir.record<struct "WithMemPtr" {!s32i, !cir.double, !cir.method<!cir.func<(!cir.ptr<!rec_Other>, !s32i, !cir.float)> in !rec_Other>}>
// CIR-AFTER-DAG: !rec_Other = !cir.record<struct "Other" {!s32i}>
// CIR-AFTER-DAG: !rec_Trivial = !cir.record<struct "Trivial" {!s32i, !cir.double, !s64i}>
// CIR-AFTER-DAG: !rec_anon_struct = !cir.record<struct  {!s64i, !s64i}>
// CIR-AFTER-DAG: !rec_WithMemPtr = !cir.record<struct "WithMemPtr" {!s32i, !cir.double, !rec_anon_struct}>

// LLVM-DAG: %struct.WithMemPtr = type { i32, double, { i64, i64 } }
// LLVM-DAG: %struct.Trivial = type { i32, double, i64 }

// CIR-AFTER-DAG: cir.global "private" constant cir_private @__const.local.localT_init = #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<2.200000e+00> : !cir.double, #cir.int<0> : !s64i}> : !rec_Trivial
// CIR-AFTER-DAG: cir.global "private" constant cir_private @__const.local.localMpt_init = #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<2.000000e+00> : !cir.double, #cir.const_record<{#cir.global_view<@_ZN5Other4funcEif> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct}> : !rec_WithMemPtr

// LLVM-DAG: @__const.local.localMpt_init = private {{.*}}constant %struct.WithMemPtr { i32 1, double 2.000000e+00, { i64, i64 } { i64 ptrtoint (ptr @_ZN5Other4funcEif to i64), i64 0 } }
// LLVM-DAG: @__const.local.localT_init = private {{.*}}constant %struct.Trivial { i32 1, double 2.200000e+00, i64 0 }

// This CAN be zero-initialized.
WithMemPtr mpt;
// CIR-DAG: cir.global external @mpt = #cir.zero : !rec_WithMemPtr {alignment = 8 : i64}
// LLVM-DAG: @mpt = global %struct.WithMemPtr zeroinitializer, align 8

WithMemPtr mpt_init{1, 2.0, &Other::func};
// CIR-BEFORE-DAG: cir.global external @mpt_init = #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<2.000000e+00> : !cir.double, #cir.method<@_ZN5Other4funcEif> : !cir.method<!cir.func<(!cir.ptr<!rec_Other>, !s32i, !cir.float)> in !rec_Other>}> : !rec_WithMemPtr {alignment = 8 : i64}
// CIR-AFTER-DAG: cir.global external @mpt_init = #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<2.000000e+00> : !cir.double, #cir.const_record<{#cir.global_view<@_ZN5Other4funcEif> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct}> : !rec_WithMemPtr {alignment = 8 : i64}
// LLVM-DAG: @mpt_init = global %struct.WithMemPtr { i32 1, double 2.000000e+00, { i64, i64 } { i64 ptrtoint (ptr @_ZN5Other4funcEif to i64), i64 0 } }, align 8

// This case has a trivial default constructor, but can't be zero-initialized.
Trivial t;
// CIR-BEFORE-DAG: cir.global external @t = #cir.const_record<{#cir.int<0> : !s32i, #cir.fp<0.000000e+00> : !cir.double, #cir.data_member<null> : !cir.data_member<!s32i in !rec_Other>}> : !rec_Trivial {alignment = 8 : i64}
// CIR-AFTER-DAG: cir.global external @t = #cir.const_record<{#cir.int<0> : !s32i, #cir.fp<0.000000e+00> : !cir.double, #cir.int<-1> : !s64i}> : !rec_Trivial {alignment = 8 : i64} loc(#loc25)
// LLVM-DAG: @t = global %struct.Trivial { i32 0, double 0.000000e+00, i64 -1 }, align 8

Trivial t_init{1,2.2, &Other::x};
// CIR-BEFORE-DAG: cir.global external @t_init = #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<2.200000e+00> : !cir.double, #cir.data_member<0> : !cir.data_member<!s32i in !rec_Other>}> : !rec_Trivial {alignment = 8 : i64}
// CIR-AFTER-DAG: cir.global external @t_init = #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<2.200000e+00> : !cir.double, #cir.int<0> : !s64i}> : !rec_Trivial {alignment = 8 : i64}
// LLVM-DAG: @t_init = global %struct.Trivial { i32 1, double 2.200000e+00, i64 0 }, align 8

extern "C" void local() {
  // CIR-LABEL: @local(
  // LLVM-LABEL: @local(
  // CIR: cir.alloca !rec_WithMemPtr, !cir.ptr<!rec_WithMemPtr>, ["localMpt"] {alignment = 8 : i64}
  // CIR: cir.alloca !rec_Trivial, !cir.ptr<!rec_Trivial>, ["localT"] {alignment = 8 : i64}
  // CIR: %[[MPT_INIT:.*]] = cir.alloca !rec_WithMemPtr, !cir.ptr<!rec_WithMemPtr>, ["localMpt_init", init] {alignment = 8 : i64}
  // CIR: %[[T_INIT:.*]] = cir.alloca !rec_Trivial, !cir.ptr<!rec_Trivial>, ["localT_init", init] {alignment = 8 : i64}

  // LLVM: alloca %struct.WithMemPtr
  // LLVM: alloca %struct.Trivial
  // LLVM: %[[MPT_INIT:.*]] = alloca %struct.WithMemPtr
  // LLVM: %[[T_INIT:.*]] = alloca %struct.Trivial
  WithMemPtr localMpt;

  Trivial localT;

  WithMemPtr localMpt_init{1, 2.0, &Other::func};
  // CIR-BEFORE: %[[MPT_INIT_VAL:.*]] = cir.const #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<2.000000e+00> : !cir.double, #cir.method<@_ZN5Other4funcEif> : !cir.method<!cir.func<(!cir.ptr<!rec_Other>, !s32i, !cir.float)> in !rec_Other>}> : !rec_WithMemPtr
  // CIR-BEFORE: cir.store align(8) %[[MPT_INIT_VAL]], %[[MPT_INIT]] : !rec_WithMemPtr, !cir.ptr<!rec_WithMemPtr>
  // CIR-AFTER: %[[MPT_INIT_VAL:.*]] = cir.get_global @__const.local.localMpt_init : !cir.ptr<!rec_WithMemPtr>
  // CIR-AFTER: cir.copy %[[MPT_INIT_VAL]] to %[[MPT_INIT]] : !cir.ptr<!rec_WithMemPtr>

  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[MPT_INIT]], ptr {{.*}}@__const.local.localMpt_init, i64 32, i1 false)

  Trivial localT_init{1,2.2, &Other::x};
  // CIR-BEFORE: %[[T_INIT_VAL:.*]] = cir.const #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<2.200000e+00> : !cir.double, #cir.data_member<0> : !cir.data_member<!s32i in !rec_Other>}> : !rec_Trivial
  // CIR-BEFORE: cir.store align(8) %[[T_INIT_VAL]], %[[T_INIT]] : !rec_Trivial, !cir.ptr<!rec_Trivial>

  // CIR-AFTER: %[[T_INIT_VAL:.*]] = cir.get_global @__const.local.localT_init : !cir.ptr<!rec_Trivial>
  // CIR-AFTER: cir.copy %[[T_INIT_VAL]] to %[[T_INIT]] : !cir.ptr<!rec_Trivial>

  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[T_INIT]], ptr {{.*}}@__const.local.localT_init, i64 24, i1 false)
}
