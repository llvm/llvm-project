// REQUIRES: aarch64-registered-target

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve                        -fclangir -emit-cir -disable-O0-optnone -o - %s  | FileCheck %s --check-prefixes=C,CIR %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-cir -disable-O0-optnone -o - %s  | FileCheck %s --check-prefixes=C,CIR %}

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve                        -fclangir -emit-llvm -disable-O0-optnone -o - %s         | FileCheck %s --check-prefixes=C,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o - %s         | FileCheck %s --check-prefixes=C,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o - -x c++ %s  | FileCheck %s --check-prefixes=CPP,LLVM %}

// RUN:                   %clang_cc1_cg_arm64_sve                                  -emit-llvm -disable-O0-optnone -o - %s         | FileCheck %s --check-prefixes=C,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o - %s         | FileCheck %s --check-prefixes=C,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o - -x c++ %s  | FileCheck %s --check-prefixes=CPP,LLVM

//=============================================================================
// NOTES
//
// Tests for SVE LEN intrinsics
//=============================================================================

#include <arm_sve.h>

#if defined __ARM_FEATURE_SME
#define MODE_ATTR __arm_streaming
#else
#define MODE_ATTR
#endif

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

// C-LABEL: @test_svlen_u8
// CPP-LABEL: @_Z13test_svlen_u8u11__SVUint8_t(
uint64_t test_svlen_u8(svuint8_t op) MODE_ATTR
{
// CIR:     %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:     %[[C16:.*]] = cir.const #cir.int<16> : !u64i
// CIR:     %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C16]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 16
  return SVE_ACLE_FUNC(svlen,_u8,,)(op);
}

// C-LABEL: @test_svlen_s8(
// CPP-LABEL: @_Z13test_svlen_s8u10__SVInt8_t(
uint64_t test_svlen_s8(svint8_t op) MODE_ATTR
{
// CIR:     %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:     %[[C16:.*]] = cir.const #cir.int<16> : !u64i
// CIR:     %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C16]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 16
  return SVE_ACLE_FUNC(svlen,_s8,,)(op);
}

// C-LABEL: @test_svlen_u16(
// CPP-LABEL: @_Z14test_svlen_u16u12__SVUint16_t(
uint64_t test_svlen_u16(svuint16_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C8:.*]] = cir.const #cir.int<8> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C8]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 8
  return SVE_ACLE_FUNC(svlen,_u16,,)(op);
}

// C-LABEL: @test_svlen_s16(
// CPP-LABEL: @_Z14test_svlen_s16u11__SVInt16_t(
uint64_t test_svlen_s16(svint16_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C8:.*]] = cir.const #cir.int<8> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C8]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 8
  return SVE_ACLE_FUNC(svlen,_s16,,)(op);
}

// C-LABEL: @test_svlen_f16(
// CPP-LABEL: @_Z14test_svlen_f16u13__SVFloat16_t(
uint64_t test_svlen_f16(svfloat16_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C8:.*]] = cir.const #cir.int<8> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C8]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 8
  return SVE_ACLE_FUNC(svlen,_f16,,)(op);
}

// C-LABEL: @test_svlen_bf16(
// CPP-LABEL: @_Z15test_svlen_bf16u14__SVBfloat16_t(
uint64_t test_svlen_bf16(svbfloat16_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C8:.*]] = cir.const #cir.int<8> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C8]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 8
  return SVE_ACLE_FUNC(svlen,_bf16,,)(op);
}

// C-LABEL: @test_svlen_u32(
// CPP-LABEL: @_Z14test_svlen_u32u12__SVUint32_t(
uint64_t test_svlen_u32(svuint32_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C4:.*]] = cir.const #cir.int<4> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C4]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64  [[VSCALE]], 4
  return SVE_ACLE_FUNC(svlen,_u32,,)(op);
}

// C-LABEL: @test_svlen_s32(
// CPP-LABEL: @_Z14test_svlen_s32u11__SVInt32_t(
uint64_t test_svlen_s32(svint32_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C4:.*]] = cir.const #cir.int<4> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C4]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 4
  return SVE_ACLE_FUNC(svlen,_s32,,)(op);
}

// C-LABEL: @test_svlen_f32(
// CPP-LABEL: @_Z14test_svlen_f32u13__SVFloat32_t(
uint64_t test_svlen_f32(svfloat32_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C4:.*]] = cir.const #cir.int<4> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C4]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 4
  return SVE_ACLE_FUNC(svlen,_f32,,)(op);
}

// C-LABEL: @test_svlen_u64(
// CPP-LABEL: @_Z14test_svlen_u64u12__SVUint64_t(
uint64_t test_svlen_u64(svuint64_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C2:.*]] = cir.const #cir.int<2> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C2]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64  [[VSCALE]], 2
  return SVE_ACLE_FUNC(svlen,_u64,,)(op);
}

// C-LABEL: @test_svlen_s64
// CPP-LABEL: @_Z14test_svlen_s64u11__SVInt64_t(
uint64_t test_svlen_s64(svint64_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C2:.*]] = cir.const #cir.int<2> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C2]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 2
  return SVE_ACLE_FUNC(svlen,_s64,,)(op);
}

// C-LABEL: @test_svlen_f64
// CPP-LABEL: @_Z14test_svlen_f64u13__SVFloat64_t(
uint64_t test_svlen_f64(svfloat64_t op) MODE_ATTR
{
// CIR:           %[[VSCALE:.*]] = cir.call_llvm_intrinsic "vscale"  : () -> !u64i
// CIR:           %[[C2:.*]] = cir.const #cir.int<2> : !u64i
// CIR:           %[[BINOP:.*]] = cir.mul nuw %[[VSCALE]], %[[C2]] : !u64i

// LLVM:    [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
// LLVM:    [[RES:%.*]] = mul nuw i64 [[VSCALE]], 2
  return SVE_ACLE_FUNC(svlen,_f64,,)(op);
}
