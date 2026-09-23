// REQUIRES: aarch64-registered-target

// DEFINE: %{optimize} = opt -passes=mem2reg,instcombine,tailcallelim -S

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve                        -fclangir -emit-cir -disable-O0-optnone -o - %s                | FileCheck %s --check-prefixes=ALL,CIR %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-cir -disable-O0-optnone -o - %s                | FileCheck %s --check-prefixes=ALL,CIR %}

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve                        -fclangir -emit-llvm -disable-O0-optnone -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM %}

// RUN:                   %clang_cc1_cg_arm64_sve                                  -emit-llvm -disable-O0-optnone -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM

//=============================================================================
// NOTES
//
// Tests for SVE ADDV intrinsics
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

// ALL-LABEL: @test_svaddv_s8
int64_t test_svaddv_s8(svbool_t pg, svint8_t op) MODE_ATTR
{
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.saddv" %{{.*}}, %{{.*}} :
// CIR-SAME:          -> !s64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call i64 @llvm.aarch64.sve.saddv.nxv16i8(<vscale x 16 x i1> [[PG]], <vscale x 16 x i8> [[OP]])
// LLVM:    ret i64 [[TMP0]]
  return SVE_ACLE_FUNC(svaddv,_s8,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_s16
int64_t test_svaddv_s16(svbool_t pg, svint16_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.saddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !s64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x i16> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i64 @llvm.aarch64.sve.saddv.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP]])
// LLVM:    ret i64 [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_s16,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_s32
int64_t test_svaddv_s32(svbool_t pg, svint32_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.saddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !s64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x i32> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i64 @llvm.aarch64.sve.saddv.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP]])
// LLVM:    ret i64 [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_s32,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_s64
int64_t test_svaddv_s64(svbool_t pg, svint64_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.saddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !s64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x i64> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i64 @llvm.aarch64.sve.saddv.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP]])
// LLVM:    ret i64 [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_s64,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_u8
uint64_t test_svaddv_u8(svbool_t pg, svuint8_t op) MODE_ATTR
{
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.uaddv" %{{.*}}, %{{.*}} :
// CIR-SAME:          -> !u64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call i64 @llvm.aarch64.sve.uaddv.nxv16i8(<vscale x 16 x i1> [[PG]], <vscale x 16 x i8> [[OP]])
// LLVM:    ret i64 [[TMP0]]
  return SVE_ACLE_FUNC(svaddv,_u8,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_u16
uint64_t test_svaddv_u16(svbool_t pg, svuint16_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.uaddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !u64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x i16> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i64 @llvm.aarch64.sve.uaddv.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP]])
// LLVM:    ret i64 [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_u16,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_u32
uint64_t test_svaddv_u32(svbool_t pg, svuint32_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.uaddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !u64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x i32> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i64 @llvm.aarch64.sve.uaddv.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP]])
// LLVM:    ret i64 [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_u32,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_u64
uint64_t test_svaddv_u64(svbool_t pg, svuint64_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.uaddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !u64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x i64> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i64 @llvm.aarch64.sve.uaddv.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP]])
// LLVM:    ret i64 [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_u64,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_f16
float16_t test_svaddv_f16(svbool_t pg, svfloat16_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.faddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.f16

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x half> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call half @llvm.aarch64.sve.faddv.nxv8f16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[OP]])
// LLVM:    ret half [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_f16,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_f32
float32_t test_svaddv_f32(svbool_t pg, svfloat32_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.faddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.float

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x float> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call float @llvm.aarch64.sve.faddv.nxv4f32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[OP]])
// LLVM:    ret float [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_f32,,)(pg, op);
}

// ALL-LABEL: @test_svaddv_f64
float64_t test_svaddv_f64(svbool_t pg, svfloat64_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.faddv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.double

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x double> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call double @llvm.aarch64.sve.faddv.nxv2f64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[OP]])
// LLVM:    ret double [[TMP1]]
  return SVE_ACLE_FUNC(svaddv,_f64,,)(pg, op);
}
