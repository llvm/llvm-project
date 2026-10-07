// REQUIRES: aarch64-registered-target

// DEFINE: %{optimize} = opt -passes=mem2reg,tailcallelim -S

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve                        -fclangir -emit-cir -disable-O0-optnone -o - %s                       | FileCheck %s --check-prefixes=C,CIR %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-cir -disable-O0-optnone -o - %s                       | FileCheck %s --check-prefixes=C,CIR %}

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve                        -fclangir -emit-llvm -disable-O0-optnone -o - %s        | %{optimize} | FileCheck %s --check-prefixes=C,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o - %s        | %{optimize} | FileCheck %s --check-prefixes=C,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o - -x c++ %s | %{optimize} | FileCheck %s --check-prefixes=CPP,LLVM %}

// RUN:                   %clang_cc1_cg_arm64_sve                                  -emit-llvm -disable-O0-optnone -o - %s        | %{optimize} | FileCheck %s --check-prefixes=C,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o - %s        | %{optimize} | FileCheck %s --check-prefixes=C,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o - -x c++ %s | %{optimize} | FileCheck %s --check-prefixes=CPP,LLVM

// RUN:                   %clang_cc1_cg_arm64_sme                        -S -disable-O0-optnone -Werror -Wall -o /dev/null %s

//=============================================================================
// NOTES
//
// Tests for SVE MINV intrinsics
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

// C-LABEL: @test_svminv_s8(
// CPP-LABEL: @_Z14test_svminv_s8u10__SVBool_tu10__SVInt8_t(
int8_t test_svminv_s8(svbool_t pg, svint8_t op) MODE_ATTR
{
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.sminv" %{{.*}}, %{{.*}} :
// CIR-SAME:          -> !s8i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call i8 @llvm.aarch64.sve.sminv.nxv16i8(<vscale x 16 x i1> [[PG]], <vscale x 16 x i8> [[OP]])
// LLVM:    ret i8 [[TMP0]]
  return SVE_ACLE_FUNC(svminv,_s8,,)(pg, op);
}

// C-LABEL: @test_svminv_s16(
// CPP-LABEL: @_Z15test_svminv_s16u10__SVBool_tu11__SVInt16_t(
int16_t test_svminv_s16(svbool_t pg, svint16_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.sminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !s16i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x i16> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i16 @llvm.aarch64.sve.sminv.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP]])
// LLVM:    ret i16 [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_s16,,)(pg, op);
}

// C-LABEL: @test_svminv_s32(
// CPP-LABEL: @_Z15test_svminv_s32u10__SVBool_tu11__SVInt32_t(
int32_t test_svminv_s32(svbool_t pg, svint32_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.sminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !s32i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x i32> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i32 @llvm.aarch64.sve.sminv.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP]])
// LLVM:    ret i32 [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_s32,,)(pg, op);
}

// C-LABEL: @test_svminv_s64(
// CPP-LABEL: @_Z15test_svminv_s64u10__SVBool_tu11__SVInt64_t(
int64_t test_svminv_s64(svbool_t pg, svint64_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.sminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !s64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x i64> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i64 @llvm.aarch64.sve.sminv.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP]])
// LLVM:    ret i64 [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_s64,,)(pg, op);
}

// C-LABEL: @test_svminv_u8(
// CPP-LABEL: @_Z14test_svminv_u8u10__SVBool_tu11__SVUint8_t(
uint8_t test_svminv_u8(svbool_t pg, svuint8_t op) MODE_ATTR
{
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.uminv" %{{.*}}, %{{.*}} :
// CIR-SAME:          -> !u8i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call i8 @llvm.aarch64.sve.uminv.nxv16i8(<vscale x 16 x i1> [[PG]], <vscale x 16 x i8> [[OP]])
// LLVM:    ret i8 [[TMP0]]
  return SVE_ACLE_FUNC(svminv,_u8,,)(pg, op);
}

// C-LABEL: @test_svminv_u16(
// CPP-LABEL: @_Z15test_svminv_u16u10__SVBool_tu12__SVUint16_t(
uint16_t test_svminv_u16(svbool_t pg, svuint16_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.uminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !u16i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x i16> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i16 @llvm.aarch64.sve.uminv.nxv8i16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[OP]])
// LLVM:    ret i16 [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_u16,,)(pg, op);
}

// C-LABEL: @test_svminv_u32(
// CPP-LABEL: @_Z15test_svminv_u32u10__SVBool_tu12__SVUint32_t(
uint32_t test_svminv_u32(svbool_t pg, svuint32_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.uminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !u32i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x i32> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i32 @llvm.aarch64.sve.uminv.nxv4i32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[OP]])
// LLVM:    ret i32 [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_u32,,)(pg, op);
}

// C-LABEL: @test_svminv_u64(
// CPP-LABEL: @_Z15test_svminv_u64u10__SVBool_tu12__SVUint64_t(
uint64_t test_svminv_u64(svbool_t pg, svuint64_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.uminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !u64i

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x i64> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call i64 @llvm.aarch64.sve.uminv.nxv2i64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[OP]])
// LLVM:    ret i64 [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_u64,,)(pg, op);
}

// C-LABEL: @test_svminv_f16(
// CPP-LABEL: @_Z15test_svminv_f16u10__SVBool_tu13__SVFloat16_t(
float16_t test_svminv_f16(svbool_t pg, svfloat16_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.fminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.f16

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x half> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call half @llvm.aarch64.sve.fminv.nxv8f16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[OP]])
// LLVM:    ret half [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_f16,,)(pg, op);
}

// C-LABEL: @test_svminv_f32(
// CPP-LABEL: @_Z15test_svminv_f32u10__SVBool_tu13__SVFloat32_t(
float32_t test_svminv_f32(svbool_t pg, svfloat32_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.fminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.float

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x float> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call float @llvm.aarch64.sve.fminv.nxv4f32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[OP]])
// LLVM:    ret float [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_f32,,)(pg, op);
}

// C-LABEL: @test_svminv_f64(
// CPP-LABEL: @_Z15test_svminv_f64u10__SVBool_tu13__SVFloat64_t(
float64_t test_svminv_f64(svbool_t pg, svfloat64_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.fminv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.double

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x double> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = tail call double @llvm.aarch64.sve.fminv.nxv2f64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[OP]])
// LLVM:    ret double [[TMP1]]
  return SVE_ACLE_FUNC(svminv,_f64,,)(pg, op);
}
