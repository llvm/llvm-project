// REQUIRES: aarch64-registered-target

// DEFINE: %{optimize} = opt -passes=mem2reg -S

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
// Tests for SVE MAXNMV intrinsics
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

// C-LABEL: @test_svmaxnmv_f16(
// CPP-LABEL: @_Z17test_svmaxnmv_f16u10__SVBool_tu13__SVFloat16_t(
float16_t test_svmaxnmv_f16(svbool_t pg, svfloat16_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.fmaxnmv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.f16

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x half> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call half @llvm.aarch64.sve.fmaxnmv.nxv8f16(<vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[OP]])
// LLVM:    ret half [[TMP1]]
  return SVE_ACLE_FUNC(svmaxnmv,_f16,,)(pg, op);
}

// C-LABEL: @test_svmaxnmv_f32(
// CPP-LABEL: @_Z17test_svmaxnmv_f32u10__SVBool_tu13__SVFloat32_t(
float32_t test_svmaxnmv_f32(svbool_t pg, svfloat32_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.fmaxnmv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.float

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x float> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call float @llvm.aarch64.sve.fmaxnmv.nxv4f32(<vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[OP]])
// LLVM:    ret float [[TMP1]]
  return SVE_ACLE_FUNC(svmaxnmv,_f32,,)(pg, op);
}

// C-LABEL: @test_svmaxnmv_f64(
// CPP-LABEL: @_Z17test_svmaxnmv_f64u10__SVBool_tu13__SVFloat64_t(
float64_t test_svmaxnmv_f64(svbool_t pg, svfloat64_t op) MODE_ATTR
{
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[RES:.*]] = cir.call_llvm_intrinsic "aarch64.sve.fmaxnmv" %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.double

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x double> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call double @llvm.aarch64.sve.fmaxnmv.nxv2f64(<vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[OP]])
// LLVM:    ret double [[TMP1]]
  return SVE_ACLE_FUNC(svmaxnmv,_f64,,)(pg, op);
}
