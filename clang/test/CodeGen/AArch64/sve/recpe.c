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
// Tests for SVE RECPE intrinsics
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

// C-LABEL: @test_svrecpe_f16(
// CPP-LABEL: @_Z16test_svrecpe_f16u13__SVFloat16_t(
svfloat16_t test_svrecpe_f16(svfloat16_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.frecpe.x" %{{.*}} : (!cir.vector<[8] x !cir.f16>) -> !cir.vector<[8] x !cir.f16>

// LLVM-SAME: <vscale x 8 x half> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sve.frecpe.x.nxv8f16(<vscale x 8 x half> [[OP]])
// LLVM:    ret <vscale x 8 x half> [[TMP0]]
  return SVE_ACLE_FUNC(svrecpe,_f16,,)(op);
}

// C-LABEL: @test_svrecpe_f32(
// CPP-LABEL: @_Z16test_svrecpe_f32u13__SVFloat32_t(
svfloat32_t test_svrecpe_f32(svfloat32_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.frecpe.x" %{{.*}} : (!cir.vector<[4] x !cir.float>) -> !cir.vector<[4] x !cir.float>

// LLVM-SAME: <vscale x 4 x float> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sve.frecpe.x.nxv4f32(<vscale x 4 x float> [[OP]])
// LLVM:    ret <vscale x 4 x float> [[TMP0]]
  return SVE_ACLE_FUNC(svrecpe,_f32,,)(op);
}

// C-LABEL: @test_svrecpe_f64(
// CPP-LABEL: @_Z16test_svrecpe_f64u13__SVFloat64_t(
svfloat64_t test_svrecpe_f64(svfloat64_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.frecpe.x" %{{.*}} : (!cir.vector<[2] x !cir.double>) -> !cir.vector<[2] x !cir.double>

// LLVM-SAME: <vscale x 2 x double> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sve.frecpe.x.nxv2f64(<vscale x 2 x double> [[OP]])
// LLVM:    ret <vscale x 2 x double> [[TMP0]]
  return SVE_ACLE_FUNC(svrecpe,_f64,,)(op);
}
