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
// Tests for plain SVE RINTX intrinsics
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

// C-LABEL: @test_svrintx_f16_z(
// CPP-LABEL: @_Z18test_svrintx_f16_zu10__SVBool_tu13__SVFloat16_t(
svfloat16_t test_svrintx_f16_z(svbool_t pg, svfloat16_t op) MODE_ATTR
{
// CIR:       %[[ZERO:.*]] = cir.const #cir.zero : !cir.vector<[8] x !cir.f16>
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %[[ZERO]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[8] x !cir.f16>, !cir.vector<[8] x !cir.int<u, 1>>, !cir.vector<[8] x !cir.f16>) -> !cir.vector<[8] x !cir.f16>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x half> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.frintx.nxv8f16(<vscale x 8 x half> zeroinitializer, <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[OP]])
// LLVM:    ret <vscale x 8 x half> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f16,_z,)(pg, op);
}

// C-LABEL: @test_svrintx_f32_z(
// CPP-LABEL: @_Z18test_svrintx_f32_zu10__SVBool_tu13__SVFloat32_t(
svfloat32_t test_svrintx_f32_z(svbool_t pg, svfloat32_t op) MODE_ATTR
{
// CIR:       %[[ZERO:.*]] = cir.const #cir.zero : !cir.vector<[4] x !cir.float>
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %[[ZERO]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[4] x !cir.float>, !cir.vector<[4] x !cir.int<u, 1>>, !cir.vector<[4] x !cir.float>) -> !cir.vector<[4] x !cir.float>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x float> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frintx.nxv4f32(<vscale x 4 x float> zeroinitializer, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[OP]])
// LLVM:    ret <vscale x 4 x float> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f32,_z,)(pg, op);
}

// C-LABEL: @test_svrintx_f64_z(
// CPP-LABEL: @_Z18test_svrintx_f64_zu10__SVBool_tu13__SVFloat64_t(
svfloat64_t test_svrintx_f64_z(svbool_t pg, svfloat64_t op) MODE_ATTR
{
// CIR:       %[[ZERO:.*]] = cir.const #cir.zero : !cir.vector<[2] x !cir.double>
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %[[ZERO]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[2] x !cir.double>, !cir.vector<[2] x !cir.int<u, 1>>, !cir.vector<[2] x !cir.double>) -> !cir.vector<[2] x !cir.double>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x double> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.frintx.nxv2f64(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[OP]])
// LLVM:    ret <vscale x 2 x double> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f64,_z,)(pg, op);
}

// C-LABEL: @test_svrintx_f16_m(
// CPP-LABEL: @_Z18test_svrintx_f16_mu13__SVFloat16_tu10__SVBool_tS_(
svfloat16_t test_svrintx_f16_m(svfloat16_t inactive, svbool_t pg, svfloat16_t op) MODE_ATTR
{
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %{{.*}}, %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[8] x !cir.f16>, !cir.vector<[8] x !cir.int<u, 1>>, !cir.vector<[8] x !cir.f16>) -> !cir.vector<[8] x !cir.f16>

// LLVM-SAME: <vscale x 8 x half> [[INACTIVE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x half> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.frintx.nxv8f16(<vscale x 8 x half> [[INACTIVE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[OP]])
// LLVM:    ret <vscale x 8 x half> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f16,_m,)(inactive, pg, op);
}

// C-LABEL: @test_svrintx_f32_m(
// CPP-LABEL: @_Z18test_svrintx_f32_mu13__SVFloat32_tu10__SVBool_tS_(
svfloat32_t test_svrintx_f32_m(svfloat32_t inactive, svbool_t pg, svfloat32_t op) MODE_ATTR
{
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %{{.*}}, %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[4] x !cir.float>, !cir.vector<[4] x !cir.int<u, 1>>, !cir.vector<[4] x !cir.float>) -> !cir.vector<[4] x !cir.float>

// LLVM-SAME: <vscale x 4 x float> [[INACTIVE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x float> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frintx.nxv4f32(<vscale x 4 x float> [[INACTIVE]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[OP]])
// LLVM:    ret <vscale x 4 x float> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f32,_m,)(inactive, pg, op);
}

// C-LABEL: @test_svrintx_f64_m(
// CPP-LABEL: @_Z18test_svrintx_f64_mu13__SVFloat64_tu10__SVBool_tS_(
svfloat64_t test_svrintx_f64_m(svfloat64_t inactive, svbool_t pg, svfloat64_t op) MODE_ATTR
{
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %{{.*}}, %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[2] x !cir.double>, !cir.vector<[2] x !cir.int<u, 1>>, !cir.vector<[2] x !cir.double>) -> !cir.vector<[2] x !cir.double>

// LLVM-SAME: <vscale x 2 x double> [[INACTIVE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x double> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.frintx.nxv2f64(<vscale x 2 x double> [[INACTIVE]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[OP]])
// LLVM:    ret <vscale x 2 x double> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f64,_m,)(inactive, pg, op);
}

// C-LABEL: @test_svrintx_f16_x(
// CPP-LABEL: @_Z18test_svrintx_f16_xu10__SVBool_tu13__SVFloat16_t(
svfloat16_t test_svrintx_f16_x(svbool_t pg, svfloat16_t op) MODE_ATTR
{
// CIR:       %[[UNDEF:.*]] = cir.const #cir.undef : !cir.vector<[8] x !cir.f16>
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %[[UNDEF]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[8] x !cir.f16>, !cir.vector<[8] x !cir.int<u, 1>>, !cir.vector<[8] x !cir.f16>) -> !cir.vector<[8] x !cir.f16>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 8 x half> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.frintx.nxv8f16(<vscale x 8 x half> undef, <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[OP]])
// LLVM:    ret <vscale x 8 x half> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f16,_x,)(pg, op);
}

// C-LABEL: @test_svrintx_f32_x(
// CPP-LABEL: @_Z18test_svrintx_f32_xu10__SVBool_tu13__SVFloat32_t(
svfloat32_t test_svrintx_f32_x(svbool_t pg, svfloat32_t op) MODE_ATTR
{
// CIR:       %[[UNDEF:.*]] = cir.const #cir.undef : !cir.vector<[4] x !cir.float>
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %[[UNDEF]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[4] x !cir.float>, !cir.vector<[4] x !cir.int<u, 1>>, !cir.vector<[4] x !cir.float>) -> !cir.vector<[4] x !cir.float>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x float> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frintx.nxv4f32(<vscale x 4 x float> undef, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[OP]])
// LLVM:    ret <vscale x 4 x float> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f32,_x,)(pg, op);
}

// C-LABEL: @test_svrintx_f64_x(
// CPP-LABEL: @_Z18test_svrintx_f64_xu10__SVBool_tu13__SVFloat64_t(
svfloat64_t test_svrintx_f64_x(svbool_t pg, svfloat64_t op) MODE_ATTR
{
// CIR:       %[[UNDEF:.*]] = cir.const #cir.undef : !cir.vector<[2] x !cir.double>
// CIR:       %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:   (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:       cir.call_llvm_intrinsic "aarch64.sve.frintx" %[[UNDEF]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:   (!cir.vector<[2] x !cir.double>, !cir.vector<[2] x !cir.int<u, 1>>, !cir.vector<[2] x !cir.double>) -> !cir.vector<[2] x !cir.double>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 2 x double> [[OP:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[TMP1:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.frintx.nxv2f64(<vscale x 2 x double> undef, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[OP]])
// LLVM:    ret <vscale x 2 x double> [[TMP1]]
  return SVE_ACLE_FUNC(svrintx,_f64,_x,)(pg, op);
}
