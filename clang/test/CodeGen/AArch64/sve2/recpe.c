// REQUIRES: aarch64-registered-target

// DEFINE: %{optimize} = opt -passes=mem2reg -S

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -target-feature +sve2                        -fclangir -emit-cir -disable-O0-optnone -o - %s                | FileCheck %s --check-prefixes=C,CIR %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -target-feature +sve2 -DSVE_OVERLOADED_FORMS -fclangir -emit-cir -disable-O0-optnone -o - %s                | FileCheck %s --check-prefixes=C,CIR %}

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -target-feature +sve2                        -fclangir -emit-llvm -disable-O0-optnone -o -        %s | %{optimize} | FileCheck %s --check-prefixes=C,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -target-feature +sve2 -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o -        %s | %{optimize} | FileCheck %s --check-prefixes=C,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -target-feature +sve2 -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o - -x c++ %s | %{optimize} | FileCheck %s --check-prefixes=CPP,LLVM %}

// RUN:                   %clang_cc1_cg_arm64_sve -target-feature +sve2                                  -emit-llvm -disable-O0-optnone -o -        %s | %{optimize} | FileCheck %s --check-prefixes=C,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -target-feature +sve2 -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o -        %s | %{optimize} | FileCheck %s --check-prefixes=C,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -target-feature +sve2 -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o - -x c++ %s | %{optimize} | FileCheck %s --check-prefixes=CPP,LLVM

//=============================================================================
// NOTES
//
// Tests for SVE2 RECPE intrinsics
//=============================================================================

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

// C-LABEL: @test_svrecpe_u32_z(
// CPP-LABEL: @_Z18test_svrecpe_u32_zu10__SVBool_tu12__SVUint32_t(
svuint32_t test_svrecpe_u32_z(svbool_t pg, svuint32_t op)
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} : (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.urecpe" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<[4] x !u32i>, !cir.vector<[4] x !cir.int<u, 1>>, !cir.vector<[4] x !u32i>) -> !cir.vector<[4] x !u32i>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x i32> [[OP:%.*]])
// LLVM:    [[PG4:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urecpe.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> [[PG4]], <vscale x 4 x i32> [[OP]])
// LLVM:    ret <vscale x 4 x i32> [[RES]]
  return SVE_ACLE_FUNC(svrecpe,_u32,_z,)(pg, op);
}

// C-LABEL: @test_svrecpe_u32_m(
// CPP-LABEL: @_Z18test_svrecpe_u32_mu12__SVUint32_tu10__SVBool_tS_(
svuint32_t test_svrecpe_u32_m(svuint32_t inactive, svbool_t pg, svuint32_t op)
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} : (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.urecpe" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<[4] x !u32i>, !cir.vector<[4] x !cir.int<u, 1>>, !cir.vector<[4] x !u32i>) -> !cir.vector<[4] x !u32i>

// LLVM-SAME: <vscale x 4 x i32> [[INACTIVE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x i32> [[OP:%.*]])
// LLVM:    [[PG4:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urecpe.nxv4i32(<vscale x 4 x i32> [[INACTIVE]], <vscale x 4 x i1> [[PG4]], <vscale x 4 x i32> [[OP]])
// LLVM:    ret <vscale x 4 x i32> [[RES]]
  return SVE_ACLE_FUNC(svrecpe,_u32,_m,)(inactive, pg, op);
}

// C-LABEL: @test_svrecpe_u32_x(
// CPP-LABEL: @_Z18test_svrecpe_u32_xu10__SVBool_tu12__SVUint32_t(
svuint32_t test_svrecpe_u32_x(svbool_t pg, svuint32_t op)
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} : (!cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.urecpe" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<[4] x !u32i>, !cir.vector<[4] x !cir.int<u, 1>>, !cir.vector<[4] x !u32i>) -> !cir.vector<[4] x !u32i>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 4 x i32> [[OP:%.*]])
// LLVM:    [[PG4:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urecpe.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i1> [[PG4]], <vscale x 4 x i32> [[OP]])
// LLVM:    ret <vscale x 4 x i32> [[RES]]
  return SVE_ACLE_FUNC(svrecpe,_u32,_x,)(pg, op);
}
