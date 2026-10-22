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
// Tests for SVE NOR intrinsics
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

// C-LABEL: @test_svnor_b_z(
// CPP-LABEL: @_Z14test_svnor_b_zu10__SVBool_tS_S_(
svbool_t test_svnor_b_z(svbool_t pg, svbool_t op1, svbool_t op2) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.nor.z" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<[16] x !cir.int<u, 1>>, !cir.vector<[16] x !cir.int<u, 1>>, !cir.vector<[16] x !cir.int<u, 1>>) -> !cir.vector<[16] x !cir.int<u, 1>>

// LLVM-SAME: <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i1> [[OP1:%.*]], <vscale x 16 x i1> [[OP2:%.*]])
// LLVM:    [[TMP0:%.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.nor.z.nxv16i1(<vscale x 16 x i1> [[PG]], <vscale x 16 x i1> [[OP1]], <vscale x 16 x i1> [[OP2]])
// LLVM:    ret <vscale x 16 x i1> [[TMP0]]
  return SVE_ACLE_FUNC(svnor,_b,_z,)(pg, op1, op2);
}
