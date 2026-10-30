// REQUIRES: aarch64-registered-target

// DEFINE: %{optimize} = opt -passes=mem2reg -S

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve                        -fclangir -emit-cir -disable-O0-optnone -o - %s                | FileCheck %s --check-prefixes=C,CIR %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-cir -disable-O0-optnone -o - %s                | FileCheck %s --check-prefixes=C,CIR %}

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve                        -fclangir -emit-llvm -disable-O0-optnone -o -        %s | %{optimize} | FileCheck %s --check-prefixes=C,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o -        %s | %{optimize} | FileCheck %s --check-prefixes=C,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS -fclangir -emit-llvm -disable-O0-optnone -o - -x c++ %s | %{optimize} | FileCheck %s --check-prefixes=CPP,LLVM %}

// RUN:                   %clang_cc1_cg_arm64_sve                                  -emit-llvm -disable-O0-optnone -o -        %s | %{optimize} | FileCheck %s --check-prefixes=C,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o -        %s | %{optimize} | FileCheck %s --check-prefixes=C,LLVM
// RUN:                   %clang_cc1_cg_arm64_sve -DSVE_OVERLOADED_FORMS           -emit-llvm -disable-O0-optnone -o - -x c++ %s | %{optimize} | FileCheck %s --check-prefixes=CPP,LLVM

// RUN:                   %clang_cc1_cg_arm64_sme                        -S -disable-O0-optnone -Werror -Wall -o /dev/null %s

//=============================================================================
// NOTES
//
// Tests for SVE PFALSE intrinsics (svpfalse_b only; svpfalse_c is SVE2.1)
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

// C-LABEL: @test_svpfalse_b(
// CPP-LABEL: @_Z15test_svpfalse_bv(
svbool_t test_svpfalse_b(void) MODE_ATTR
{
// CIR:     cir.const #cir.zero : !cir.vector<[16] x !cir.int<u, 1>>

// LLVM:    ret <vscale x 16 x i1> zeroinitializer
  return SVE_ACLE_FUNC(svpfalse,_b,,)();
}
