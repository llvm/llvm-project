// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svuint32_t test_svrsqrte_u32_z(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svrsqrte_u32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ursqrte.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrsqrte_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svrsqrte_u32_z'}}
  return SVE_ACLE_FUNC(svrsqrte,_u32,_z,)(pg, op);
}

svuint32_t test_svrsqrte_u32_m(svuint32_t inactive, svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svrsqrte_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ursqrte.nxv4i32(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrsqrte_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svrsqrte_u32_m'}}
  return SVE_ACLE_FUNC(svrsqrte,_u32,_m,)(inactive, pg, op);
}

svuint32_t test_svrsqrte_u32_x(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svrsqrte_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ursqrte.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrsqrte_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svrsqrte_u32_x'}}
  return SVE_ACLE_FUNC(svrsqrte,_u32,_x,)(pg, op);
}
