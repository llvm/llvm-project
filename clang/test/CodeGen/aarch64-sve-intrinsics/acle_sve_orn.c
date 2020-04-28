// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svbool_t test_svorn_b_z(svbool_t pg, svbool_t op1, svbool_t op2)
{
  // CHECK-LABEL: test_svorn_b_z
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.orn.z.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %op1, <vscale x 16 x i1> %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorn,_b,_z,)(pg, op1, op2);
}
