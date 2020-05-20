// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svbool_t test_svpfirst_b(svbool_t pg, svbool_t op)
{
  // CHECK-LABEL: test_svpfirst_b
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.pfirst.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %op)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svpfirst,_b,,)(pg, op);
}
