// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svbfloat16_t test_svcvtnt_bf16_f32_x(svbfloat16_t even, svbool_t pg, svfloat32_t op) {
  // CHECK-LABEL: test_svcvtnt_bf16_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.fcvtnt.bf16f32(<vscale x 8 x bfloat> %even, <vscale x 8 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvtnt_bf16, _f32, _x, )(even, pg, op);
}

svbfloat16_t test_svcvtnt_bf16_f32_m(svbfloat16_t even, svbool_t pg, svfloat32_t op) {
  // CHECK-LABEL: test_svcvtnt_bf16_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.fcvtnt.bf16f32(<vscale x 8 x bfloat> %even, <vscale x 8 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvtnt_bf16, _f32, _m, )(even, pg, op);
}
