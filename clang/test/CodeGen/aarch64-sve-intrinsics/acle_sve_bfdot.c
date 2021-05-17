// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svfloat32_t test_bfdot_f32(svfloat32_t x, svbfloat16_t y, svbfloat16_t z) {
  // CHECK-LABEL: test_bfdot_f32
  // CHECK: %[[RET:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot(<vscale x 4 x float> %x, <vscale x 8 x bfloat> %y, <vscale x 8 x bfloat> %z)
  // CHECK: ret <vscale x 4 x float> %[[RET]]
  return SVE_ACLE_FUNC(svbfdot, _f32, , )(x, y, z);
}

svfloat32_t test_bfdot_lane_0_f32(svfloat32_t x, svbfloat16_t y, svbfloat16_t z) {
  // CHECK-LABEL: test_bfdot_lane_0_f32
  // CHECK: %[[RET:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float> %x, <vscale x 8 x bfloat> %y, <vscale x 8 x bfloat> %z, i64 0)
  // CHECK: ret <vscale x 4 x float> %[[RET]]
  return SVE_ACLE_FUNC(svbfdot_lane, _f32, , )(x, y, z, 0);
}

svfloat32_t test_bfdot_lane_3_f32(svfloat32_t x, svbfloat16_t y, svbfloat16_t z) {
  // CHECK-LABEL: test_bfdot_lane_3_f32
  // CHECK: %[[RET:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot.lane(<vscale x 4 x float> %x, <vscale x 8 x bfloat> %y, <vscale x 8 x bfloat> %z, i64 3)
  // CHECK: ret <vscale x 4 x float> %[[RET]]
  return SVE_ACLE_FUNC(svbfdot_lane, _f32, , )(x, y, z, 3);
}

svfloat32_t test_bfdot_n_f32(svfloat32_t x, svbfloat16_t y, bfloat16_t z) {
  // CHECK-LABEL: test_bfdot_n_f32
  // CHECK: %[[SPLAT:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.x.nxv8bf16(bfloat %z)
  // CHECK: %[[RET:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.bfdot(<vscale x 4 x float> %x, <vscale x 8 x bfloat> %y, <vscale x 8 x bfloat> %[[SPLAT]])
  // CHECK: ret <vscale x 4 x float> %[[RET]]
  return SVE_ACLE_FUNC(svbfdot, _n_f32, , )(x, y, z);
}
