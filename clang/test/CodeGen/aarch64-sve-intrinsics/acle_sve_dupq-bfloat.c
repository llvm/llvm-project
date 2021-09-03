// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svbfloat16_t test_svdupq_lane_bf16(svbfloat16_t data, uint64_t index) {
  // CHECK-LABEL: test_svdupq_lane_bf16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dupq.lane.nxv8bf16(<vscale x 8 x bfloat> %data, i64 %index)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svdupq_lane_bf16'}}
  return SVE_ACLE_FUNC(svdupq_lane, _bf16, , )(data, index);
}
svbfloat16_t test_svdupq_n_bf16(bfloat16_t x0, bfloat16_t x1, bfloat16_t x2, bfloat16_t x3,
                                bfloat16_t x4, bfloat16_t x5, bfloat16_t x6, bfloat16_t x7) {
  // CHECK-LABEL: test_svdupq_n_bf16
  // CHECK: insertelement <8 x bfloat> undef, bfloat %x0, i32 0
  // <assume other insertelement>
  // CHECK: %[[VEC:.*]] = insertelement <8 x bfloat> %[[X:.*]], bfloat %x7, i32 7
  // CHECK-NOT: insertelement
  // CHECK: %[[INS:.*]] = call <vscale x 8 x bfloat> @llvm.experimental.vector.insert.nxv8bf16.v8bf16(<vscale x 8 x bfloat> undef, <8 x bfloat> %[[VEC]], i64 0)
  // CHECK: %[[DUPQ:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dupq.lane.nxv8bf16(<vscale x 8 x bfloat> %[[INS]], i64 0)
  // CHECK: ret <vscale x 8 x bfloat> %[[DUPQ]]
  // expected-warning@+1 {{implicit declaration of function 'svdupq_n_bf16'}}
  return SVE_ACLE_FUNC(svdupq, _n, _bf16, )(x0, x1, x2, x3, x4, x5, x6, x7);
}
