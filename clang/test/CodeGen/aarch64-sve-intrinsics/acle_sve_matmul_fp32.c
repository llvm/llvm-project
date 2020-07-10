// RUN: %clang_cc1 -target-feature +f32mm -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -target-feature +f32mm -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svfloat32_t test_svmmla_f32(svfloat32_t x, svfloat32_t y, svfloat32_t z) {
  // CHECK-LABEL: test_svmmla_f32
  // CHECK: %[[RET:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmmla.nxv4f32(<vscale x 4 x float> %x, <vscale x 4 x float> %y, <vscale x 4 x float> %z)
  // CHECK: ret <vscale x 4 x float> %[[RET]]
  return SVE_ACLE_FUNC(svmmla,_f32,,)(x, y, z);
}
