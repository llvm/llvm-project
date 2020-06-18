// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_MATMUL_FP64 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_MATMUL_FP64 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svfloat64_t test_svmmla_f64(svfloat64_t x, svfloat64_t y, svfloat64_t z) {
  // CHECK-LABEL: test_svmmla_f64
  // CHECK: %[[RET:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmmla.nxv2f64(<vscale x 2 x double> %x, <vscale x 2 x double> %y, <vscale x 2 x double> %z)
  // CHECK: ret <vscale x 2 x double> %[[RET]]
  return SVE_ACLE_FUNC(svmmla,_f64,,)(x, y, z);
}
