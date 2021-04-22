// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svfloat16_t test_svtsmul_f16(svfloat16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svtsmul_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ftsmul.x.nxv8f16(<vscale x 8 x half> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtsmul,_f16,,)(op1, op2);
}

svfloat32_t test_svtsmul_f32(svfloat32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svtsmul_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ftsmul.x.nxv4f32(<vscale x 4 x float> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtsmul,_f32,,)(op1, op2);
}

svfloat64_t test_svtsmul_f64(svfloat64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svtsmul_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ftsmul.x.nxv2f64(<vscale x 2 x double> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtsmul,_f64,,)(op1, op2);
}
