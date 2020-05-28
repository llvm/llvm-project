// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svfloat16_t test_svrecpe_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svrecpe_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.frecpe.x.nxv8f16(<vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrecpe,_f16,,)(op);
}

svfloat32_t test_svrecpe_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svrecpe_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frecpe.x.nxv4f32(<vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrecpe,_f32,,)(op);
}

svfloat64_t test_svrecpe_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svrecpe_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.frecpe.x.nxv2f64(<vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrecpe,_f64,,)(op);
}
