// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
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

float16_t test_svadda_f16(svbool_t pg, float16_t initial, svfloat16_t op)
{
  // CHECK-LABEL: test_svadda_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call half @llvm.aarch64.sve.fadda.nxv8f16(<vscale x 8 x i1> %[[PG]], half %initial, <vscale x 8 x half> %op)
  // CHECK: ret half %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svadda,_f16,,)(pg, initial, op);
}

float32_t test_svadda_f32(svbool_t pg, float32_t initial, svfloat32_t op)
{
  // CHECK-LABEL: test_svadda_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call float @llvm.aarch64.sve.fadda.nxv4f32(<vscale x 4 x i1> %[[PG]], float %initial, <vscale x 4 x float> %op)
  // CHECK: ret float %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svadda,_f32,,)(pg, initial, op);
}

float64_t test_svadda_f64(svbool_t pg, float64_t initial, svfloat64_t op)
{
  // CHECK-LABEL: test_svadda_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call double @llvm.aarch64.sve.fadda.nxv2f64(<vscale x 2 x i1> %[[PG]], double %initial, <vscale x 2 x double> %op)
  // CHECK: ret double %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svadda,_f64,,)(pg, initial, op);
}
