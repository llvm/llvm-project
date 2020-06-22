// RUN: %clang_cc1 -D__ARM_FEATURE_SVE_MATMUL_FP64 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE_MATMUL_FP64 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svbfloat16_t test_svld1ro_bf16(svbool_t pg, const bfloat16_t *base) {
  // CHECK-LABEL: test_svld1ro_bf16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1ro.nxv8bf16(<vscale x 8 x i1> %[[PG]], bfloat* %base)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _bf16, , )(pg, base);
}
