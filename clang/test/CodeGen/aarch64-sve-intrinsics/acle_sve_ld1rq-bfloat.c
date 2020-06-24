// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svbfloat16_t test_svld1rq_bf16(svbool_t pg, const bfloat16_t *base)
{
  // CHECK-LABEL: test_svld1rq_bf16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1rq.nxv8bf16(<vscale x 8 x i1> %[[PG]], bfloat* %base)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svld1rq_bf16'}}
  return SVE_ACLE_FUNC(svld1rq,_bf16,,)(pg, base);
}
