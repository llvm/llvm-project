// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE2 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE2 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

// Test expected warnings for implicit declaration when +sve2 is missing
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

// Test expected warnings for implicit declaration when +bf16 is missing
// NOTE: +bf16 doesn't currently imply __ARM_FEATURE_SVE_BF16, once the
// implementation is complete it will, at which point -target-feature +bf16
// should be removed.
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE2 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s

// Test expected ambiguous call error for overloaded form when +bf16 is missing
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE2 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify=overload-bf16 -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svbool_t test_svwhilerw_bf16(const bfloat16_t *op1, const bfloat16_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_bf16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nxv8i1.p0bf16(bfloat* %op1, bfloat* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+3 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+2 {{implicit declaration of function 'svwhilerw_bf16'}}
  // overload-bf16-error@+1 {{call to 'svwhilerw' is ambiguous}}
  return SVE_ACLE_FUNC(svwhilerw,_bf16,,)(op1, op2);
}
