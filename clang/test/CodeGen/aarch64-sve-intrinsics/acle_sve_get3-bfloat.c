// RUN: %clang_cc1 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -D__ARM_FEATURE_SVE_BF16 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svbfloat16_t test_svget3_bf16_0(svbfloat16x3_t tuple)
{
  // CHECK-LABEL: test_svget3_bf16_0
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 8 x bfloat> %[[EXT]]
  // expected-warning@+1 {{implicit declaration of function 'svget3_bf16'}}
  return SVE_ACLE_FUNC(svget3,_bf16,,)(tuple, 0);
}

svbfloat16_t test_svget3_bf16_1(svbfloat16x3_t tuple)
{
  // CHECK-LABEL: test_svget3_bf16_1
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %tuple, i32 1)
  // CHECK-NEXT: ret <vscale x 8 x bfloat> %[[EXT]]
  // expected-warning@+1 {{implicit declaration of function 'svget3_bf16'}}
  return SVE_ACLE_FUNC(svget3,_bf16,,)(tuple, 1);
}

svbfloat16_t test_svget3_bf16_2(svbfloat16x3_t tuple)
{
  // CHECK-LABEL: test_svget3_bf16_2
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %tuple, i32 2)
  // CHECK-NEXT: ret <vscale x 8 x bfloat> %[[EXT]]
  // expected-warning@+1 {{implicit declaration of function 'svget3_bf16'}}
  return SVE_ACLE_FUNC(svget3,_bf16,,)(tuple, 2);
}
