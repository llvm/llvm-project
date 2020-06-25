// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE2 -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE2 -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +bf16 -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE2 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svbfloat16_t test_svtbl2_bf16(svbfloat16x2_t data, svuint16_t indices) {
  // CHECK-LABEL: test_svtbl2_bf16
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tbl2.nxv8bf16(<vscale x 8 x bfloat> %[[V0]], <vscale x 8 x bfloat> %[[V1]], <vscale x 8 x i16> %indices)
  // CHECK-NEXT: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_bf16'}}
  return SVE_ACLE_FUNC(svtbl2, _bf16, , )(data, indices);
}
