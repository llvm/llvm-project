// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svbfloat16_t test_svdupq_lane_bf16(svbfloat16_t data, uint64_t index) {
  // CHECK-LABEL: test_svdupq_lane_bf16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dupq.lane.nxv8bf16(<vscale x 8 x bfloat> %data, i64 %index)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svdupq_lane_bf16'}}
  return SVE_ACLE_FUNC(svdupq_lane, _bf16, , )(data, index);
}
svbfloat16_t test_svdupq_n_bf16(bfloat16_t x0, bfloat16_t x1, bfloat16_t x2, bfloat16_t x3,
                                bfloat16_t x4, bfloat16_t x5, bfloat16_t x6, bfloat16_t x7) {
  // CHECK-LABEL: test_svdupq_n_bf16
  // CHECK: %[[ALLOCA:.*]] = alloca [8 x bfloat], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [8 x bfloat], [8 x bfloat]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store bfloat %x0, bfloat* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [8 x bfloat], [8 x bfloat]* %[[ALLOCA]], i64 0, i64 7
  // CHECK: store bfloat %x7, bfloat* %[[GEP]], align 2
  // CHECK-NOT: store
  // CHECK: call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1rq.nxv8bf16(<vscale x 8 x i1> %{{.*}}, bfloat* nonnull %[[BASE]])
  // CHECK: ret <vscale x 8 x bfloat> %[[LOAD]]
  // expected-warning@+1 {{implicit declaration of function 'svdupq_n_bf16'}}
  return SVE_ACLE_FUNC(svdupq, _n, _bf16, )(x0, x1, x2, x3, x4, x5, x6, x7);
}
