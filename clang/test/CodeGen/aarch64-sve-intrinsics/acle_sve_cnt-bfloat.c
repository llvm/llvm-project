// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svuint16_t test_svcnt_bf16_z(svbool_t pg, svbfloat16_t op) {
  // CHECK-LABEL: test_svcnt_bf16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8bf16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> %[[PG]], <vscale x 8 x bfloat> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcnt_bf16_z'}}
  return SVE_ACLE_FUNC(svcnt, _bf16, _z, )(pg, op);
}

svuint16_t test_svcnt_bf16_m(svuint16_t inactive, svbool_t pg, svbfloat16_t op) {
  // CHECK-LABEL: test_svcnt_bf16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8bf16(<vscale x 8 x i16> %inactive, <vscale x 8 x i1> %[[PG]], <vscale x 8 x bfloat> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcnt_bf16_m'}}
  return SVE_ACLE_FUNC(svcnt, _bf16, _m, )(inactive, pg, op);
}
svuint16_t test_svcnt_bf16_x(svbool_t pg, svbfloat16_t op) {
  // CHECK-LABEL: test_svcnt_bf16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8bf16(<vscale x 8 x i16> undef, <vscale x 8 x i1> %[[PG]], <vscale x 8 x bfloat> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcnt_bf16_x'}}
  return SVE_ACLE_FUNC(svcnt, _bf16, _x, )(pg, op);
}
