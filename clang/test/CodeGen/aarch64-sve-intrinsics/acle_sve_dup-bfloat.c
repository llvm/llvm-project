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

svbfloat16_t test_svdup_n_bf16(bfloat16_t op) {
  // CHECK-LABEL: test_svdup_n_bf16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.x.nxv8bf16(bfloat %op)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svdup_n_bf16'}}
  return SVE_ACLE_FUNC(svdup, _n, _bf16, )(op);
}

svbfloat16_t test_svdup_n_bf16_z(svbool_t pg, bfloat16_t op) {
  // CHECK-LABEL: test_svdup_n_bf16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.nxv8bf16(<vscale x 8 x bfloat> zeroinitializer, <vscale x 8 x i1> %[[PG]], bfloat %op)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svdup_n_bf16_z'}}
  return SVE_ACLE_FUNC(svdup, _n, _bf16_z, )(pg, op);
}

svbfloat16_t test_svdup_n_bf16_m(svbfloat16_t inactive, svbool_t pg, bfloat16_t op) {
  // CHECK-LABEL: test_svdup_n_bf16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.nxv8bf16(<vscale x 8 x bfloat> %inactive, <vscale x 8 x i1> %[[PG]], bfloat %op)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svdup_n_bf16_m'}}
  return SVE_ACLE_FUNC(svdup, _n, _bf16_m, )(inactive, pg, op);
}

svbfloat16_t test_svdup_n_bf16_x(svbool_t pg, bfloat16_t op) {
  // CHECK-LABEL: test_svdup_n_bf16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.nxv8bf16(<vscale x 8 x bfloat> undef, <vscale x 8 x i1> %[[PG]], bfloat %op)
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svdup_n_bf16_x'}}
  return SVE_ACLE_FUNC(svdup, _n, _bf16_x, )(pg, op);
}

svbfloat16_t test_svdup_lane_bf16(svbfloat16_t data, uint16_t index)
{
  // CHECK-LABEL: test_svdup_lane_bf16
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tbl.nxv8bf16(<vscale x 8 x bfloat> %data, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x bfloat> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svdup_lane_bf16'}}
  return SVE_ACLE_FUNC(svdup_lane,_bf16,,)(data, index);
}
