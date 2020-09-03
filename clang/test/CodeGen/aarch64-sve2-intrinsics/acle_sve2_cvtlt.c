// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svfloat32_t test_svcvtlt_f32_f16_m(svfloat32_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvtlt_f32_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvtlt.f32f16(<vscale x 4 x float> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcvtlt_f32_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svcvtlt_f32_f16_m'}}
  return SVE_ACLE_FUNC(svcvtlt_f32,_f16,_m,)(inactive, pg, op);
}

svfloat64_t test_svcvtlt_f64_f32_m(svfloat64_t inactive, svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvtlt_f64_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcvtlt.f64f32(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcvtlt_f64_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svcvtlt_f64_f32_m'}}
  return SVE_ACLE_FUNC(svcvtlt_f64,_f32,_m,)(inactive, pg, op);
}

svfloat32_t test_svcvtlt_f32_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvtlt_f32_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvtlt.f32f16(<vscale x 4 x float> undef, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcvtlt_f32_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svcvtlt_f32_f16_x'}}
  return SVE_ACLE_FUNC(svcvtlt_f32,_f16,_x,)(pg, op);
}

svfloat64_t test_svcvtlt_f64_f32_x(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvtlt_f64_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcvtlt.f64f32(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcvtlt_f64_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svcvtlt_f64_f32_x'}}
  return SVE_ACLE_FUNC(svcvtlt_f64,_f32,_x,)(pg, op);
}
