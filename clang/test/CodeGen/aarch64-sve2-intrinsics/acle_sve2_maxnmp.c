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

svfloat16_t test_svmaxnmp_f16_m(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // CHECK-LABEL: test_svmaxnmp_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmaxnmp.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmaxnmp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svmaxnmp_f16_m'}}
  return SVE_ACLE_FUNC(svmaxnmp,_f16,_m,)(pg, op1, op2);
}

svfloat32_t test_svmaxnmp_f32_m(svbool_t pg, svfloat32_t op1, svfloat32_t op2)
{
  // CHECK-LABEL: test_svmaxnmp_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmaxnmp.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmaxnmp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svmaxnmp_f32_m'}}
  return SVE_ACLE_FUNC(svmaxnmp,_f32,_m,)(pg, op1, op2);
}

svfloat64_t test_svmaxnmp_f64_m(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
{
  // CHECK-LABEL: test_svmaxnmp_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmaxnmp.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmaxnmp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svmaxnmp_f64_m'}}
  return SVE_ACLE_FUNC(svmaxnmp,_f64,_m,)(pg, op1, op2);
}

svfloat16_t test_svmaxnmp_f16_x(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // CHECK-LABEL: test_svmaxnmp_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmaxnmp.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmaxnmp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svmaxnmp_f16_x'}}
  return SVE_ACLE_FUNC(svmaxnmp,_f16,_x,)(pg, op1, op2);
}

svfloat32_t test_svmaxnmp_f32_x(svbool_t pg, svfloat32_t op1, svfloat32_t op2)
{
  // CHECK-LABEL: test_svmaxnmp_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmaxnmp.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmaxnmp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svmaxnmp_f32_x'}}
  return SVE_ACLE_FUNC(svmaxnmp,_f32,_x,)(pg, op1, op2);
}

svfloat64_t test_svmaxnmp_f64_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
{
  // CHECK-LABEL: test_svmaxnmp_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmaxnmp.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmaxnmp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svmaxnmp_f64_x'}}
  return SVE_ACLE_FUNC(svmaxnmp,_f64,_x,)(pg, op1, op2);
}
