// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svfloat16_t test_svrintn_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svrintn_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.frintn.nxv8f16(<vscale x 8 x half> zeroinitializer, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f16,_z,)(pg, op);
}

svfloat32_t test_svrintn_f32_z(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svrintn_f32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frintn.nxv4f32(<vscale x 4 x float> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f32,_z,)(pg, op);
}

svfloat64_t test_svrintn_f64_z(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svrintn_f64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.frintn.nxv2f64(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f64,_z,)(pg, op);
}

svfloat16_t test_svrintn_f16_m(svfloat16_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svrintn_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.frintn.nxv8f16(<vscale x 8 x half> %inactive, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f16,_m,)(inactive, pg, op);
}

svfloat32_t test_svrintn_f32_m(svfloat32_t inactive, svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svrintn_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frintn.nxv4f32(<vscale x 4 x float> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f32,_m,)(inactive, pg, op);
}

svfloat64_t test_svrintn_f64_m(svfloat64_t inactive, svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svrintn_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.frintn.nxv2f64(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f64,_m,)(inactive, pg, op);
}

svfloat16_t test_svrintn_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svrintn_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.frintn.nxv8f16(<vscale x 8 x half> undef, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f16,_x,)(pg, op);
}

svfloat32_t test_svrintn_f32_x(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svrintn_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frintn.nxv4f32(<vscale x 4 x float> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f32,_x,)(pg, op);
}

svfloat64_t test_svrintn_f64_x(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svrintn_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.frintn.nxv2f64(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrintn,_f64,_x,)(pg, op);
}
