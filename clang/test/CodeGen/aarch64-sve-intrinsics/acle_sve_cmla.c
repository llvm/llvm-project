// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svfloat16_t test_svcmla_f16_z(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svcmla_f16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.sel.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %[[SEL]], <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 0)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 0);
}

svfloat16_t test_svcmla_f16_z_1(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svcmla_f16_z_1
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.sel.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %[[SEL]], <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 90)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 90);
}

svfloat16_t test_svcmla_f16_z_2(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svcmla_f16_z_2
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.sel.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %[[SEL]], <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 180)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 180);
}

svfloat16_t test_svcmla_f16_z_3(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svcmla_f16_z_3
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.sel.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %[[SEL]], <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 270)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 270);
}

svfloat32_t test_svcmla_f32_z(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svcmla_f32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.sel.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcmla.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %[[SEL]], <vscale x 4 x float> %op2, <vscale x 4 x float> %op3, i32 0)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f32,_z,)(pg, op1, op2, op3, 0);
}

svfloat64_t test_svcmla_f64_z(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // CHECK-LABEL: test_svcmla_f64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.sel.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcmla.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %[[SEL]], <vscale x 2 x double> %op2, <vscale x 2 x double> %op3, i32 90)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f64,_z,)(pg, op1, op2, op3, 90);
}

svfloat16_t test_svcmla_f16_m(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svcmla_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 180)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f16,_m,)(pg, op1, op2, op3, 180);
}

svfloat32_t test_svcmla_f32_m(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svcmla_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcmla.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %op3, i32 270)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f32,_m,)(pg, op1, op2, op3, 270);
}

svfloat64_t test_svcmla_f64_m(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // CHECK-LABEL: test_svcmla_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcmla.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2, <vscale x 2 x double> %op3, i32 0)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f64,_m,)(pg, op1, op2, op3, 0);
}

svfloat16_t test_svcmla_f16_x(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svcmla_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 90)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f16,_x,)(pg, op1, op2, op3, 90);
}

svfloat32_t test_svcmla_f32_x(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svcmla_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcmla.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %op3, i32 180)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f32,_x,)(pg, op1, op2, op3, 180);
}

svfloat64_t test_svcmla_f64_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // CHECK-LABEL: test_svcmla_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcmla.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2, <vscale x 2 x double> %op3, i32 270)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla,_f64,_x,)(pg, op1, op2, op3, 270);
}

svfloat16_t test_svcmla_lane_f16(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.lane.nxv8f16(<vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 0, i32 0)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, 0, 0);
}

svfloat16_t test_svcmla_lane_f16_1(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_f16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcmla.lane.nxv8f16(<vscale x 8 x half> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 3, i32 90)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, 3, 90);
}

svfloat32_t test_svcmla_lane_f32(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcmla.lane.nxv4f32(<vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %op3, i32 0, i32 180)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla_lane,_f32,,)(op1, op2, op3, 0, 180);
}

svfloat32_t test_svcmla_lane_f32_1(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_f32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcmla.lane.nxv4f32(<vscale x 4 x float> %op1, <vscale x 4 x float> %op2, <vscale x 4 x float> %op3, i32 1, i32 270)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmla_lane,_f32,,)(op1, op2, op3, 1, 270);
}
