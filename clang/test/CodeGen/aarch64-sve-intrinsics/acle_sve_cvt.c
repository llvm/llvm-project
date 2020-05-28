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

svint16_t test_svcvt_s16_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s16_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.fcvtzs.nxv8i16.nxv8f16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s16,_f16,_z,)(pg, op);
}

svint16_t test_svcvt_s16_f16_m(svint16_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s16_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.fcvtzs.nxv8i16.nxv8f16(<vscale x 8 x i16> %inactive, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s16,_f16,_m,)(inactive, pg, op);
}

svint16_t test_svcvt_s16_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s16_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.fcvtzs.nxv8i16.nxv8f16(<vscale x 8 x i16> undef, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s16,_f16,_x,)(pg, op);
}

svuint16_t test_svcvt_u16_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u16_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.fcvtzu.nxv8i16.nxv8f16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u16,_f16,_z,)(pg, op);
}

svuint16_t test_svcvt_u16_f16_m(svuint16_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u16_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.fcvtzu.nxv8i16.nxv8f16(<vscale x 8 x i16> %inactive, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u16,_f16,_m,)(inactive, pg, op);
}

svuint16_t test_svcvt_u16_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u16_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.fcvtzu.nxv8i16.nxv8f16(<vscale x 8 x i16> undef, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u16,_f16,_x,)(pg, op);
}

svint32_t test_svcvt_s32_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.i32f16(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f16,_z,)(pg, op);
}

svint32_t test_svcvt_s32_f32_z(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.nxv4i32.nxv4f32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f32,_z,)(pg, op);
}

svint32_t test_svcvt_s32_f64_z(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.i32f64(<vscale x 4 x i32> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f64,_z,)(pg, op);
}

svint32_t test_svcvt_s32_f16_m(svint32_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.i32f16(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f16,_m,)(inactive, pg, op);
}

svint32_t test_svcvt_s32_f32_m(svint32_t inactive, svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.nxv4i32.nxv4f32(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f32,_m,)(inactive, pg, op);
}

svint32_t test_svcvt_s32_f64_m(svint32_t inactive, svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.i32f64(<vscale x 4 x i32> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f64,_m,)(inactive, pg, op);
}

svint32_t test_svcvt_s32_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.i32f16(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f16,_x,)(pg, op);
}

svint32_t test_svcvt_s32_f32_x(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.nxv4i32.nxv4f32(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f32,_x,)(pg, op);
}

svint32_t test_svcvt_s32_f64_x(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_s32_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzs.i32f64(<vscale x 4 x i32> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s32,_f64,_x,)(pg, op);
}

svint64_t test_svcvt_s64_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.i64f16(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f16,_z,)(pg, op);
}

svint64_t test_svcvt_s64_f32_z(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f32_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.i64f32(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f32,_z,)(pg, op);
}

svint64_t test_svcvt_s64_f64_z(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.nxv2i64.nxv2f64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f64,_z,)(pg, op);
}

svint64_t test_svcvt_s64_f16_m(svint64_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.i64f16(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f16,_m,)(inactive, pg, op);
}

svint64_t test_svcvt_s64_f32_m(svint64_t inactive, svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.i64f32(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f32,_m,)(inactive, pg, op);
}

svint64_t test_svcvt_s64_f64_m(svint64_t inactive, svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.nxv2i64.nxv2f64(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f64,_m,)(inactive, pg, op);
}

svint64_t test_svcvt_s64_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.i64f16(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f16,_x,)(pg, op);
}

svint64_t test_svcvt_s64_f32_x(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.i64f32(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f32,_x,)(pg, op);
}

svint64_t test_svcvt_s64_f64_x(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_s64_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzs.nxv2i64.nxv2f64(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_s64,_f64,_x,)(pg, op);
}

svuint32_t test_svcvt_u32_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.i32f16(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f16,_z,)(pg, op);
}

svuint32_t test_svcvt_u32_f32_z(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.nxv4i32.nxv4f32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f32,_z,)(pg, op);
}

svuint32_t test_svcvt_u32_f64_z(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.i32f64(<vscale x 4 x i32> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f64,_z,)(pg, op);
}

svuint32_t test_svcvt_u32_f16_m(svuint32_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.i32f16(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f16,_m,)(inactive, pg, op);
}

svuint32_t test_svcvt_u32_f32_m(svuint32_t inactive, svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.nxv4i32.nxv4f32(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f32,_m,)(inactive, pg, op);
}

svuint32_t test_svcvt_u32_f64_m(svuint32_t inactive, svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.i32f64(<vscale x 4 x i32> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f64,_m,)(inactive, pg, op);
}

svuint32_t test_svcvt_u32_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.i32f16(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f16,_x,)(pg, op);
}

svuint32_t test_svcvt_u32_f32_x(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.nxv4i32.nxv4f32(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f32,_x,)(pg, op);
}

svuint32_t test_svcvt_u32_f64_x(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_u32_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.fcvtzu.i32f64(<vscale x 4 x i32> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u32,_f64,_x,)(pg, op);
}

svuint64_t test_svcvt_u64_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.i64f16(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f16,_z,)(pg, op);
}

svuint64_t test_svcvt_u64_f32_z(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f32_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.i64f32(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f32,_z,)(pg, op);
}

svuint64_t test_svcvt_u64_f64_z(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.nxv2i64.nxv2f64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f64,_z,)(pg, op);
}

svuint64_t test_svcvt_u64_f16_m(svuint64_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.i64f16(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f16,_m,)(inactive, pg, op);
}

svuint64_t test_svcvt_u64_f32_m(svuint64_t inactive, svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.i64f32(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f32,_m,)(inactive, pg, op);
}

svuint64_t test_svcvt_u64_f64_m(svuint64_t inactive, svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.nxv2i64.nxv2f64(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f64,_m,)(inactive, pg, op);
}

svuint64_t test_svcvt_u64_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.i64f16(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f16,_x,)(pg, op);
}

svuint64_t test_svcvt_u64_f32_x(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.i64f32(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f32,_x,)(pg, op);
}

svuint64_t test_svcvt_u64_f64_x(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_u64_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.fcvtzu.nxv2i64.nxv2f64(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_u64,_f64,_x,)(pg, op);
}

svfloat16_t test_svcvt_f16_s32_z(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_s32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.scvtf.f16i32(<vscale x 8 x half> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_s32,_z,)(pg, op);
}

svfloat32_t test_svcvt_f32_s32_z(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f32_s32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.scvtf.nxv4f32.nxv4i32(<vscale x 4 x float> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_s32,_z,)(pg, op);
}

svfloat64_t test_svcvt_f64_s32_z(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_s32_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.scvtf.f64i32(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_s32,_z,)(pg, op);
}

svfloat16_t test_svcvt_f16_s32_m(svfloat16_t inactive, svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.scvtf.f16i32(<vscale x 8 x half> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_s32,_m,)(inactive, pg, op);
}

svfloat32_t test_svcvt_f32_s32_m(svfloat32_t inactive, svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f32_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.scvtf.nxv4f32.nxv4i32(<vscale x 4 x float> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_s32,_m,)(inactive, pg, op);
}

svfloat64_t test_svcvt_f64_s32_m(svfloat64_t inactive, svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.scvtf.f64i32(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_s32,_m,)(inactive, pg, op);
}

svfloat16_t test_svcvt_f16_s32_x(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.scvtf.f16i32(<vscale x 8 x half> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_s32,_x,)(pg, op);
}

svfloat32_t test_svcvt_f32_s32_x(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f32_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.scvtf.nxv4f32.nxv4i32(<vscale x 4 x float> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_s32,_x,)(pg, op);
}

svfloat64_t test_svcvt_f64_s32_x(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.scvtf.f64i32(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_s32,_x,)(pg, op);
}

svfloat16_t test_svcvt_f16_s64_z(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_s64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.scvtf.f16i64(<vscale x 8 x half> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_s64,_z,)(pg, op);
}

svfloat32_t test_svcvt_f32_s64_z(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_s64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.scvtf.f32i64(<vscale x 4 x float> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_s64,_z,)(pg, op);
}

svfloat64_t test_svcvt_f64_s64_z(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f64_s64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.scvtf.nxv2f64.nxv2i64(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_s64,_z,)(pg, op);
}

svfloat16_t test_svcvt_f16_s64_m(svfloat16_t inactive, svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.scvtf.f16i64(<vscale x 8 x half> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_s64,_m,)(inactive, pg, op);
}

svfloat32_t test_svcvt_f32_s64_m(svfloat32_t inactive, svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.scvtf.f32i64(<vscale x 4 x float> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_s64,_m,)(inactive, pg, op);
}

svfloat64_t test_svcvt_f64_s64_m(svfloat64_t inactive, svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f64_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.scvtf.nxv2f64.nxv2i64(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_s64,_m,)(inactive, pg, op);
}

svfloat16_t test_svcvt_f16_s64_x(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.scvtf.f16i64(<vscale x 8 x half> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_s64,_x,)(pg, op);
}

svfloat32_t test_svcvt_f32_s64_x(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.scvtf.f32i64(<vscale x 4 x float> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_s64,_x,)(pg, op);
}

svfloat64_t test_svcvt_f64_s64_x(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svcvt_f64_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.scvtf.nxv2f64.nxv2i64(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_s64,_x,)(pg, op);
}

svfloat16_t test_svcvt_f16_u32_z(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_u32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ucvtf.f16i32(<vscale x 8 x half> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_u32,_z,)(pg, op);
}

svfloat32_t test_svcvt_f32_u32_z(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f32_u32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ucvtf.nxv4f32.nxv4i32(<vscale x 4 x float> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_u32,_z,)(pg, op);
}

svfloat64_t test_svcvt_f64_u32_z(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_u32_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ucvtf.f64i32(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_u32,_z,)(pg, op);
}

svfloat16_t test_svcvt_f16_u32_m(svfloat16_t inactive, svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ucvtf.f16i32(<vscale x 8 x half> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_u32,_m,)(inactive, pg, op);
}

svfloat32_t test_svcvt_f32_u32_m(svfloat32_t inactive, svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f32_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ucvtf.nxv4f32.nxv4i32(<vscale x 4 x float> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_u32,_m,)(inactive, pg, op);
}

svfloat64_t test_svcvt_f64_u32_m(svfloat64_t inactive, svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ucvtf.f64i32(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_u32,_m,)(inactive, pg, op);
}

svfloat16_t test_svcvt_f16_u32_x(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ucvtf.f16i32(<vscale x 8 x half> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_u32,_x,)(pg, op);
}

svfloat32_t test_svcvt_f32_u32_x(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f32_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ucvtf.nxv4f32.nxv4i32(<vscale x 4 x float> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_u32,_x,)(pg, op);
}

svfloat64_t test_svcvt_f64_u32_x(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ucvtf.f64i32(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_u32,_x,)(pg, op);
}

svfloat16_t test_svcvt_f16_u64_z(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_u64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ucvtf.f16i64(<vscale x 8 x half> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_u64,_z,)(pg, op);
}

svfloat32_t test_svcvt_f32_u64_z(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_u64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ucvtf.f32i64(<vscale x 4 x float> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_u64,_z,)(pg, op);
}

svfloat64_t test_svcvt_f64_u64_z(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f64_u64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ucvtf.nxv2f64.nxv2i64(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_u64,_z,)(pg, op);
}

svfloat16_t test_svcvt_f16_u64_m(svfloat16_t inactive, svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_u64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ucvtf.f16i64(<vscale x 8 x half> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_u64,_m,)(inactive, pg, op);
}

svfloat32_t test_svcvt_f32_u64_m(svfloat32_t inactive, svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_u64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ucvtf.f32i64(<vscale x 4 x float> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_u64,_m,)(inactive, pg, op);
}

svfloat64_t test_svcvt_f64_u64_m(svfloat64_t inactive, svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f64_u64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ucvtf.nxv2f64.nxv2i64(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_u64,_m,)(inactive, pg, op);
}

svfloat16_t test_svcvt_f16_u64_x(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_u64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ucvtf.f16i64(<vscale x 8 x half> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_u64,_x,)(pg, op);
}

svfloat32_t test_svcvt_f32_u64_x(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_u64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ucvtf.f32i64(<vscale x 4 x float> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_u64,_x,)(pg, op);
}

svfloat64_t test_svcvt_f64_u64_x(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svcvt_f64_u64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ucvtf.nxv2f64.nxv2i64(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_u64,_x,)(pg, op);
}

svfloat32_t test_svcvt_f32_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_f32_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvt.f32f16(<vscale x 4 x float> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_f16,_z,)(pg, op);
}

svfloat64_t test_svcvt_f64_f16_z(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_f64_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcvt.f64f16(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_f16,_z,)(pg, op);
}

svfloat32_t test_svcvt_f32_f16_m(svfloat32_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_f32_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvt.f32f16(<vscale x 4 x float> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_f16,_m,)(inactive, pg, op);
}

svfloat64_t test_svcvt_f64_f16_m(svfloat64_t inactive, svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_f64_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcvt.f64f16(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_f16,_m,)(inactive, pg, op);
}

svfloat32_t test_svcvt_f32_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_f32_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvt.f32f16(<vscale x 4 x float> undef, <vscale x 4 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_f16,_x,)(pg, op);
}

svfloat64_t test_svcvt_f64_f16_x(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svcvt_f64_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcvt.f64f16(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_f16,_x,)(pg, op);
}

svfloat64_t test_svcvt_f64_f32_z(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_f32_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcvt.f64f32(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_f32,_z,)(pg, op);
}

svfloat64_t test_svcvt_f64_f32_m(svfloat64_t inactive, svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcvt.f64f32(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_f32,_m,)(inactive, pg, op);
}

svfloat64_t test_svcvt_f64_f32_x(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_f64_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fcvt.f64f32(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f64,_f32,_x,)(pg, op);
}

svfloat16_t test_svcvt_f16_f32_z(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_f32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcvt.f16f32(<vscale x 8 x half> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_f32,_z,)(pg, op);
}

svfloat16_t test_svcvt_f16_f64_z(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_f64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcvt.f16f64(<vscale x 8 x half> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_f64,_z,)(pg, op);
}

svfloat16_t test_svcvt_f16_f32_m(svfloat16_t inactive, svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcvt.f16f32(<vscale x 8 x half> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_f32,_m,)(inactive, pg, op);
}

svfloat16_t test_svcvt_f16_f64_m(svfloat16_t inactive, svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcvt.f16f64(<vscale x 8 x half> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_f64,_m,)(inactive, pg, op);
}

svfloat16_t test_svcvt_f16_f32_x(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svcvt_f16_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcvt.f16f32(<vscale x 8 x half> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_f32,_x,)(pg, op);
}

svfloat16_t test_svcvt_f16_f64_x(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_f16_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fcvt.f16f64(<vscale x 8 x half> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f16,_f64,_x,)(pg, op);
}

svfloat32_t test_svcvt_f32_f64_z(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_f64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvt.f32f64(<vscale x 4 x float> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_f64,_z,)(pg, op);
}

svfloat32_t test_svcvt_f32_f64_m(svfloat32_t inactive, svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvt.f32f64(<vscale x 4 x float> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_f64,_m,)(inactive, pg, op);
}

svfloat32_t test_svcvt_f32_f64_x(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svcvt_f32_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvt.f32f64(<vscale x 4 x float> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcvt_f32,_f64,_x,)(pg, op);
}
