// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svdup_n_s8(int8_t op)
{
  // CHECK-LABEL: test_svdup_n_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s8,)(op);
}

svint16_t test_svdup_n_s16(int16_t op)
{
  // CHECK-LABEL: test_svdup_n_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s16,)(op);
}

svint32_t test_svdup_n_s32(int32_t op)
{
  // CHECK-LABEL: test_svdup_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s32,)(op);
}

svint64_t test_svdup_n_s64(int64_t op)
{
  // CHECK-LABEL: test_svdup_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s64,)(op);
}

svuint8_t test_svdup_n_u8(uint8_t op)
{
  // CHECK-LABEL: test_svdup_n_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u8,)(op);
}

svuint16_t test_svdup_n_u16(uint16_t op)
{
  // CHECK-LABEL: test_svdup_n_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u16,)(op);
}

svuint32_t test_svdup_n_u32(uint32_t op)
{
  // CHECK-LABEL: test_svdup_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u32,)(op);
}

svuint64_t test_svdup_n_u64(uint64_t op)
{
  // CHECK-LABEL: test_svdup_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u64,)(op);
}

svfloat16_t test_svdup_n_f16(float16_t op)
{
  // CHECK-LABEL: test_svdup_n_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f16,)(op);
}

svfloat32_t test_svdup_n_f32(float32_t op)
{
  // CHECK-LABEL: test_svdup_n_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f32,)(op);
}

svfloat64_t test_svdup_n_f64(float64_t op)
{
  // CHECK-LABEL: test_svdup_n_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f64,)(op);
}

svint8_t test_svdup_n_s8_z(svbool_t pg, int8_t op)
{
  // CHECK-LABEL: test_svdup_n_s8_z
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> zeroinitializer, <vscale x 16 x i1> %pg, i8 %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s8_z,)(pg, op);
}

svint16_t test_svdup_n_s16_z(svbool_t pg, int16_t op)
{
  // CHECK-LABEL: test_svdup_n_s16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> %[[PG]], i16 %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s16_z,)(pg, op);
}

svint32_t test_svdup_n_s32_z(svbool_t pg, int32_t op)
{
  // CHECK-LABEL: test_svdup_n_s32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], i32 %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s32_z,)(pg, op);
}

svint64_t test_svdup_n_s64_z(svbool_t pg, int64_t op)
{
  // CHECK-LABEL: test_svdup_n_s64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], i64 %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s64_z,)(pg, op);
}

svuint8_t test_svdup_n_u8_z(svbool_t pg, uint8_t op)
{
  // CHECK-LABEL: test_svdup_n_u8_z
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> zeroinitializer, <vscale x 16 x i1> %pg, i8 %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u8_z,)(pg, op);
}

svuint16_t test_svdup_n_u16_z(svbool_t pg, uint16_t op)
{
  // CHECK-LABEL: test_svdup_n_u16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> %[[PG]], i16 %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u16_z,)(pg, op);
}

svuint32_t test_svdup_n_u32_z(svbool_t pg, uint32_t op)
{
  // CHECK-LABEL: test_svdup_n_u32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], i32 %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u32_z,)(pg, op);
}

svuint64_t test_svdup_n_u64_z(svbool_t pg, uint64_t op)
{
  // CHECK-LABEL: test_svdup_n_u64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], i64 %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u64_z,)(pg, op);
}

svfloat16_t test_svdup_n_f16_z(svbool_t pg, float16_t op)
{
  // CHECK-LABEL: test_svdup_n_f16_z
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half> zeroinitializer, <vscale x 8 x i1> %[[PG]], half %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f16_z,)(pg, op);
}

svfloat32_t test_svdup_n_f32_z(svbool_t pg, float32_t op)
{
  // CHECK-LABEL: test_svdup_n_f32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float> zeroinitializer, <vscale x 4 x i1> %[[PG]], float %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f32_z,)(pg, op);
}

svfloat64_t test_svdup_n_f64_z(svbool_t pg, float64_t op)
{
  // CHECK-LABEL: test_svdup_n_f64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> %[[PG]], double %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f64_z,)(pg, op);
}

svint8_t test_svdup_n_s8_m(svint8_t inactive, svbool_t pg, int8_t op)
{
  // CHECK-LABEL: test_svdup_n_s8_m
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %inactive, <vscale x 16 x i1> %pg, i8 %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s8_m,)(inactive, pg, op);
}

svint16_t test_svdup_n_s16_m(svint16_t inactive, svbool_t pg, int16_t op)
{
  // CHECK-LABEL: test_svdup_n_s16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> %inactive, <vscale x 8 x i1> %[[PG]], i16 %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s16_m,)(inactive, pg, op);
}

svint32_t test_svdup_n_s32_m(svint32_t inactive, svbool_t pg, int32_t op)
{
  // CHECK-LABEL: test_svdup_n_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], i32 %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s32_m,)(inactive, pg, op);
}

svint64_t test_svdup_n_s64_m(svint64_t inactive, svbool_t pg, int64_t op)
{
  // CHECK-LABEL: test_svdup_n_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], i64 %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s64_m,)(inactive, pg, op);
}

svuint8_t test_svdup_n_u8_m(svuint8_t inactive, svbool_t pg, uint8_t op)
{
  // CHECK-LABEL: test_svdup_n_u8_m
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %inactive, <vscale x 16 x i1> %pg, i8 %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u8_m,)(inactive, pg, op);
}

svuint16_t test_svdup_n_u16_m(svuint16_t inactive, svbool_t pg, uint16_t op)
{
  // CHECK-LABEL: test_svdup_n_u16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> %inactive, <vscale x 8 x i1> %[[PG]], i16 %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u16_m,)(inactive, pg, op);
}

svuint32_t test_svdup_n_u32_m(svuint32_t inactive, svbool_t pg, uint32_t op)
{
  // CHECK-LABEL: test_svdup_n_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], i32 %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u32_m,)(inactive, pg, op);
}

svuint64_t test_svdup_n_u64_m(svuint64_t inactive, svbool_t pg, uint64_t op)
{
  // CHECK-LABEL: test_svdup_n_u64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], i64 %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u64_m,)(inactive, pg, op);
}

svfloat16_t test_svdup_n_f16_m(svfloat16_t inactive, svbool_t pg, float16_t op)
{
  // CHECK-LABEL: test_svdup_n_f16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half> %inactive, <vscale x 8 x i1> %[[PG]], half %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f16_m,)(inactive, pg, op);
}

svfloat32_t test_svdup_n_f32_m(svfloat32_t inactive, svbool_t pg, float32_t op)
{
  // CHECK-LABEL: test_svdup_n_f32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float> %inactive, <vscale x 4 x i1> %[[PG]], float %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f32_m,)(inactive, pg, op);
}

svfloat64_t test_svdup_n_f64_m(svfloat64_t inactive, svbool_t pg, float64_t op)
{
  // CHECK-LABEL: test_svdup_n_f64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> %inactive, <vscale x 2 x i1> %[[PG]], double %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f64_m,)(inactive, pg, op);
}

svint8_t test_svdup_n_s8_x(svbool_t pg, int8_t op)
{
  // CHECK-LABEL: test_svdup_n_s8_x
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> undef, <vscale x 16 x i1> %pg, i8 %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s8_x,)(pg, op);
}

svint16_t test_svdup_n_s16_x(svbool_t pg, int16_t op)
{
  // CHECK-LABEL: test_svdup_n_s16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> undef, <vscale x 8 x i1> %[[PG]], i16 %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s16_x,)(pg, op);
}

svint32_t test_svdup_n_s32_x(svbool_t pg, int32_t op)
{
  // CHECK-LABEL: test_svdup_n_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], i32 %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s32_x,)(pg, op);
}

svint64_t test_svdup_n_s64_x(svbool_t pg, int64_t op)
{
  // CHECK-LABEL: test_svdup_n_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], i64 %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_s64_x,)(pg, op);
}

svuint8_t test_svdup_n_u8_x(svbool_t pg, uint8_t op)
{
  // CHECK-LABEL: test_svdup_n_u8_x
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> undef, <vscale x 16 x i1> %pg, i8 %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u8_x,)(pg, op);
}

svuint16_t test_svdup_n_u16_x(svbool_t pg, uint16_t op)
{
  // CHECK-LABEL: test_svdup_n_u16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> undef, <vscale x 8 x i1> %[[PG]], i16 %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u16_x,)(pg, op);
}

svuint32_t test_svdup_n_u32_x(svbool_t pg, uint32_t op)
{
  // CHECK-LABEL: test_svdup_n_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], i32 %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u32_x,)(pg, op);
}

svuint64_t test_svdup_n_u64_x(svbool_t pg, uint64_t op)
{
  // CHECK-LABEL: test_svdup_n_u64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], i64 %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_u64_x,)(pg, op);
}

svfloat16_t test_svdup_n_f16_x(svbool_t pg, float16_t op)
{
  // CHECK-LABEL: test_svdup_n_f16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half> undef, <vscale x 8 x i1> %[[PG]], half %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f16_x,)(pg, op);
}

svfloat32_t test_svdup_n_f32_x(svbool_t pg, float32_t op)
{
  // CHECK-LABEL: test_svdup_n_f32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float> undef, <vscale x 4 x i1> %[[PG]], float %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f32_x,)(pg, op);
}

svfloat64_t test_svdup_n_f64_x(svbool_t pg, float64_t op)
{
  // CHECK-LABEL: test_svdup_n_f64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> undef, <vscale x 2 x i1> %[[PG]], double %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup,_n,_f64_x,)(pg, op);
}

svint8_t test_svdup_lane_s8(svint8_t data, uint8_t index)
{
  // CHECK-LABEL: test_svdup_lane_s8
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbl.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_s8,,)(data, index);
}

svint16_t test_svdup_lane_s16(svint16_t data, uint16_t index)
{
  // CHECK-LABEL: test_svdup_lane_s16
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_s16,,)(data, index);
}

svint32_t test_svdup_lane_s32(svint32_t data, uint32_t index)
{
  // CHECK-LABEL: test_svdup_lane_s32
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tbl.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_s32,,)(data, index);
}

svint64_t test_svdup_lane_s64(svint64_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdup_lane_s64
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tbl.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_s64,,)(data, index);
}

svuint8_t test_svdup_lane_u8(svuint8_t data, uint8_t index)
{
  // CHECK-LABEL: test_svdup_lane_u8
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbl.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_u8,,)(data, index);
}

svuint16_t test_svdup_lane_u16(svuint16_t data, uint16_t index)
{
  // CHECK-LABEL: test_svdup_lane_u16
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_u16,,)(data, index);
}

svuint32_t test_svdup_lane_u32(svuint32_t data, uint32_t index)
{
  // CHECK-LABEL: test_svdup_lane_u32
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tbl.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_u32,,)(data, index);
}

svuint64_t test_svdup_lane_u64(svuint64_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdup_lane_u64
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tbl.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_u64,,)(data, index);
}

svfloat16_t test_svdup_lane_f16(svfloat16_t data, uint16_t index)
{
  // CHECK-LABEL: test_svdup_lane_f16
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.tbl.nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_f16,,)(data, index);
}

svfloat32_t test_svdup_lane_f32(svfloat32_t data, uint32_t index)
{
  // CHECK-LABEL: test_svdup_lane_f32
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.tbl.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_f32,,)(data, index);
}

svfloat64_t test_svdup_lane_f64(svfloat64_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdup_lane_f64
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %index)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.tbl.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdup_lane,_f64,,)(data, index);
}

svbool_t test_svdup_n_b8(bool op)
{
  // CHECK-LABEL: test_svdup_n_b8
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.dup.x.nxv16i1(i1 %op)
  // CHECK: ret <vscale x 16 x i1> %[[DUP]]
  return SVE_ACLE_FUNC(svdup,_n,_b8,)(op);
}

svbool_t test_svdup_n_b16(bool op)
{
  // CHECK-LABEL: test_svdup_n_b16
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.dup.x.nxv8i1(i1 %op)
  // CHECK: %[[CVT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[DUP]])
  // CHECK: ret <vscale x 16 x i1> %[[CVT]]
  return SVE_ACLE_FUNC(svdup,_n,_b16,)(op);
}

svbool_t test_svdup_n_b32(bool op)
{
  // CHECK-LABEL: test_svdup_n_b32
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.dup.x.nxv4i1(i1 %op)
  // CHECK: %[[CVT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[DUP]])
  // CHECK: ret <vscale x 16 x i1> %[[CVT]]
  return SVE_ACLE_FUNC(svdup,_n,_b32,)(op);
}

svbool_t test_svdup_n_b64(bool op)
{
  // CHECK-LABEL: test_svdup_n_b64
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.dup.x.nxv2i1(i1 %op)
  // CHECK: %[[CVT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[DUP]])
  // CHECK: ret <vscale x 16 x i1> %[[CVT]]
  return SVE_ACLE_FUNC(svdup,_n,_b64,)(op);
}
