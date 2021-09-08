// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

int64_t test_svaddv_s8(svbool_t pg, svint8_t op)
{
  // CHECK-LABEL: test_svaddv_s8
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.saddv.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_s8,,)(pg, op);
}

int64_t test_svaddv_s16(svbool_t pg, svint16_t op)
{
  // CHECK-LABEL: test_svaddv_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.saddv.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_s16,,)(pg, op);
}

int64_t test_svaddv_s32(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svaddv_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.saddv.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_s32,,)(pg, op);
}

int64_t test_svaddv_s64(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svaddv_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.saddv.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_s64,,)(pg, op);
}

uint64_t test_svaddv_u8(svbool_t pg, svuint8_t op)
{
  // CHECK-LABEL: test_svaddv_u8
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uaddv.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_u8,,)(pg, op);
}

uint64_t test_svaddv_u16(svbool_t pg, svuint16_t op)
{
  // CHECK-LABEL: test_svaddv_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uaddv.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_u16,,)(pg, op);
}

uint64_t test_svaddv_u32(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svaddv_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uaddv.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_u32,,)(pg, op);
}

uint64_t test_svaddv_u64(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svaddv_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uaddv.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_u64,,)(pg, op);
}

float16_t test_svaddv_f16(svbool_t pg, svfloat16_t op)
{
  // CHECK-LABEL: test_svaddv_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call half @llvm.aarch64.sve.faddv.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op)
  // CHECK: ret half %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_f16,,)(pg, op);
}

float32_t test_svaddv_f32(svbool_t pg, svfloat32_t op)
{
  // CHECK-LABEL: test_svaddv_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call float @llvm.aarch64.sve.faddv.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op)
  // CHECK: ret float %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_f32,,)(pg, op);
}

float64_t test_svaddv_f64(svbool_t pg, svfloat64_t op)
{
  // CHECK-LABEL: test_svaddv_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call double @llvm.aarch64.sve.faddv.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op)
  // CHECK: ret double %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svaddv,_f64,,)(pg, op);
}
