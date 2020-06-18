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

svint8_t test_svreinterpret_s8_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_s8
  // CHECK: ret <vscale x 16 x i8> %op
  return SVE_ACLE_FUNC(svreinterpret_s8,_s8,,)(op);
}

svint8_t test_svreinterpret_s8_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_s16,,)(op);
}

svint8_t test_svreinterpret_s8_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_s32,,)(op);
}

svint8_t test_svreinterpret_s8_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_s64,,)(op);
}

svint8_t test_svreinterpret_s8_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_u8
  // CHECK: ret <vscale x 16 x i8> %op
  return SVE_ACLE_FUNC(svreinterpret_s8,_u8,,)(op);
}

svint8_t test_svreinterpret_s8_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_u16,,)(op);
}

svint8_t test_svreinterpret_s8_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_u32,,)(op);
}

svint8_t test_svreinterpret_s8_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_u64,,)(op);
}

svint8_t test_svreinterpret_s8_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_f16,,)(op);
}

svint8_t test_svreinterpret_s8_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_f32,,)(op);
}

svint8_t test_svreinterpret_s8_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s8_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8,_f64,,)(op);
}

svint16_t test_svreinterpret_s16_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_s8,,)(op);
}

svint16_t test_svreinterpret_s16_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_s16
  // CHECK: ret <vscale x 8 x i16> %op
  return SVE_ACLE_FUNC(svreinterpret_s16,_s16,,)(op);
}

svint16_t test_svreinterpret_s16_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_s32,,)(op);
}

svint16_t test_svreinterpret_s16_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_s64,,)(op);
}

svint16_t test_svreinterpret_s16_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_u8,,)(op);
}

svint16_t test_svreinterpret_s16_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_u16
  // CHECK: ret <vscale x 8 x i16> %op
  return SVE_ACLE_FUNC(svreinterpret_s16,_u16,,)(op);
}

svint16_t test_svreinterpret_s16_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_u32,,)(op);
}

svint16_t test_svreinterpret_s16_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_u64,,)(op);
}

svint16_t test_svreinterpret_s16_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_f16,,)(op);
}

svint16_t test_svreinterpret_s16_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_f32,,)(op);
}

svint16_t test_svreinterpret_s16_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s16_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16,_f64,,)(op);
}

svint32_t test_svreinterpret_s32_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_s8,,)(op);
}

svint32_t test_svreinterpret_s32_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_s16,,)(op);
}

svint32_t test_svreinterpret_s32_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_s32
  // CHECK: ret <vscale x 4 x i32> %op
  return SVE_ACLE_FUNC(svreinterpret_s32,_s32,,)(op);
}

svint32_t test_svreinterpret_s32_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_s64,,)(op);
}

svint32_t test_svreinterpret_s32_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_u8,,)(op);
}

svint32_t test_svreinterpret_s32_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_u16,,)(op);
}

svint32_t test_svreinterpret_s32_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_u32
  // CHECK: ret <vscale x 4 x i32> %op
  return SVE_ACLE_FUNC(svreinterpret_s32,_u32,,)(op);
}

svint32_t test_svreinterpret_s32_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_u64,,)(op);
}

svint32_t test_svreinterpret_s32_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_f16,,)(op);
}

svint32_t test_svreinterpret_s32_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_f32,,)(op);
}

svint32_t test_svreinterpret_s32_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s32_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32,_f64,,)(op);
}

svint64_t test_svreinterpret_s64_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_s8,,)(op);
}

svint64_t test_svreinterpret_s64_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_s16,,)(op);
}

svint64_t test_svreinterpret_s64_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_s32,,)(op);
}

svint64_t test_svreinterpret_s64_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_s64
  // CHECK: ret <vscale x 2 x i64> %op
  return SVE_ACLE_FUNC(svreinterpret_s64,_s64,,)(op);
}

svint64_t test_svreinterpret_s64_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_u8,,)(op);
}

svint64_t test_svreinterpret_s64_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_u16,,)(op);
}

svint64_t test_svreinterpret_s64_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_u32,,)(op);
}

svint64_t test_svreinterpret_s64_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_u64
  // CHECK: ret <vscale x 2 x i64> %op
  return SVE_ACLE_FUNC(svreinterpret_s64,_u64,,)(op);
}

svint64_t test_svreinterpret_s64_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_f16,,)(op);
}

svint64_t test_svreinterpret_s64_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_f32,,)(op);
}

svint64_t test_svreinterpret_s64_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_s64_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64,_f64,,)(op);
}

svuint8_t test_svreinterpret_u8_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_s8
  // CHECK: ret <vscale x 16 x i8> %op
  return SVE_ACLE_FUNC(svreinterpret_u8,_s8,,)(op);
}

svuint8_t test_svreinterpret_u8_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_s16,,)(op);
}

svuint8_t test_svreinterpret_u8_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_s32,,)(op);
}

svuint8_t test_svreinterpret_u8_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_s64,,)(op);
}

svuint8_t test_svreinterpret_u8_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_u8
  // CHECK: ret <vscale x 16 x i8> %op
  return SVE_ACLE_FUNC(svreinterpret_u8,_u8,,)(op);
}

svuint8_t test_svreinterpret_u8_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_u16,,)(op);
}

svuint8_t test_svreinterpret_u8_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_u32,,)(op);
}

svuint8_t test_svreinterpret_u8_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_u64,,)(op);
}

svuint8_t test_svreinterpret_u8_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_f16,,)(op);
}

svuint8_t test_svreinterpret_u8_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_f32,,)(op);
}

svuint8_t test_svreinterpret_u8_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u8_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8,_f64,,)(op);
}

svuint16_t test_svreinterpret_u16_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_s8,,)(op);
}

svuint16_t test_svreinterpret_u16_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_s16
  // CHECK: ret <vscale x 8 x i16> %op
  return SVE_ACLE_FUNC(svreinterpret_u16,_s16,,)(op);
}

svuint16_t test_svreinterpret_u16_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_s32,,)(op);
}

svuint16_t test_svreinterpret_u16_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_s64,,)(op);
}

svuint16_t test_svreinterpret_u16_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_u8,,)(op);
}

svuint16_t test_svreinterpret_u16_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_u16
  // CHECK: ret <vscale x 8 x i16> %op
  return SVE_ACLE_FUNC(svreinterpret_u16,_u16,,)(op);
}

svuint16_t test_svreinterpret_u16_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_u32,,)(op);
}

svuint16_t test_svreinterpret_u16_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_u64,,)(op);
}

svuint16_t test_svreinterpret_u16_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_f16,,)(op);
}

svuint16_t test_svreinterpret_u16_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_f32,,)(op);
}

svuint16_t test_svreinterpret_u16_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u16_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16,_f64,,)(op);
}

svuint32_t test_svreinterpret_u32_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_s8,,)(op);
}

svuint32_t test_svreinterpret_u32_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_s16,,)(op);
}

svuint32_t test_svreinterpret_u32_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_s32
  // CHECK: ret <vscale x 4 x i32> %op
  return SVE_ACLE_FUNC(svreinterpret_u32,_s32,,)(op);
}

svuint32_t test_svreinterpret_u32_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_s64,,)(op);
}

svuint32_t test_svreinterpret_u32_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_u8,,)(op);
}

svuint32_t test_svreinterpret_u32_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_u16,,)(op);
}

svuint32_t test_svreinterpret_u32_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_u32
  // CHECK: ret <vscale x 4 x i32> %op
  return SVE_ACLE_FUNC(svreinterpret_u32,_u32,,)(op);
}

svuint32_t test_svreinterpret_u32_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_u64,,)(op);
}

svuint32_t test_svreinterpret_u32_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_f16,,)(op);
}

svuint32_t test_svreinterpret_u32_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_f32,,)(op);
}

svuint32_t test_svreinterpret_u32_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u32_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32,_f64,,)(op);
}

svuint64_t test_svreinterpret_u64_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_s8,,)(op);
}

svuint64_t test_svreinterpret_u64_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_s16,,)(op);
}

svuint64_t test_svreinterpret_u64_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_s32,,)(op);
}

svuint64_t test_svreinterpret_u64_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_s64
  // CHECK: ret <vscale x 2 x i64> %op
  return SVE_ACLE_FUNC(svreinterpret_u64,_s64,,)(op);
}

svuint64_t test_svreinterpret_u64_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_u8,,)(op);
}

svuint64_t test_svreinterpret_u64_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_u16,,)(op);
}

svuint64_t test_svreinterpret_u64_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_u32,,)(op);
}

svuint64_t test_svreinterpret_u64_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_u64
  // CHECK: ret <vscale x 2 x i64> %op
  return SVE_ACLE_FUNC(svreinterpret_u64,_u64,,)(op);
}

svuint64_t test_svreinterpret_u64_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_f16,,)(op);
}

svuint64_t test_svreinterpret_u64_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_f32,,)(op);
}

svuint64_t test_svreinterpret_u64_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_u64_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64,_f64,,)(op);
}

svfloat16_t test_svreinterpret_f16_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_s8,,)(op);
}

svfloat16_t test_svreinterpret_f16_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_s16,,)(op);
}

svfloat16_t test_svreinterpret_f16_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_s32,,)(op);
}

svfloat16_t test_svreinterpret_f16_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_s64,,)(op);
}

svfloat16_t test_svreinterpret_f16_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_u8,,)(op);
}

svfloat16_t test_svreinterpret_f16_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_u16,,)(op);
}

svfloat16_t test_svreinterpret_f16_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_u32,,)(op);
}

svfloat16_t test_svreinterpret_f16_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_u64,,)(op);
}

svfloat16_t test_svreinterpret_f16_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_f16
  // CHECK: ret <vscale x 8 x half> %op
  return SVE_ACLE_FUNC(svreinterpret_f16,_f16,,)(op);
}

svfloat16_t test_svreinterpret_f16_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_f32,,)(op);
}

svfloat16_t test_svreinterpret_f16_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f16_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16,_f64,,)(op);
}

svfloat32_t test_svreinterpret_f32_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_s8,,)(op);
}

svfloat32_t test_svreinterpret_f32_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_s16,,)(op);
}

svfloat32_t test_svreinterpret_f32_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_s32,,)(op);
}

svfloat32_t test_svreinterpret_f32_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_s64,,)(op);
}

svfloat32_t test_svreinterpret_f32_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_u8,,)(op);
}

svfloat32_t test_svreinterpret_f32_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_u16,,)(op);
}

svfloat32_t test_svreinterpret_f32_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_u32,,)(op);
}

svfloat32_t test_svreinterpret_f32_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_u64,,)(op);
}

svfloat32_t test_svreinterpret_f32_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_f16,,)(op);
}

svfloat32_t test_svreinterpret_f32_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_f32
  // CHECK: ret <vscale x 4 x float> %op
  return SVE_ACLE_FUNC(svreinterpret_f32,_f32,,)(op);
}

svfloat32_t test_svreinterpret_f32_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f32_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32,_f64,,)(op);
}

svfloat64_t test_svreinterpret_f64_s8(svint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_s8,,)(op);
}

svfloat64_t test_svreinterpret_f64_s16(svint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_s16,,)(op);
}

svfloat64_t test_svreinterpret_f64_s32(svint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_s32,,)(op);
}

svfloat64_t test_svreinterpret_f64_s64(svint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_s64,,)(op);
}

svfloat64_t test_svreinterpret_f64_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_u8,,)(op);
}

svfloat64_t test_svreinterpret_f64_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_u16,,)(op);
}

svfloat64_t test_svreinterpret_f64_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_u32,,)(op);
}

svfloat64_t test_svreinterpret_f64_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_u64,,)(op);
}

svfloat64_t test_svreinterpret_f64_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_f16,,)(op);
}

svfloat64_t test_svreinterpret_f64_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64,_f32,,)(op);
}

svfloat64_t test_svreinterpret_f64_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svreinterpret_f64_f64
  // CHECK: ret <vscale x 2 x double> %op
  return SVE_ACLE_FUNC(svreinterpret_f64,_f64,,)(op);
}
