// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8x2_t test_svcreate2_s8(svint8_t x0, svint8_t x1)
{
  // CHECK-LABEL: test_svcreate2_s8
  // CHECK: %[[CREATE:.*]] = call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8> %x0, <vscale x 16 x i8> %x1)
  // CHECK-NEXT: ret <vscale x 32 x i8> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_s8,,)(x0, x1);
}

svint16x2_t test_svcreate2_s16(svint16_t x0, svint16_t x1)
{
  // CHECK-LABEL: test_svcreate2_s16
  // CHECK: %[[CREATE:.*]] = call <vscale x 16 x i16> @llvm.aarch64.sve.tuple.create2.nxv16i16.nxv8i16(<vscale x 8 x i16> %x0, <vscale x 8 x i16> %x1)
  // CHECK-NEXT: ret <vscale x 16 x i16> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_s16,,)(x0, x1);
}

svint32x2_t test_svcreate2_s32(svint32_t x0, svint32_t x1)
{
  // CHECK-LABEL: test_svcreate2_s32
  // CHECK: %[[CREATE:.*]] = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %x0, <vscale x 4 x i32> %x1)
  // CHECK-NEXT: ret <vscale x 8 x i32> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_s32,,)(x0, x1);
}

svint64x2_t test_svcreate2_s64(svint64_t x0, svint64_t x1)
{
  // CHECK-LABEL: test_svcreate2_s64
  // CHECK: %[[CREATE:.*]] = call <vscale x 4 x i64> @llvm.aarch64.sve.tuple.create2.nxv4i64.nxv2i64(<vscale x 2 x i64> %x0, <vscale x 2 x i64> %x1)
  // CHECK-NEXT: ret <vscale x 4 x i64> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_s64,,)(x0, x1);
}

svuint8x2_t test_svcreate2_u8(svuint8_t x0, svuint8_t x1)
{
  // CHECK-LABEL: test_svcreate2_u8
  // CHECK: %[[CREATE:.*]] = call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8> %x0, <vscale x 16 x i8> %x1)
  // CHECK-NEXT: ret <vscale x 32 x i8> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_u8,,)(x0, x1);
}

svuint16x2_t test_svcreate2_u16(svuint16_t x0, svuint16_t x1)
{
  // CHECK-LABEL: test_svcreate2_u16
  // CHECK: %[[CREATE:.*]] = call <vscale x 16 x i16> @llvm.aarch64.sve.tuple.create2.nxv16i16.nxv8i16(<vscale x 8 x i16> %x0, <vscale x 8 x i16> %x1)
  // CHECK-NEXT: ret <vscale x 16 x i16> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_u16,,)(x0, x1);
}

svuint32x2_t test_svcreate2_u32(svuint32_t x0, svuint32_t x1)
{
  // CHECK-LABEL: test_svcreate2_u32
  // CHECK: %[[CREATE:.*]] = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %x0, <vscale x 4 x i32> %x1)
  // CHECK-NEXT: ret <vscale x 8 x i32> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_u32,,)(x0, x1);
}

svuint64x2_t test_svcreate2_u64(svuint64_t x0, svuint64_t x1)
{
  // CHECK-LABEL: test_svcreate2_u64
  // CHECK: %[[CREATE:.*]] = call <vscale x 4 x i64> @llvm.aarch64.sve.tuple.create2.nxv4i64.nxv2i64(<vscale x 2 x i64> %x0, <vscale x 2 x i64> %x1)
  // CHECK-NEXT: ret <vscale x 4 x i64> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_u64,,)(x0, x1);
}

svfloat16x2_t test_svcreate2_f16(svfloat16_t x0, svfloat16_t x1)
{
  // CHECK-LABEL: test_svcreate2_f16
  // CHECK: %[[CREATE:.*]] = call <vscale x 16 x half> @llvm.aarch64.sve.tuple.create2.nxv16f16.nxv8f16(<vscale x 8 x half> %x0, <vscale x 8 x half> %x1)
  // CHECK-NEXT: ret <vscale x 16 x half> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_f16,,)(x0, x1);
}

svfloat32x2_t test_svcreate2_f32(svfloat32_t x0, svfloat32_t x1)
{
  // CHECK-LABEL: test_svcreate2_f32
  // CHECK: %[[CREATE:.*]] = call <vscale x 8 x float> @llvm.aarch64.sve.tuple.create2.nxv8f32.nxv4f32(<vscale x 4 x float> %x0, <vscale x 4 x float> %x1)
  // CHECK-NEXT: ret <vscale x 8 x float> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_f32,,)(x0, x1);
}

svfloat64x2_t test_svcreate2_f64(svfloat64_t x0, svfloat64_t x1)
{
  // CHECK-LABEL: test_svcreate2_f64
  // CHECK: %[[CREATE:.*]] = call <vscale x 4 x double> @llvm.aarch64.sve.tuple.create2.nxv4f64.nxv2f64(<vscale x 2 x double> %x0, <vscale x 2 x double> %x1)
  // CHECK-NEXT: ret <vscale x 4 x double> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate2,_f64,,)(x0, x1);
}
