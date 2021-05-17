// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8x3_t test_svcreate3_s8(svint8_t x0, svint8_t x1, svint8_t x2)
{
  // CHECK-LABEL: test_svcreate3_s8
  // CHECK: %[[CREATE:.*]] = call <vscale x 48 x i8> @llvm.aarch64.sve.tuple.create3.nxv48i8.nxv16i8(<vscale x 16 x i8> %x0, <vscale x 16 x i8> %x1, <vscale x 16 x i8> %x2)
  // CHECK-NEXT: ret <vscale x 48 x i8> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_s8,,)(x0, x1, x2);
}

svint16x3_t test_svcreate3_s16(svint16_t x0, svint16_t x1, svint16_t x2)
{
  // CHECK-LABEL: test_svcreate3_s16
  // CHECK: %[[CREATE:.*]] = call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16> %x0, <vscale x 8 x i16> %x1, <vscale x 8 x i16> %x2)
  // CHECK-NEXT: ret <vscale x 24 x i16> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_s16,,)(x0, x1, x2);
}

svint32x3_t test_svcreate3_s32(svint32_t x0, svint32_t x1, svint32_t x2)
{
  // CHECK-LABEL: test_svcreate3_s32
  // CHECK: %[[CREATE:.*]] = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %x0, <vscale x 4 x i32> %x1, <vscale x 4 x i32> %x2)
  // CHECK-NEXT: ret <vscale x 12 x i32> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_s32,,)(x0, x1, x2);
}

svint64x3_t test_svcreate3_s64(svint64_t x0, svint64_t x1, svint64_t x2)
{
  // CHECK-LABEL: test_svcreate3_s64
  // CHECK: %[[CREATE:.*]] = call <vscale x 6 x i64> @llvm.aarch64.sve.tuple.create3.nxv6i64.nxv2i64(<vscale x 2 x i64> %x0, <vscale x 2 x i64> %x1, <vscale x 2 x i64> %x2)
  // CHECK-NEXT: ret <vscale x 6 x i64> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_s64,,)(x0, x1, x2);
}

svuint8x3_t test_svcreate3_u8(svuint8_t x0, svuint8_t x1, svuint8_t x2)
{
  // CHECK-LABEL: test_svcreate3_u8
  // CHECK: %[[CREATE:.*]] = call <vscale x 48 x i8> @llvm.aarch64.sve.tuple.create3.nxv48i8.nxv16i8(<vscale x 16 x i8> %x0, <vscale x 16 x i8> %x1, <vscale x 16 x i8> %x2)
  // CHECK-NEXT: ret <vscale x 48 x i8> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_u8,,)(x0, x1, x2);
}

svuint16x3_t test_svcreate3_u16(svuint16_t x0, svuint16_t x1, svuint16_t x2)
{
  // CHECK-LABEL: test_svcreate3_u16
  // CHECK: %[[CREATE:.*]] = call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16> %x0, <vscale x 8 x i16> %x1, <vscale x 8 x i16> %x2)
  // CHECK-NEXT: ret <vscale x 24 x i16> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_u16,,)(x0, x1, x2);
}

svuint32x3_t test_svcreate3_u32(svuint32_t x0, svuint32_t x1, svuint32_t x2)
{
  // CHECK-LABEL: test_svcreate3_u32
  // CHECK: %[[CREATE:.*]] = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %x0, <vscale x 4 x i32> %x1, <vscale x 4 x i32> %x2)
  // CHECK-NEXT: ret <vscale x 12 x i32> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_u32,,)(x0, x1, x2);
}

svuint64x3_t test_svcreate3_u64(svuint64_t x0, svuint64_t x1, svuint64_t x2)
{
  // CHECK-LABEL: test_svcreate3_u64
  // CHECK: %[[CREATE:.*]] = call <vscale x 6 x i64> @llvm.aarch64.sve.tuple.create3.nxv6i64.nxv2i64(<vscale x 2 x i64> %x0, <vscale x 2 x i64> %x1, <vscale x 2 x i64> %x2)
  // CHECK-NEXT: ret <vscale x 6 x i64> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_u64,,)(x0, x1, x2);
}

svfloat16x3_t test_svcreate3_f16(svfloat16_t x0, svfloat16_t x1, svfloat16_t x2)
{
  // CHECK-LABEL: test_svcreate3_f16
  // CHECK: %[[CREATE:.*]] = call <vscale x 24 x half> @llvm.aarch64.sve.tuple.create3.nxv24f16.nxv8f16(<vscale x 8 x half> %x0, <vscale x 8 x half> %x1, <vscale x 8 x half> %x2)
  // CHECK-NEXT: ret <vscale x 24 x half> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_f16,,)(x0, x1, x2);
}

svfloat32x3_t test_svcreate3_f32(svfloat32_t x0, svfloat32_t x1, svfloat32_t x2)
{
  // CHECK-LABEL: test_svcreate3_f32
  // CHECK: %[[CREATE:.*]] = call <vscale x 12 x float> @llvm.aarch64.sve.tuple.create3.nxv12f32.nxv4f32(<vscale x 4 x float> %x0, <vscale x 4 x float> %x1, <vscale x 4 x float> %x2)
  // CHECK-NEXT: ret <vscale x 12 x float> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_f32,,)(x0, x1, x2);
}

svfloat64x3_t test_svcreate3_f64(svfloat64_t x0, svfloat64_t x1, svfloat64_t x2)
{
  // CHECK-LABEL: test_svcreate3_f64
  // CHECK: %[[CREATE:.*]] = call <vscale x 6 x double> @llvm.aarch64.sve.tuple.create3.nxv6f64.nxv2f64(<vscale x 2 x double> %x0, <vscale x 2 x double> %x1, <vscale x 2 x double> %x2)
  // CHECK-NEXT: ret <vscale x 6 x double> %[[CREATE]]
  return SVE_ACLE_FUNC(svcreate3,_f64,,)(x0, x1, x2);
}
