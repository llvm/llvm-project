// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif


// NOTE: For these tests clang converts the struct parameter into
// several parameters, one for each member of the original struct.
svint8x3_t test_svset3_s8(svint8x3_t tuple, svint8_t x)
{
  // CHECK-LABEL: test_svset3_s8
  // CHECK: %[[INSERT:.*]] = call <vscale x 48 x i8> @llvm.aarch64.sve.tuple.set.nxv48i8.nxv16i8(<vscale x 48 x i8> %tuple, i32 1, <vscale x 16 x i8> %x)
  // CHECK-NEXT: ret <vscale x 48 x i8> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_s8,,)(tuple, 1, x);
}

svint16x3_t test_svset3_s16(svint16x3_t tuple, svint16_t x)
{
  // CHECK-LABEL: test_svset3_s16
  // CHECK: %[[INSERT:.*]] = call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.set.nxv24i16.nxv8i16(<vscale x 24 x i16> %tuple, i32 2, <vscale x 8 x i16> %x)
  // CHECK-NEXT: ret <vscale x 24 x i16> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_s16,,)(tuple, 2, x);
}

svint32x3_t test_svset3_s32(svint32x3_t tuple, svint32_t x)
{
  // CHECK-LABEL: test_svset3_s32
  // CHECK: %[[INSERT:.*]] = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.set.nxv12i32.nxv4i32(<vscale x 12 x i32> %tuple, i32 0, <vscale x 4 x i32> %x)
  // CHECK-NEXT: ret <vscale x 12 x i32> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_s32,,)(tuple, 0, x);
}

svint64x3_t test_svset3_s64(svint64x3_t tuple, svint64_t x)
{
  // CHECK-LABEL: test_svset3_s64
  // CHECK: %[[INSERT:.*]] = call <vscale x 6 x i64> @llvm.aarch64.sve.tuple.set.nxv6i64.nxv2i64(<vscale x 6 x i64> %tuple, i32 1, <vscale x 2 x i64> %x)
  // CHECK-NEXT: ret <vscale x 6 x i64> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_s64,,)(tuple, 1, x);
}

svuint8x3_t test_svset3_u8(svuint8x3_t tuple, svuint8_t x)
{
  // CHECK-LABEL: test_svset3_u8
  // CHECK: %[[INSERT:.*]] = call <vscale x 48 x i8> @llvm.aarch64.sve.tuple.set.nxv48i8.nxv16i8(<vscale x 48 x i8> %tuple, i32 2, <vscale x 16 x i8> %x)
  // CHECK-NEXT: ret <vscale x 48 x i8> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_u8,,)(tuple, 2, x);
}

svuint16x3_t test_svset3_u16(svuint16x3_t tuple, svuint16_t x)
{
  // CHECK-LABEL: test_svset3_u16
  // CHECK: %[[INSERT:.*]] = call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.set.nxv24i16.nxv8i16(<vscale x 24 x i16> %tuple, i32 0, <vscale x 8 x i16> %x)
  // CHECK-NEXT: ret <vscale x 24 x i16> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_u16,,)(tuple, 0, x);
}

svuint32x3_t test_svset3_u32(svuint32x3_t tuple, svuint32_t x)
{
  // CHECK-LABEL: test_svset3_u32
  // CHECK: %[[INSERT:.*]] = call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.set.nxv12i32.nxv4i32(<vscale x 12 x i32> %tuple, i32 1, <vscale x 4 x i32> %x)
  // CHECK-NEXT: ret <vscale x 12 x i32> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_u32,,)(tuple, 1, x);
}

svuint64x3_t test_svset3_u64(svuint64x3_t tuple, svuint64_t x)
{
  // CHECK-LABEL: test_svset3_u64
  // CHECK: %[[INSERT:.*]] = call <vscale x 6 x i64> @llvm.aarch64.sve.tuple.set.nxv6i64.nxv2i64(<vscale x 6 x i64> %tuple, i32 2, <vscale x 2 x i64> %x)
  // CHECK-NEXT: ret <vscale x 6 x i64> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_u64,,)(tuple, 2, x);
}

svfloat16x3_t test_svset3_f16(svfloat16x3_t tuple, svfloat16_t x)
{
  // CHECK-LABEL: test_svset3_f16
  // CHECK: %[[INSERT:.*]] = call <vscale x 24 x half> @llvm.aarch64.sve.tuple.set.nxv24f16.nxv8f16(<vscale x 24 x half> %tuple, i32 0, <vscale x 8 x half> %x)
  // CHECK-NEXT: ret <vscale x 24 x half> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_f16,,)(tuple, 0, x);
}

svfloat32x3_t test_svset3_f32(svfloat32x3_t tuple, svfloat32_t x)
{
  // CHECK-LABEL: test_svset3_f32
  // CHECK: %[[INSERT:.*]] = call <vscale x 12 x float> @llvm.aarch64.sve.tuple.set.nxv12f32.nxv4f32(<vscale x 12 x float> %tuple, i32 1, <vscale x 4 x float> %x)
  // CHECK-NEXT: ret <vscale x 12 x float> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_f32,,)(tuple, 1, x);
}

svfloat64x3_t test_svset3_f64(svfloat64x3_t tuple, svfloat64_t x)
{
  // CHECK-LABEL: test_svset3_f64
  // CHECK: %[[INSERT:.*]] = call <vscale x 6 x double> @llvm.aarch64.sve.tuple.set.nxv6f64.nxv2f64(<vscale x 6 x double> %tuple, i32 2, <vscale x 2 x double> %x)
  // CHECK-NEXT: ret <vscale x 6 x double> %[[INSERT]]
  return SVE_ACLE_FUNC(svset3,_f64,,)(tuple, 2, x);
}
