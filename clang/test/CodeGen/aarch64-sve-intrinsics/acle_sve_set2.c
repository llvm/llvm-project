// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8x2_t test_svset2_s8(svint8x2_t tuple, svint8_t x)
{
  // CHECK-LABEL: test_svset2_s8
  // CHECK: %[[INSERT:.*]] = call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.set.nxv32i8.nxv16i8(<vscale x 32 x i8> %tuple, i32 1, <vscale x 16 x i8> %x)
  // CHECK-NEXT: ret <vscale x 32 x i8> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_s8,,)(tuple, 1, x);
}

svint16x2_t test_svset2_s16(svint16x2_t tuple, svint16_t x)
{
  // CHECK-LABEL: test_svset2_s16
  // CHECK: %[[INSERT:.*]] = call <vscale x 16 x i16> @llvm.aarch64.sve.tuple.set.nxv16i16.nxv8i16(<vscale x 16 x i16> %tuple, i32 0, <vscale x 8 x i16> %x)
  // CHECK-NEXT: ret <vscale x 16 x i16> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_s16,,)(tuple, 0, x);
}

svint32x2_t test_svset2_s32(svint32x2_t tuple, svint32_t x)
{
  // CHECK-LABEL: test_svset2_s32
  // CHECK: %[[INSERT:.*]] = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.set.nxv8i32.nxv4i32(<vscale x 8 x i32> %tuple, i32 1, <vscale x 4 x i32> %x)
  // CHECK-NEXT: ret <vscale x 8 x i32> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_s32,,)(tuple, 1, x);
}

svint64x2_t test_svset2_s64(svint64x2_t tuple, svint64_t x)
{
  // CHECK-LABEL: test_svset2_s64
  // CHECK: %[[INSERT:.*]] = call <vscale x 4 x i64> @llvm.aarch64.sve.tuple.set.nxv4i64.nxv2i64(<vscale x 4 x i64> %tuple, i32 0, <vscale x 2 x i64> %x)
  // CHECK-NEXT: ret <vscale x 4 x i64> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_s64,,)(tuple, 0, x);
}

svuint8x2_t test_svset2_u8(svuint8x2_t tuple, svuint8_t x)
{
  // CHECK-LABEL: test_svset2_u8
  // CHECK: %[[INSERT:.*]] = call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.set.nxv32i8.nxv16i8(<vscale x 32 x i8> %tuple, i32 1, <vscale x 16 x i8> %x)
  // CHECK-NEXT: ret <vscale x 32 x i8> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_u8,,)(tuple, 1, x);
}

svuint16x2_t test_svset2_u16(svuint16x2_t tuple, svuint16_t x)
{
  // CHECK-LABEL: test_svset2_u16
  // CHECK: %[[INSERT:.*]] = call <vscale x 16 x i16> @llvm.aarch64.sve.tuple.set.nxv16i16.nxv8i16(<vscale x 16 x i16> %tuple, i32 0, <vscale x 8 x i16> %x)
  // CHECK-NEXT: ret <vscale x 16 x i16> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_u16,,)(tuple, 0, x);
}

svuint32x2_t test_svset2_u32(svuint32x2_t tuple, svuint32_t x)
{
  // CHECK-LABEL: test_svset2_u32
  // CHECK: %[[INSERT:.*]] = call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.set.nxv8i32.nxv4i32(<vscale x 8 x i32> %tuple, i32 1, <vscale x 4 x i32> %x)
  // CHECK-NEXT: ret <vscale x 8 x i32> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_u32,,)(tuple, 1, x);
}

svuint64x2_t test_svset2_u64(svuint64x2_t tuple, svuint64_t x)
{
  // CHECK-LABEL: test_svset2_u64
  // CHECK: %[[INSERT:.*]] = call <vscale x 4 x i64> @llvm.aarch64.sve.tuple.set.nxv4i64.nxv2i64(<vscale x 4 x i64> %tuple, i32 0, <vscale x 2 x i64> %x)
  // CHECK-NEXT: ret <vscale x 4 x i64> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_u64,,)(tuple, 0, x);
}

svfloat16x2_t test_svset2_f16(svfloat16x2_t tuple, svfloat16_t x)
{
  // CHECK-LABEL: test_svset2_f16
  // CHECK: %[[INSERT:.*]] = call <vscale x 16 x half> @llvm.aarch64.sve.tuple.set.nxv16f16.nxv8f16(<vscale x 16 x half> %tuple, i32 1, <vscale x 8 x half> %x)
  // CHECK-NEXT: ret <vscale x 16 x half> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_f16,,)(tuple, 1, x);
}

svfloat32x2_t test_svset2_f32(svfloat32x2_t tuple, svfloat32_t x)
{
  // CHECK-LABEL: test_svset2_f32
  // CHECK: %[[INSERT:.*]] = call <vscale x 8 x float> @llvm.aarch64.sve.tuple.set.nxv8f32.nxv4f32(<vscale x 8 x float> %tuple, i32 0, <vscale x 4 x float> %x)
  // CHECK-NEXT: ret <vscale x 8 x float> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_f32,,)(tuple, 0, x);
}

svfloat64x2_t test_svset2_f64(svfloat64x2_t tuple, svfloat64_t x)
{
  // CHECK-LABEL: test_svset2_f64
  // CHECK: %[[INSERT:.*]] = call <vscale x 4 x double> @llvm.aarch64.sve.tuple.set.nxv4f64.nxv2f64(<vscale x 4 x double> %tuple, i32 1, <vscale x 2 x double> %x)
  // CHECK-NEXT: ret <vscale x 4 x double> %[[INSERT]]
  return SVE_ACLE_FUNC(svset2,_f64,,)(tuple, 1, x);
}
