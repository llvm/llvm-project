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
svint8_t test_svget4_s8(svint8x4_t tuple)
{
  // CHECK-LABEL: test_svget4_s8
  // CHECK: %[[EXT:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv64i8(<vscale x 64 x i8> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 16 x i8> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_s8,,)(tuple, 0);
}

svint16_t test_svget4_s16(svint16x4_t tuple)
{
  // CHECK-LABEL: test_svget4_s16
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv32i16(<vscale x 32 x i16> %tuple, i32 2)
  // CHECK-NEXT: ret <vscale x 8 x i16> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_s16,,)(tuple, 2);
}

svint32_t test_svget4_s32(svint32x4_t tuple)
{
  // CHECK-LABEL: test_svget4_s32
  // CHECK: %[[EXT:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv16i32(<vscale x 16 x i32> %tuple, i32 2)
  // CHECK-NEXT: ret <vscale x 4 x i32> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_s32,,)(tuple, 2);
}

svint64_t test_svget4_s64(svint64x4_t tuple)
{
  // CHECK-LABEL: test_svget4_s64
  // CHECK: %[[EXT:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv8i64(<vscale x 8 x i64> %tuple, i32 3)
  // CHECK-NEXT: ret <vscale x 2 x i64> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_s64,,)(tuple, 3);
}

svuint8_t test_svget4_u8(svuint8x4_t tuple)
{
  // CHECK-LABEL: test_svget4_u8
  // CHECK: %[[EXT:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv64i8(<vscale x 64 x i8> %tuple, i32 2)
  // CHECK-NEXT: ret <vscale x 16 x i8> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_u8,,)(tuple, 2);
}

svuint16_t test_svget4_u16(svuint16x4_t tuple)
{
  // CHECK-LABEL: test_svget4_u16
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv32i16(<vscale x 32 x i16> %tuple, i32 3)
  // CHECK-NEXT: ret <vscale x 8 x i16> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_u16,,)(tuple, 3);
}

svuint32_t test_svget4_u32(svuint32x4_t tuple)
{
  // CHECK-LABEL: test_svget4_u32
  // CHECK: %[[EXT:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv16i32(<vscale x 16 x i32> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 4 x i32> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_u32,,)(tuple, 0);
}

svuint64_t test_svget4_u64(svuint64x4_t tuple)
{
  // CHECK-LABEL: test_svget4_u64
  // CHECK: %[[EXT:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv8i64(<vscale x 8 x i64> %tuple, i32 3)
  // CHECK-NEXT: ret <vscale x 2 x i64> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_u64,,)(tuple, 3);
}

svfloat16_t test_svget4_f16(svfloat16x4_t tuple)
{
  // CHECK-LABEL: test_svget4_f16
  // CHECK: %[[EXT:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv32f16(<vscale x 32 x half> %tuple, i32 2)
  // CHECK-NEXT: ret <vscale x 8 x half> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_f16,,)(tuple, 2);
}

svfloat32_t test_svget4_f32(svfloat32x4_t tuple)
{
  // CHECK-LABEL: test_svget4_f32
  // CHECK: %[[EXT:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv16f32(<vscale x 16 x float> %tuple, i32 0)
  // CHECK-NEXT: ret <vscale x 4 x float> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_f32,,)(tuple, 0);
}

svfloat64_t test_svget4_f64(svfloat64x4_t tuple)
{
  // CHECK-LABEL: test_svget4_f64
  // CHECK: %[[EXT:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv8f64(<vscale x 8 x double> %tuple, i32 2)
  // CHECK-NEXT: ret <vscale x 2 x double> %[[EXT]]
  return SVE_ACLE_FUNC(svget4,_f64,,)(tuple, 2);
}
