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

svint8_t test_svtbl_s8(svint8_t data, svuint8_t indices)
{
  // CHECK-LABEL: test_svtbl_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbl.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8> %indices)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_s8,,)(data, indices);
}

svint16_t test_svtbl_s16(svint16_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbl_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16> %indices)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_s16,,)(data, indices);
}

svint32_t test_svtbl_s32(svint32_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbl_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tbl.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_s32,,)(data, indices);
}

svint64_t test_svtbl_s64(svint64_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbl_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tbl.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_s64,,)(data, indices);
}

svuint8_t test_svtbl_u8(svuint8_t data, svuint8_t indices)
{
  // CHECK-LABEL: test_svtbl_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbl.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8> %indices)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_u8,,)(data, indices);
}

svuint16_t test_svtbl_u16(svuint16_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbl_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16> %indices)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_u16,,)(data, indices);
}

svuint32_t test_svtbl_u32(svuint32_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbl_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tbl.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_u32,,)(data, indices);
}

svuint64_t test_svtbl_u64(svuint64_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbl_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tbl.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_u64,,)(data, indices);
}

svfloat16_t test_svtbl_f16(svfloat16_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbl_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.tbl.nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x i16> %indices)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_f16,,)(data, indices);
}

svfloat32_t test_svtbl_f32(svfloat32_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbl_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.tbl.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_f32,,)(data, indices);
}

svfloat64_t test_svtbl_f64(svfloat64_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbl_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.tbl.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svtbl,_f64,,)(data, indices);
}
