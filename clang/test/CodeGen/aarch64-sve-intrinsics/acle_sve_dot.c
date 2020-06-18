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

svint32_t test_svdot_s32(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svdot_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sdot.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot,_s32,,)(op1, op2, op3);
}

svint64_t test_svdot_s64(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svdot_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sdot.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot,_s64,,)(op1, op2, op3);
}

svuint32_t test_svdot_u32(svuint32_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svdot_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.udot.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot,_u32,,)(op1, op2, op3);
}

svuint64_t test_svdot_u64(svuint64_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svdot_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.udot.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot,_u64,,)(op1, op2, op3);
}

svint32_t test_svdot_n_s32(svint32_t op1, svint8_t op2, int8_t op3)
{
  // CHECK-LABEL: test_svdot_n_s32
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sdot.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot,_n_s32,,)(op1, op2, op3);
}

svint64_t test_svdot_n_s64(svint64_t op1, svint16_t op2, int16_t op3)
{
  // CHECK-LABEL: test_svdot_n_s64
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sdot.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot,_n_s64,,)(op1, op2, op3);
}

svuint32_t test_svdot_n_u32(svuint32_t op1, svuint8_t op2, uint8_t op3)
{
  // CHECK-LABEL: test_svdot_n_u32
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.udot.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot,_n_u32,,)(op1, op2, op3);
}

svuint64_t test_svdot_n_u64(svuint64_t op1, svuint16_t op2, uint16_t op3)
{
  // CHECK-LABEL: test_svdot_n_u64
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.udot.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot,_n_u64,,)(op1, op2, op3);
}

svint32_t test_svdot_lane_s32(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svdot_lane_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sdot.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot_lane,_s32,,)(op1, op2, op3, 0);
}

svint32_t test_svdot_lane_s32_1(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svdot_lane_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sdot.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot_lane,_s32,,)(op1, op2, op3, 3);
}

svint64_t test_svdot_lane_s64(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svdot_lane_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sdot.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot_lane,_s64,,)(op1, op2, op3, 0);
}

svint64_t test_svdot_lane_s64_1(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svdot_lane_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sdot.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot_lane,_s64,,)(op1, op2, op3, 1);
}

svuint32_t test_svdot_lane_u32(svuint32_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svdot_lane_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.udot.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot_lane,_u32,,)(op1, op2, op3, 3);
}

svuint64_t test_svdot_lane_u64(svuint64_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svdot_lane_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.udot.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdot_lane,_u64,,)(op1, op2, op3, 1);
}
