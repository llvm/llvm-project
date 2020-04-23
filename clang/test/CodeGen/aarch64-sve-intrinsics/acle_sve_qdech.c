// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint16_t test_svqdech_pat_s16(svint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdech.nxv8i16(<vscale x 8 x i16> %op, i32 0, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return svqdech_pat_s16(op, SV_POW2, 1);
}

svint16_t test_svqdech_pat_s16_all(svint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_s16_all
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdech.nxv8i16(<vscale x 8 x i16> %op, i32 31, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return svqdech_pat_s16(op, SV_ALL, 16);
}

svuint16_t test_svqdech_pat_u16_pow2(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_pow2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 0, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_POW2, 16);
}

svuint16_t test_svqdech_pat_u16_vl1(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 1, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL1, 16);
}

svuint16_t test_svqdech_pat_u16_vl2(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 2, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL2, 16);
}

svuint16_t test_svqdech_pat_u16_vl3(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 3, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL3, 16);
}

svuint16_t test_svqdech_pat_u16_vl4(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl4
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 4, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL4, 16);
}

svuint16_t test_svqdech_pat_u16_vl5(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl5
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 5, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL5, 16);
}

svuint16_t test_svqdech_pat_u16_vl6(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl6
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 6, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL6, 16);
}

svuint16_t test_svqdech_pat_u16_vl7(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl7
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 7, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL7, 16);
}

svuint16_t test_svqdech_pat_u16_vl8(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 8, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL8, 16);
}

svuint16_t test_svqdech_pat_u16_vl16(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 9, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL16, 16);
}

svuint16_t test_svqdech_pat_u16_vl32(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 10, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL32, 16);
}

svuint16_t test_svqdech_pat_u16_vl64(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 11, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL64, 16);
}

svuint16_t test_svqdech_pat_u16_vl128(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl128
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 12, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL128, 16);
}

svuint16_t test_svqdech_pat_u16_vl256(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_vl256
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 13, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL256, 16);
}

svuint16_t test_svqdech_pat_u16_mul4(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_mul4
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 29, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_MUL4, 16);
}

svuint16_t test_svqdech_pat_u16_mul3(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_mul3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 30, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_MUL3, 16);
}

svuint16_t test_svqdech_pat_u16_all(svuint16_t op)
{
  // CHECK-LABEL: test_svqdech_pat_u16_all
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %op, i32 31, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_ALL, 16);
}
