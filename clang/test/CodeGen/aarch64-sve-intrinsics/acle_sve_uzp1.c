// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - %s >/dev/null
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svuzp1_s8(svint8_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svuzp1_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.uzp1.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_s8,,)(op1, op2);
}

svint16_t test_svuzp1_s16(svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svuzp1_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uzp1.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_s16,,)(op1, op2);
}

svint32_t test_svuzp1_s32(svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svuzp1_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uzp1.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_s32,,)(op1, op2);
}

svint64_t test_svuzp1_s64(svint64_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svuzp1_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uzp1.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_s64,,)(op1, op2);
}

svuint8_t test_svuzp1_u8(svuint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svuzp1_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.uzp1.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_u8,,)(op1, op2);
}

svuint16_t test_svuzp1_u16(svuint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svuzp1_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uzp1.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_u16,,)(op1, op2);
}

svuint32_t test_svuzp1_u32(svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svuzp1_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uzp1.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_u32,,)(op1, op2);
}

svuint64_t test_svuzp1_u64(svuint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svuzp1_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uzp1.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_u64,,)(op1, op2);
}

svfloat16_t test_svuzp1_f16(svfloat16_t op1, svfloat16_t op2)
{
  // CHECK-LABEL: test_svuzp1_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.uzp1.nxv8f16(<vscale x 8 x half> %op1, <vscale x 8 x half> %op2)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_f16,,)(op1, op2);
}

svfloat32_t test_svuzp1_f32(svfloat32_t op1, svfloat32_t op2)
{
  // CHECK-LABEL: test_svuzp1_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.uzp1.nxv4f32(<vscale x 4 x float> %op1, <vscale x 4 x float> %op2)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_f32,,)(op1, op2);
}

svfloat64_t test_svuzp1_f64(svfloat64_t op1, svfloat64_t op2)
{
  // CHECK-LABEL: test_svuzp1_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.uzp1.nxv2f64(<vscale x 2 x double> %op1, <vscale x 2 x double> %op2)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svuzp1,_f64,,)(op1, op2);
}

svbool_t test_svuzp1_b8(svbool_t op1, svbool_t op2)
{
  // CHECK-LABEL: test_svuzp1_b8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.uzp1.nxv16i1(<vscale x 16 x i1> %op1, <vscale x 16 x i1> %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svuzp1_b8(op1, op2);
}

svbool_t test_svuzp1_b16(svbool_t op1, svbool_t op2)
{
  // CHECK-LABEL: test_svuzp1_b16
  // CHECK-DAG: %[[OP1:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %op1)
  // CHECK-DAG: %[[OP2:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.uzp1.nxv8i1(<vscale x 8 x i1> %[[OP1]], <vscale x 8 x i1> %[[OP2]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svuzp1_b16(op1, op2);
}

svbool_t test_svuzp1_b32(svbool_t op1, svbool_t op2)
{
  // CHECK-LABEL: test_svuzp1_b32
  // CHECK-DAG: %[[OP1:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %op1)
  // CHECK-DAG: %[[OP2:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.uzp1.nxv4i1(<vscale x 4 x i1> %[[OP1]], <vscale x 4 x i1> %[[OP2]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svuzp1_b32(op1, op2);
}

svbool_t test_svuzp1_b64(svbool_t op1, svbool_t op2)
{
  // CHECK-LABEL: test_svuzp1_b64
  // CHECK-DAG: %[[OP1:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %op1)
  // CHECK-DAG: %[[OP2:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.uzp1.nxv2i1(<vscale x 2 x i1> %[[OP1]], <vscale x 2 x i1> %[[OP2]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svuzp1_b64(op1, op2);
}
