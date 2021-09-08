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

svint8_t test_svrev_s8(svint8_t op)
{
  // CHECK-LABEL: test_svrev_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.rev.nxv16i8(<vscale x 16 x i8> %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_s8,,)(op);
}

svint16_t test_svrev_s16(svint16_t op)
{
  // CHECK-LABEL: test_svrev_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.rev.nxv8i16(<vscale x 8 x i16> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_s16,,)(op);
}

svint32_t test_svrev_s32(svint32_t op)
{
  // CHECK-LABEL: test_svrev_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.rev.nxv4i32(<vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_s32,,)(op);
}

svint64_t test_svrev_s64(svint64_t op)
{
  // CHECK-LABEL: test_svrev_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.rev.nxv2i64(<vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_s64,,)(op);
}

svuint8_t test_svrev_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svrev_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.rev.nxv16i8(<vscale x 16 x i8> %op)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_u8,,)(op);
}

svuint16_t test_svrev_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svrev_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.rev.nxv8i16(<vscale x 8 x i16> %op)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_u16,,)(op);
}

svuint32_t test_svrev_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svrev_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.rev.nxv4i32(<vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_u32,,)(op);
}

svuint64_t test_svrev_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svrev_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.rev.nxv2i64(<vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_u64,,)(op);
}

svfloat16_t test_svrev_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svrev_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.rev.nxv8f16(<vscale x 8 x half> %op)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_f16,,)(op);
}

svfloat32_t test_svrev_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svrev_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.rev.nxv4f32(<vscale x 4 x float> %op)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_f32,,)(op);
}

svfloat64_t test_svrev_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svrev_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.rev.nxv2f64(<vscale x 2 x double> %op)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svrev,_f64,,)(op);
}

svbool_t test_svrev_b8(svbool_t op)
{
  // CHECK-LABEL: test_svrev_b8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.rev.nxv16i1(<vscale x 16 x i1> %op)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svrev_b8(op);
}

svbool_t test_svrev_b16(svbool_t op)
{
  // CHECK-LABEL: test_svrev_b16
  // CHECK: %[[OP:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %op)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.rev.nxv8i1(<vscale x 8 x i1> %[[OP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svrev_b16(op);
}

svbool_t test_svrev_b32(svbool_t op)
{
  // CHECK-LABEL: test_svrev_b32
  // CHECK: %[[OP:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %op)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.rev.nxv4i1(<vscale x 4 x i1> %[[OP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svrev_b32(op);
}

svbool_t test_svrev_b64(svbool_t op)
{
  // CHECK-LABEL: test_svrev_b64
  // CHECK: %[[OP:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %op)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.rev.nxv2i1(<vscale x 2 x i1> %[[OP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svrev_b64(op);
}
