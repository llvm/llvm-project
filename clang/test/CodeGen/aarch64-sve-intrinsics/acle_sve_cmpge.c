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

svbool_t test_svcmpge_s8(svbool_t pg, svint8_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svcmpge_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmpge,_s8,,)(pg, op1, op2);
}

svbool_t test_svcmpge_s16(svbool_t pg, svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svcmpge_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_s16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_s32(svbool_t pg, svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svcmpge_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_s32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_s64(svbool_t pg, svint64_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpge.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_s64,,)(pg, op1, op2);
}

svbool_t test_svcmpge_u8(svbool_t pg, svuint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svcmpge_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmpge,_u8,,)(pg, op1, op2);
}

svbool_t test_svcmpge_u16(svbool_t pg, svuint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svcmpge_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_u16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_u32(svbool_t pg, svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svcmpge_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_u32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_u64(svbool_t pg, svuint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.cmphs.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_u64,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_s64(svbool_t pg, svint64_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpge.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_s64,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_u64(svbool_t pg, svuint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.cmphs.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_u64,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_s8(svbool_t pg, svint8_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmpge_wide,_s8,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_s16(svbool_t pg, svint16_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 2 x i64> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge_wide,_s16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_s32(svbool_t pg, svint32_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 2 x i64> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge_wide,_s32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_u8(svbool_t pg, svuint8_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmpge_wide,_u8,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_u16(svbool_t pg, svuint16_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 2 x i64> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge_wide,_u16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_u32(svbool_t pg, svuint32_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 2 x i64> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge_wide,_u32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_s8(svbool_t pg, svint8_t op1, int8_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_s8
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmpge,_n_s8,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_s16(svbool_t pg, svint16_t op1, int16_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_s16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_s32(svbool_t pg, svint32_t op1, int32_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_s32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_u8(svbool_t pg, svuint8_t op1, uint8_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_u8
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmpge,_n_u8,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_u16(svbool_t pg, svuint16_t op1, uint16_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_u16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_u32(svbool_t pg, svuint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_u32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_f16(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // CHECK-LABEL: test_svcmpge_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.fcmpge.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_f16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_f32(svbool_t pg, svfloat32_t op1, svfloat32_t op2)
{
  // CHECK-LABEL: test_svcmpge_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.fcmpge.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_f32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_f64(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
{
  // CHECK-LABEL: test_svcmpge_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.fcmpge.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_f64,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_f16(svbool_t pg, svfloat16_t op1, float16_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_f16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.fcmpge.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %op1, <vscale x 8 x half> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_f16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_f32(svbool_t pg, svfloat32_t op1, float32_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.fcmpge.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %op1, <vscale x 4 x float> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_f32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_n_f64(svbool_t pg, svfloat64_t op1, float64_t op2)
{
  // CHECK-LABEL: test_svcmpge_n_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.fcmpge.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %op1, <vscale x 2 x double> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge,_n_f64,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_n_s8(svbool_t pg, svint8_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_n_s8
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmpge_wide,_n_s8,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_n_s16(svbool_t pg, svint16_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_n_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge_wide,_n_s16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_n_s32(svbool_t pg, svint32_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_n_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge_wide,_n_s32,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_n_u8(svbool_t pg, svuint8_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_n_u8
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svcmpge_wide,_n_u8,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_n_u16(svbool_t pg, svuint16_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_n_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge_wide,_n_u16,,)(pg, op1, op2);
}

svbool_t test_svcmpge_wide_n_u32(svbool_t pg, svuint32_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svcmpge_wide_n_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svcmpge_wide,_n_u32,,)(pg, op1, op2);
}
