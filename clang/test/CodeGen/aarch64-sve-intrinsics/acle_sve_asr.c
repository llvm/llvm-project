// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svasr_s8_z(svbool_t pg, svint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svasr_s8_z
  // CHECK: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s8,_z,)(pg, op1, op2);
}

svint16_t test_svasr_s16_z(svbool_t pg, svint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svasr_s16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s16,_z,)(pg, op1, op2);
}

svint32_t test_svasr_s32_z(svbool_t pg, svint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svasr_s32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s32,_z,)(pg, op1, op2);
}

svint64_t test_svasr_s64_z(svbool_t pg, svint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_s64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s64,_z,)(pg, op1, op2);
}

svint8_t test_svasr_s8_m(svbool_t pg, svint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svasr_s8_m
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s8,_m,)(pg, op1, op2);
}

svint16_t test_svasr_s16_m(svbool_t pg, svint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svasr_s16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s16,_m,)(pg, op1, op2);
}

svint32_t test_svasr_s32_m(svbool_t pg, svint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svasr_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s32,_m,)(pg, op1, op2);
}

svint64_t test_svasr_s64_m(svbool_t pg, svint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s64,_m,)(pg, op1, op2);
}

svint8_t test_svasr_s8_x(svbool_t pg, svint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svasr_s8_x
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s8,_x,)(pg, op1, op2);
}

svint16_t test_svasr_s16_x(svbool_t pg, svint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svasr_s16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s16,_x,)(pg, op1, op2);
}

svint32_t test_svasr_s32_x(svbool_t pg, svint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svasr_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s32,_x,)(pg, op1, op2);
}

svint64_t test_svasr_s64_x(svbool_t pg, svint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_s64,_x,)(pg, op1, op2);
}

svint64_t test_svasr_n_s64_z(svbool_t pg, svint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_n_s64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s64,_z,)(pg, op1, op2);
}

svint64_t test_svasr_n_s64_m(svbool_t pg, svint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_n_s64_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s64,_m,)(pg, op1, op2);
}

svint64_t test_svasr_n_s64_x(svbool_t pg, svint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_n_s64_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s64,_x,)(pg, op1, op2);
}

svint8_t test_svasr_wide_s8_z(svbool_t pg, svint8_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s8_z
  // CHECK: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s8,_z,)(pg, op1, op2);
}

svint16_t test_svasr_wide_s16_z(svbool_t pg, svint16_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s16,_z,)(pg, op1, op2);
}

svint32_t test_svasr_wide_s32_z(svbool_t pg, svint32_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s32,_z,)(pg, op1, op2);
}

svint8_t test_svasr_wide_s8_m(svbool_t pg, svint8_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s8_m
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s8,_m,)(pg, op1, op2);
}

svint16_t test_svasr_wide_s16_m(svbool_t pg, svint16_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s16,_m,)(pg, op1, op2);
}

svint32_t test_svasr_wide_s32_m(svbool_t pg, svint32_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s32,_m,)(pg, op1, op2);
}

svint8_t test_svasr_wide_s8_x(svbool_t pg, svint8_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s8_x
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s8,_x,)(pg, op1, op2);
}

svint16_t test_svasr_wide_s16_x(svbool_t pg, svint16_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s16,_x,)(pg, op1, op2);
}

svint32_t test_svasr_wide_s32_x(svbool_t pg, svint32_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_s32,_x,)(pg, op1, op2);
}

svint8_t test_svasr_n_s8_z(svbool_t pg, svint8_t op1, uint8_t op2)
{
  // CHECK-LABEL: test_svasr_n_s8_z
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[SEL]], <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s8,_z,)(pg, op1, op2);
}

svint16_t test_svasr_n_s16_z(svbool_t pg, svint16_t op1, uint16_t op2)
{
  // CHECK-LABEL: test_svasr_n_s16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op2)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s16,_z,)(pg, op1, op2);
}

svint32_t test_svasr_n_s32_z(svbool_t pg, svint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svasr_n_s32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s32,_z,)(pg, op1, op2);
}

svint8_t test_svasr_n_s8_m(svbool_t pg, svint8_t op1, uint8_t op2)
{
  // CHECK-LABEL: test_svasr_n_s8_m
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s8,_m,)(pg, op1, op2);
}

svint16_t test_svasr_n_s16_m(svbool_t pg, svint16_t op1, uint16_t op2)
{
  // CHECK-LABEL: test_svasr_n_s16_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s16,_m,)(pg, op1, op2);
}

svint32_t test_svasr_n_s32_m(svbool_t pg, svint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svasr_n_s32_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s32,_m,)(pg, op1, op2);
}

svint8_t test_svasr_n_s8_x(svbool_t pg, svint8_t op1, uint8_t op2)
{
  // CHECK-LABEL: test_svasr_n_s8_x
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s8,_x,)(pg, op1, op2);
}

svint16_t test_svasr_n_s16_x(svbool_t pg, svint16_t op1, uint16_t op2)
{
  // CHECK-LABEL: test_svasr_n_s16_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s16,_x,)(pg, op1, op2);
}

svint32_t test_svasr_n_s32_x(svbool_t pg, svint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svasr_n_s32_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr,_n_s32,_x,)(pg, op1, op2);
}

svint8_t test_svasr_wide_n_s8_m(svbool_t pg, svint8_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s8_m
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s8,_m,)(pg, op1, op2);
}

svint16_t test_svasr_wide_n_s16_m(svbool_t pg, svint16_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s16_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s16,_m,)(pg, op1, op2);
}

svint32_t test_svasr_wide_n_s32_m(svbool_t pg, svint32_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s32_m
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s32,_m,)(pg, op1, op2);
}

svint8_t test_svasr_wide_n_s8_z(svbool_t pg, svint8_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s8_z
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sel.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %[[PG]], <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s8,_z,)(pg, op1, op2);
}

svint16_t test_svasr_wide_n_s16_z(svbool_t pg, svint16_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK-DAG: %[[OP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[OP]], <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s16,_z,)(pg, op1, op2);
}

svint32_t test_svasr_wide_n_s32_z(svbool_t pg, svint32_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK-DAG: %[[OP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[OP]], <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s32,_z,)(pg, op1, op2);
}

svint8_t test_svasr_wide_n_s8_x(svbool_t pg, svint8_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s8_x
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s8,_x,)(pg, op1, op2);
}

svint16_t test_svasr_wide_n_s16_x(svbool_t pg, svint16_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s16_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s16,_x,)(pg, op1, op2);
}

svint32_t test_svasr_wide_n_s32_x(svbool_t pg, svint32_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svasr_wide_n_s32_x
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svasr_wide,_n_s32,_x,)(pg, op1, op2);
}
