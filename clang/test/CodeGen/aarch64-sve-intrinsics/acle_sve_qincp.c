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

int32_t test_svqincp_n_s32_b8(int32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_s32_b8
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqincp.n32.nxv16i1(i32 %op, <vscale x 16 x i1> %pg)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_s32,_b8,)(op, pg);
}

int32_t test_svqincp_n_s32_b16(int32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_s32_b16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqincp.n32.nxv8i1(i32 %op, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_s32,_b16,)(op, pg);
}

int32_t test_svqincp_n_s32_b32(int32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_s32_b32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqincp.n32.nxv4i1(i32 %op, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_s32,_b32,)(op, pg);
}

int32_t test_svqincp_n_s32_b64(int32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_s32_b64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqincp.n32.nxv2i1(i32 %op, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_s32,_b64,)(op, pg);
}

int64_t test_svqincp_n_s64_b8(int64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_s64_b8
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqincp.n64.nxv16i1(i64 %op, <vscale x 16 x i1> %pg)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_s64,_b8,)(op, pg);
}

int64_t test_svqincp_n_s64_b16(int64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_s64_b16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqincp.n64.nxv8i1(i64 %op, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_s64,_b16,)(op, pg);
}

int64_t test_svqincp_n_s64_b32(int64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_s64_b32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqincp.n64.nxv4i1(i64 %op, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_s64,_b32,)(op, pg);
}

int64_t test_svqincp_n_s64_b64(int64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_s64_b64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqincp.n64.nxv2i1(i64 %op, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_s64,_b64,)(op, pg);
}

uint32_t test_svqincp_n_u32_b8(uint32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_u32_b8
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqincp.n32.nxv16i1(i32 %op, <vscale x 16 x i1> %pg)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_u32,_b8,)(op, pg);
}

uint32_t test_svqincp_n_u32_b16(uint32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_u32_b16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqincp.n32.nxv8i1(i32 %op, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_u32,_b16,)(op, pg);
}

uint32_t test_svqincp_n_u32_b32(uint32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_u32_b32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqincp.n32.nxv4i1(i32 %op, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_u32,_b32,)(op, pg);
}

uint32_t test_svqincp_n_u32_b64(uint32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_u32_b64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqincp.n32.nxv2i1(i32 %op, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_u32,_b64,)(op, pg);
}

uint64_t test_svqincp_n_u64_b8(uint64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_u64_b8
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqincp.n64.nxv16i1(i64 %op, <vscale x 16 x i1> %pg)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_u64,_b8,)(op, pg);
}

uint64_t test_svqincp_n_u64_b16(uint64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_u64_b16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqincp.n64.nxv8i1(i64 %op, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_u64,_b16,)(op, pg);
}

uint64_t test_svqincp_n_u64_b32(uint64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_u64_b32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqincp.n64.nxv4i1(i64 %op, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_u64,_b32,)(op, pg);
}

uint64_t test_svqincp_n_u64_b64(uint64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_n_u64_b64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqincp.n64.nxv2i1(i64 %op, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_n_u64,_b64,)(op, pg);
}

svint16_t test_svqincp_s16(svint16_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqincp.nxv8i16(<vscale x 8 x i16> %op, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_s16,,)(op, pg);
}

svint32_t test_svqincp_s32(svint32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqincp.nxv4i32(<vscale x 4 x i32> %op, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_s32,,)(op, pg);
}

svint64_t test_svqincp_s64(svint64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqincp.nxv2i64(<vscale x 2 x i64> %op, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_s64,,)(op, pg);
}

svuint16_t test_svqincp_u16(svuint16_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqincp.nxv8i16(<vscale x 8 x i16> %op, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_u16,,)(op, pg);
}

svuint32_t test_svqincp_u32(svuint32_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uqincp.nxv4i32(<vscale x 4 x i32> %op, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_u32,,)(op, pg);
}

svuint64_t test_svqincp_u64(svuint64_t op, svbool_t pg)
{
  // CHECK-LABEL: test_svqincp_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uqincp.nxv2i64(<vscale x 2 x i64> %op, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqincp,_u64,,)(op, pg);
}
