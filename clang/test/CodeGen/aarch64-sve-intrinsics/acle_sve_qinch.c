// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

int32_t test_svqinch_n_s32(int32_t op)
{
  // CHECK-LABEL: test_svqinch_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqinch.n32(i32 %op, i32 31, i32 1)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch,_n_s32,,)(op, 1);
}

int32_t test_svqinch_n_s32_1(int32_t op)
{
  // CHECK-LABEL: test_svqinch_n_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqinch.n32(i32 %op, i32 31, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch,_n_s32,,)(op, 16);
}

int64_t test_svqinch_n_s64(int64_t op)
{
  // CHECK-LABEL: test_svqinch_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqinch.n64(i64 %op, i32 31, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch,_n_s64,,)(op, 1);
}

uint32_t test_svqinch_n_u32(uint32_t op)
{
  // CHECK-LABEL: test_svqinch_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqinch.n32(i32 %op, i32 31, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch,_n_u32,,)(op, 16);
}

uint64_t test_svqinch_n_u64(uint64_t op)
{
  // CHECK-LABEL: test_svqinch_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqinch.n64(i64 %op, i32 31, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch,_n_u64,,)(op, 1);
}

int32_t test_svqinch_pat_n_s32(int32_t op)
{
  // CHECK-LABEL: test_svqinch_pat_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqinch.n32(i32 %op, i32 30, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch_pat,_n_s32,,)(op, SV_MUL3, 16);
}

int64_t test_svqinch_pat_n_s64(int64_t op)
{
  // CHECK-LABEL: test_svqinch_pat_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqinch.n64(i64 %op, i32 31, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch_pat,_n_s64,,)(op, SV_ALL, 1);
}

uint32_t test_svqinch_pat_n_u32(uint32_t op)
{
  // CHECK-LABEL: test_svqinch_pat_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqinch.n32(i32 %op, i32 0, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch_pat,_n_u32,,)(op, SV_POW2, 16);
}

uint64_t test_svqinch_pat_n_u64(uint64_t op)
{
  // CHECK-LABEL: test_svqinch_pat_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqinch.n64(i64 %op, i32 1, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch_pat,_n_u64,,)(op, SV_VL1, 1);
}

svint16_t test_svqinch_s16(svint16_t op)
{
  // CHECK-LABEL: test_svqinch_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqinch.nxv8i16(<vscale x 8 x i16> %op, i32 31, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch,_s16,,)(op, 16);
}

svuint16_t test_svqinch_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svqinch_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqinch.nxv8i16(<vscale x 8 x i16> %op, i32 31, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch,_u16,,)(op, 1);
}

svint16_t test_svqinch_pat_s16(svint16_t op)
{
  // CHECK-LABEL: test_svqinch_pat_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqinch.nxv8i16(<vscale x 8 x i16> %op, i32 2, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch_pat,_s16,,)(op, SV_VL2, 16);
}

svuint16_t test_svqinch_pat_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svqinch_pat_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqinch.nxv8i16(<vscale x 8 x i16> %op, i32 3, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqinch_pat,_u16,,)(op, SV_VL3, 1);
}
