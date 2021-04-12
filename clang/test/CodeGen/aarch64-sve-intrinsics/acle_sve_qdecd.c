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

int32_t test_svqdecd_n_s32(int32_t op)
{
  // CHECK-LABEL: test_svqdecd_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqdecd.n32(i32 %op, i32 31, i32 1)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd,_n_s32,,)(op, 1);
}

int32_t test_svqdecd_n_s32_1(int32_t op)
{
  // CHECK-LABEL: test_svqdecd_n_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqdecd.n32(i32 %op, i32 31, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd,_n_s32,,)(op, 16);
}

int64_t test_svqdecd_n_s64(int64_t op)
{
  // CHECK-LABEL: test_svqdecd_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqdecd.n64(i64 %op, i32 31, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd,_n_s64,,)(op, 1);
}

uint32_t test_svqdecd_n_u32(uint32_t op)
{
  // CHECK-LABEL: test_svqdecd_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqdecd.n32(i32 %op, i32 31, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd,_n_u32,,)(op, 16);
}

uint64_t test_svqdecd_n_u64(uint64_t op)
{
  // CHECK-LABEL: test_svqdecd_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqdecd.n64(i64 %op, i32 31, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd,_n_u64,,)(op, 1);
}

int32_t test_svqdecd_pat_n_s32(int32_t op)
{
  // CHECK-LABEL: test_svqdecd_pat_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqdecd.n32(i32 %op, i32 4, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd_pat,_n_s32,,)(op, SV_VL4, 16);
}

int64_t test_svqdecd_pat_n_s64(int64_t op)
{
  // CHECK-LABEL: test_svqdecd_pat_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqdecd.n64(i64 %op, i32 5, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd_pat,_n_s64,,)(op, SV_VL5, 1);
}

uint32_t test_svqdecd_pat_n_u32(uint32_t op)
{
  // CHECK-LABEL: test_svqdecd_pat_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqdecd.n32(i32 %op, i32 6, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd_pat,_n_u32,,)(op, SV_VL6, 16);
}

uint64_t test_svqdecd_pat_n_u64(uint64_t op)
{
  // CHECK-LABEL: test_svqdecd_pat_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqdecd.n64(i64 %op, i32 7, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd_pat,_n_u64,,)(op, SV_VL7, 1);
}

svint64_t test_svqdecd_s64(svint64_t op)
{
  // CHECK-LABEL: test_svqdecd_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdecd.nxv2i64(<vscale x 2 x i64> %op, i32 31, i32 16)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd,_s64,,)(op, 16);
}

svuint64_t test_svqdecd_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svqdecd_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uqdecd.nxv2i64(<vscale x 2 x i64> %op, i32 31, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd,_u64,,)(op, 1);
}

svint64_t test_svqdecd_pat_s64(svint64_t op)
{
  // CHECK-LABEL: test_svqdecd_pat_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdecd.nxv2i64(<vscale x 2 x i64> %op, i32 8, i32 16)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd_pat,_s64,,)(op, SV_VL8, 16);
}

svuint64_t test_svqdecd_pat_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svqdecd_pat_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uqdecd.nxv2i64(<vscale x 2 x i64> %op, i32 9, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecd_pat,_u64,,)(op, SV_VL16, 1);
}
