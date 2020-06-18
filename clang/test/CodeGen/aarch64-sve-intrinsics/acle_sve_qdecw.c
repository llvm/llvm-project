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

int32_t test_svqdecw_n_s32(int32_t op)
{
  // CHECK-LABEL: test_svqdecw_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqdecw.n32(i32 %op, i32 31, i32 1)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw,_n_s32,,)(op, 1);
}

int32_t test_svqdecw_n_s32_1(int32_t op)
{
  // CHECK-LABEL: test_svqdecw_n_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqdecw.n32(i32 %op, i32 31, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw,_n_s32,,)(op, 16);
}

int64_t test_svqdecw_n_s64(int64_t op)
{
  // CHECK-LABEL: test_svqdecw_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqdecw.n64(i64 %op, i32 31, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw,_n_s64,,)(op, 1);
}

uint32_t test_svqdecw_n_u32(uint32_t op)
{
  // CHECK-LABEL: test_svqdecw_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqdecw.n32(i32 %op, i32 31, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw,_n_u32,,)(op, 16);
}

uint64_t test_svqdecw_n_u64(uint64_t op)
{
  // CHECK-LABEL: test_svqdecw_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqdecw.n64(i64 %op, i32 31, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw,_n_u64,,)(op, 1);
}

int32_t test_svqdecw_pat_n_s32(int32_t op)
{
  // CHECK-LABEL: test_svqdecw_pat_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.sqdecw.n32(i32 %op, i32 31, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw_pat,_n_s32,,)(op, SV_ALL, 16);
}

int64_t test_svqdecw_pat_n_s64(int64_t op)
{
  // CHECK-LABEL: test_svqdecw_pat_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.sqdecw.n64(i64 %op, i32 0, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw_pat,_n_s64,,)(op, SV_POW2, 1);
}

uint32_t test_svqdecw_pat_n_u32(uint32_t op)
{
  // CHECK-LABEL: test_svqdecw_pat_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.uqdecw.n32(i32 %op, i32 1, i32 16)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw_pat,_n_u32,,)(op, SV_VL1, 16);
}

uint64_t test_svqdecw_pat_n_u64(uint64_t op)
{
  // CHECK-LABEL: test_svqdecw_pat_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.uqdecw.n64(i64 %op, i32 2, i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw_pat,_n_u64,,)(op, SV_VL2, 1);
}

svint32_t test_svqdecw_s32(svint32_t op)
{
  // CHECK-LABEL: test_svqdecw_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdecw.nxv4i32(<vscale x 4 x i32> %op, i32 31, i32 16)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw,_s32,,)(op, 16);
}

svuint32_t test_svqdecw_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svqdecw_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uqdecw.nxv4i32(<vscale x 4 x i32> %op, i32 31, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw,_u32,,)(op, 1);
}

svint32_t test_svqdecw_pat_s32(svint32_t op)
{
  // CHECK-LABEL: test_svqdecw_pat_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdecw.nxv4i32(<vscale x 4 x i32> %op, i32 3, i32 16)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw_pat,_s32,,)(op, SV_VL3, 16);
}

svuint32_t test_svqdecw_pat_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svqdecw_pat_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uqdecw.nxv4i32(<vscale x 4 x i32> %op, i32 4, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svqdecw_pat,_u32,,)(op, SV_VL4, 1);
}
