// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
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

svuint32_t test_svadrb_u32base_s32offset(svuint32_t bases, svint32_t offsets)
{
  // CHECK-LABEL: test_svadrb_u32base_s32offset
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.adrb.nxv4i32(<vscale x 4 x i32> %bases, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svadrb_,u32base_s32,offset,)(bases, offsets);
}

svuint64_t test_svadrb_u64base_s64offset(svuint64_t bases, svint64_t offsets)
{
  // CHECK-LABEL: test_svadrb_u64base_s64offset
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.adrb.nxv2i64(<vscale x 2 x i64>  %bases, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svadrb_,u64base_s64,offset,)(bases, offsets);
}

svuint32_t test_svadrb_u32base_u32offset(svuint32_t bases, svuint32_t offsets)
{
  // CHECK-LABEL: test_svadrb_u32base_u32offset
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.adrb.nxv4i32(<vscale x 4 x i32> %bases, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svadrb_,u32base_u32,offset,)(bases, offsets);
}

svuint64_t test_svadrb_u64base_u64offset(svuint64_t bases, svuint64_t offsets)
{
  // CHECK-LABEL: test_svadrb_u64base_u64offset
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.adrb.nxv2i64(<vscale x 2 x i64>  %bases, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svadrb_,u64base_u64,offset,)(bases, offsets);
}
