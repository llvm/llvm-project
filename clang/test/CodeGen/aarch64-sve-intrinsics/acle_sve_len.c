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

uint64_t test_svlen_s8(svint8_t op)
{
  // CHECK-LABEL: test_svlen_s8
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 4
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_s8,,)(op);
}

uint64_t test_svlen_s16(svint16_t op)
{
  // CHECK-LABEL: test_svlen_s16
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 3
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_s16,,)(op);
}

uint64_t test_svlen_s32(svint32_t op)
{
  // CHECK-LABEL: test_svlen_s32
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 2
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_s32,,)(op);
}

uint64_t test_svlen_s64(svint64_t op)
{
  // CHECK-LABEL: test_svlen_s64
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 1
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_s64,,)(op);
}

uint64_t test_svlen_u8(svuint8_t op)
{
  // CHECK-LABEL: test_svlen_u8
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 4
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_u8,,)(op);
}

uint64_t test_svlen_u16(svuint16_t op)
{
  // CHECK-LABEL: test_svlen_u16
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 3
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_u16,,)(op);
}

uint64_t test_svlen_u32(svuint32_t op)
{
  // CHECK-LABEL: test_svlen_u32
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 2
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_u32,,)(op);
}

uint64_t test_svlen_u64(svuint64_t op)
{
  // CHECK-LABEL: test_svlen_u64
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 1
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_u64,,)(op);
}

uint64_t test_svlen_f16(svfloat16_t op)
{
  // CHECK-LABEL: test_svlen_f16
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 3
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_f16,,)(op);
}

uint64_t test_svlen_f32(svfloat32_t op)
{
  // CHECK-LABEL: test_svlen_f32
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 2
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_f32,,)(op);
}

uint64_t test_svlen_f64(svfloat64_t op)
{
  // CHECK-LABEL: test_svlen_f64
  // CHECK: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
  // CHECK: %[[SHL:.*]] = shl i64 %[[VSCALE]], 1
  // CHECK: ret i64 %[[SHL]]
  return SVE_ACLE_FUNC(svlen,_f64,,)(op);
}
