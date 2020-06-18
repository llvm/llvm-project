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

int8_t test_svorv_s8(svbool_t pg, svint8_t op)
{
  // CHECK-LABEL: test_svorv_s8
  // CHECK: %[[INTRINSIC:.*]] = call i8 @llvm.aarch64.sve.orv.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op)
  // CHECK: ret i8 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorv,_s8,,)(pg, op);
}

int16_t test_svorv_s16(svbool_t pg, svint16_t op)
{
  // CHECK-LABEL: test_svorv_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i16 @llvm.aarch64.sve.orv.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op)
  // CHECK: ret i16 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorv,_s16,,)(pg, op);
}

int32_t test_svorv_s32(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svorv_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.orv.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorv,_s32,,)(pg, op);
}

int64_t test_svorv_s64(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svorv_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.orv.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorv,_s64,,)(pg, op);
}

uint8_t test_svorv_u8(svbool_t pg, svuint8_t op)
{
  // CHECK-LABEL: test_svorv_u8
  // CHECK: %[[INTRINSIC:.*]] = call i8 @llvm.aarch64.sve.orv.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op)
  // CHECK: ret i8 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorv,_u8,,)(pg, op);
}

uint16_t test_svorv_u16(svbool_t pg, svuint16_t op)
{
  // CHECK-LABEL: test_svorv_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i16 @llvm.aarch64.sve.orv.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op)
  // CHECK: ret i16 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorv,_u16,,)(pg, op);
}

uint32_t test_svorv_u32(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svorv_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.orv.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorv,_u32,,)(pg, op);
}

uint64_t test_svorv_u64(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svorv_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.orv.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svorv,_u64,,)(pg, op);
}
