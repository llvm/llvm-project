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

svint8_t test_svinsr_n_s8(svint8_t op1, int8_t op2)
{
  // CHECK-LABEL: test_svinsr_n_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.insr.nxv16i8(<vscale x 16 x i8> %op1, i8 %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_s8,,)(op1, op2);
}

svint16_t test_svinsr_n_s16(svint16_t op1, int16_t op2)
{
  // CHECK-LABEL: test_svinsr_n_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.insr.nxv8i16(<vscale x 8 x i16> %op1, i16 %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_s16,,)(op1, op2);
}

svint32_t test_svinsr_n_s32(svint32_t op1, int32_t op2)
{
  // CHECK-LABEL: test_svinsr_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.insr.nxv4i32(<vscale x 4 x i32> %op1, i32 %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_s32,,)(op1, op2);
}

svint64_t test_svinsr_n_s64(svint64_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svinsr_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.insr.nxv2i64(<vscale x 2 x i64> %op1, i64 %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_s64,,)(op1, op2);
}

svuint8_t test_svinsr_n_u8(svuint8_t op1, uint8_t op2)
{
  // CHECK-LABEL: test_svinsr_n_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.insr.nxv16i8(<vscale x 16 x i8> %op1, i8 %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_u8,,)(op1, op2);
}

svuint16_t test_svinsr_n_u16(svuint16_t op1, uint16_t op2)
{
  // CHECK-LABEL: test_svinsr_n_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.insr.nxv8i16(<vscale x 8 x i16> %op1, i16 %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_u16,,)(op1, op2);
}

svuint32_t test_svinsr_n_u32(svuint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svinsr_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.insr.nxv4i32(<vscale x 4 x i32> %op1, i32 %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_u32,,)(op1, op2);
}

svuint64_t test_svinsr_n_u64(svuint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svinsr_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.insr.nxv2i64(<vscale x 2 x i64> %op1, i64 %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_u64,,)(op1, op2);
}

svfloat16_t test_svinsr_n_f16(svfloat16_t op1, float16_t op2)
{
  // CHECK-LABEL: test_svinsr_n_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.insr.nxv8f16(<vscale x 8 x half> %op1, half %op2)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_f16,,)(op1, op2);
}

svfloat32_t test_svinsr_n_f32(svfloat32_t op1, float32_t op2)
{
  // CHECK-LABEL: test_svinsr_n_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.insr.nxv4f32(<vscale x 4 x float> %op1, float %op2)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_f32,,)(op1, op2);
}

svfloat64_t test_svinsr_n_f64(svfloat64_t op1, float64_t op2)
{
  // CHECK-LABEL: test_svinsr_n_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.insr.nxv2f64(<vscale x 2 x double> %op1, double %op2)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svinsr,_n_f64,,)(op1, op2);
}
