// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svint8_t test_svreinterpret_s8_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_s8_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s8, _bf16, , )(op);
}

svint16_t test_svreinterpret_s16_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_s16_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s16, _bf16, , )(op);
}

svint32_t test_svreinterpret_s32_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_s32_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s32, _bf16, , )(op);
}
svint64_t test_svreinterpret_s64_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_s64_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_s64, _bf16, , )(op);
}

svuint8_t test_svreinterpret_u8_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_u8_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 16 x i8>
  // CHECK: ret <vscale x 16 x i8> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u8, _bf16, , )(op);
}

svuint16_t test_svreinterpret_u16_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_u16_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u16, _bf16, , )(op);
}

svuint32_t test_svreinterpret_u32_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_u32_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u32, _bf16, , )(op);
}

svuint64_t test_svreinterpret_u64_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_u64_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_u64, _bf16, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_s8(svint8_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_s8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _s8, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_s16(svint16_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_s16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _s16, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_s32(svint32_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_s32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _s32, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_s64(svint64_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_s64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _s64, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_u8(svuint8_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_u8
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 16 x i8> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _u8, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_u16(svuint16_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_u16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x i16> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _u16, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_u32(svuint32_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_u32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x i32> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _u32, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_u64(svuint64_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_u64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x i64> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _u64, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_bf16
  // CHECK: ret <vscale x 8 x bfloat> %op
  return SVE_ACLE_FUNC(svreinterpret_bf16, _bf16, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_f16(svfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_f16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x half> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _f16, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_f32(svfloat32_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_f32
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 4 x float> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _f32, , )(op);
}

svbfloat16_t test_svreinterpret_bf16_f64(svfloat64_t op) {
  // CHECK-LABEL: test_svreinterpret_bf16_f64
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 2 x double> %op to <vscale x 8 x bfloat>
  // CHECK: ret <vscale x 8 x bfloat> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_bf16, _f64, , )(op);
}

svfloat32_t test_svreinterpret_f32_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_f32_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 4 x float>
  // CHECK: ret <vscale x 4 x float> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f32, _bf16, , )(op);
}

svfloat16_t test_svreinterpret_f16_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_f16_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 8 x half>
  // CHECK: ret <vscale x 8 x half> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f16, _bf16, , )(op);
}

svfloat64_t test_svreinterpret_f64_bf16(svbfloat16_t op) {
  // CHECK-LABEL: test_svreinterpret_f64_bf16
  // CHECK: %[[CAST:.*]] = bitcast <vscale x 8 x bfloat> %op to <vscale x 2 x double>
  // CHECK: ret <vscale x 2 x double> %[[CAST]]
  return SVE_ACLE_FUNC(svreinterpret_f64, _bf16, , )(op);
}
