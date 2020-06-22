// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O2 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

svint8x4_t test_svundef4_s8()
{
  // CHECK-LABEL: test_svundef4_s8
  // CHECK: ret <vscale x 64 x i8> undef
  return svundef4_s8();
}

svint16x4_t test_svundef4_s16()
{
  // CHECK-LABEL: test_svundef4_s16
  // CHECK: ret <vscale x 32 x i16> undef
  return svundef4_s16();
}

svint32x4_t test_svundef4_s32()
{
  // CHECK-LABEL: test_svundef4_s32
  // CHECK: ret <vscale x 16 x i32> undef
  return svundef4_s32();
}

svint64x4_t test_svundef4_s64()
{
  // CHECK-LABEL: test_svundef4_s64
  // CHECK: ret <vscale x 8 x i64> undef
  return svundef4_s64();
}

svuint8x4_t test_svundef4_u8()
{
  // CHECK-LABEL: test_svundef4_u8
  // CHECK: ret <vscale x 64 x i8> undef
  return svundef4_u8();
}

svuint16x4_t test_svundef4_u16()
{
  // CHECK-LABEL: test_svundef4_u16
  // CHECK: ret <vscale x 32 x i16> undef
  return svundef4_u16();
}

svuint32x4_t test_svundef4_u32()
{
  // CHECK-LABEL: test_svundef4_u32
  // CHECK: ret <vscale x 16 x i32> undef
  return svundef4_u32();
}

svuint64x4_t test_svundef4_u64()
{
  // CHECK-LABEL: test_svundef4_u64
  // CHECK: ret <vscale x 8 x i64> undef
  return svundef4_u64();
}

svfloat16x4_t test_svundef4_f16()
{
  // CHECK-LABEL: test_svundef4_f16
  // CHECK: ret <vscale x 32 x half> undef
  return svundef4_f16();
}

svfloat32x4_t test_svundef4_f32()
{
  // CHECK-LABEL: test_svundef4_f32
  // CHECK: ret <vscale x 16 x float> undef
  return svundef4_f32();
}

svfloat64x4_t test_svundef4_f64()
{
  // CHECK-LABEL: test_svundef4_f64
  // CHECK: ret <vscale x 8 x double> undef
  return svundef4_f64();
}
