// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O2 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

svint8x3_t test_svundef3_s8()
{
  // CHECK-LABEL: test_svundef3_s8
  // CHECK: ret <vscale x 48 x i8> undef
  return svundef3_s8();
}

svint16x3_t test_svundef3_s16()
{
  // CHECK-LABEL: test_svundef3_s16
  // CHECK: ret <vscale x 24 x i16> undef
  return svundef3_s16();
}

svint32x3_t test_svundef3_s32()
{
  // CHECK-LABEL: test_svundef3_s32
  // CHECK: ret <vscale x 12 x i32> undef
  return svundef3_s32();
}

svint64x3_t test_svundef3_s64()
{
  // CHECK-LABEL: test_svundef3_s64
  // CHECK: ret <vscale x 6 x i64> undef
  return svundef3_s64();
}

svuint8x3_t test_svundef3_u8()
{
  // CHECK-LABEL: test_svundef3_u8
  // CHECK: ret <vscale x 48 x i8> undef
  return svundef3_u8();
}

svuint16x3_t test_svundef3_u16()
{
  // CHECK-LABEL: test_svundef3_u16
  // CHECK: ret <vscale x 24 x i16> undef
  return svundef3_u16();
}

svuint32x3_t test_svundef3_u32()
{
  // CHECK-LABEL: test_svundef3_u32
  // CHECK: ret <vscale x 12 x i32> undef
  return svundef3_u32();
}

svuint64x3_t test_svundef3_u64()
{
  // CHECK-LABEL: test_svundef3_u64
  // CHECK: ret <vscale x 6 x i64> undef
  return svundef3_u64();
}

svfloat16x3_t test_svundef3_f16()
{
  // CHECK-LABEL: test_svundef3_f16
  // CHECK: ret <vscale x 24 x half> undef
  return svundef3_f16();
}

svfloat32x3_t test_svundef3_f32()
{
  // CHECK-LABEL: test_svundef3_f32
  // CHECK: ret <vscale x 12 x float> undef
  return svundef3_f32();
}

svfloat64x3_t test_svundef3_f64()
{
  // CHECK-LABEL: test_svundef3_f64
  // CHECK: ret <vscale x 6 x double> undef
  return svundef3_f64();
}
