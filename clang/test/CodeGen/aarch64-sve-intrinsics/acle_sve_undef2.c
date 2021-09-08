// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O2 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O2 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s

#include <arm_sve.h>

svint8x2_t test_svundef2_s8()
{
  // CHECK-LABEL: test_svundef2_s8
  // CHECK: ret <vscale x 32 x i8> undef
  return svundef2_s8();
}

svint16x2_t test_svundef2_s16()
{
  // CHECK-LABEL: test_svundef2_s16
  // CHECK: ret <vscale x 16 x i16> undef
  return svundef2_s16();
}

svint32x2_t test_svundef2_s32()
{
  // CHECK-LABEL: test_svundef2_s32
  // CHECK: ret <vscale x 8 x i32> undef
  return svundef2_s32();
}

svint64x2_t test_svundef2_s64()
{
  // CHECK-LABEL: test_svundef2_s64
  // CHECK: ret <vscale x 4 x i64> undef
  return svundef2_s64();
}

svuint8x2_t test_svundef2_u8()
{
  // CHECK-LABEL: test_svundef2_u8
  // CHECK: ret <vscale x 32 x i8> undef
  return svundef2_u8();
}

svuint16x2_t test_svundef2_u16()
{
  // CHECK-LABEL: test_svundef2_u16
  // CHECK: ret <vscale x 16 x i16> undef
  return svundef2_u16();
}

svuint32x2_t test_svundef2_u32()
{
  // CHECK-LABEL: test_svundef2_u32
  // CHECK: ret <vscale x 8 x i32> undef
  return svundef2_u32();
}

svuint64x2_t test_svundef2_u64()
{
  // CHECK-LABEL: test_svundef2_u64
  // CHECK: ret <vscale x 4 x i64> undef
  return svundef2_u64();
}

svfloat16x2_t test_svundef2_f16()
{
  // CHECK-LABEL: test_svundef2_f16
  // CHECK: ret <vscale x 16 x half> undef
  return svundef2_f16();
}

svfloat32x2_t test_svundef2_f32()
{
  // CHECK-LABEL: test_svundef2_f32
  // CHECK: ret <vscale x 8 x float> undef
  return svundef2_f32();
}

svfloat64x2_t test_svundef2_f64()
{
  // CHECK-LABEL: test_svundef2_f64
  // CHECK: ret <vscale x 4 x double> undef
  return svundef2_f64();
}
