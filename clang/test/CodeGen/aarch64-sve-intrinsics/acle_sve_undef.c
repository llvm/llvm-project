// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

svint8_t test_svundef_s8()
{
  // CHECK-LABEL: test_svundef_s8
  // CHECK: ret <vscale x 16 x i8> undef
  return svundef_s8();
}

svint16_t test_svundef_s16()
{
  // CHECK-LABEL: test_svundef_s16
  // CHECK: ret <vscale x 8 x i16> undef
  return svundef_s16();
}

svint32_t test_svundef_s32()
{
  // CHECK-LABEL: test_svundef_s32
  // CHECK: ret <vscale x 4 x i32> undef
  return svundef_s32();
}

svint64_t test_svundef_s64()
{
  // CHECK-LABEL: test_svundef_s64
  // CHECK: ret <vscale x 2 x i64> undef
  return svundef_s64();
}

svuint8_t test_svundef_u8()
{
  // CHECK-LABEL: test_svundef_u8
  // CHECK: ret <vscale x 16 x i8> undef
  return svundef_u8();
}

svuint16_t test_svundef_u16()
{
  // CHECK-LABEL: test_svundef_u16
  // CHECK: ret <vscale x 8 x i16> undef
  return svundef_u16();
}

svuint32_t test_svundef_u32()
{
  // CHECK-LABEL: test_svundef_u32
  // CHECK: ret <vscale x 4 x i32> undef
  return svundef_u32();
}

svuint64_t test_svundef_u64()
{
  // CHECK-LABEL: test_svundef_u64
  // CHECK: ret <vscale x 2 x i64> undef
  return svundef_u64();
}

svfloat16_t test_svundef_f16()
{
  // CHECK-LABEL: test_svundef_f16
  // CHECK: ret <vscale x 8 x half> undef
  return svundef_f16();
}

svfloat32_t test_svundef_f32()
{
  // CHECK-LABEL: test_svundef_f32
  // CHECK: ret <vscale x 4 x float> undef
  return svundef_f32();
}

svfloat64_t test_svundef_f64()
{
  // CHECK-LABEL: test_svundef_f64
  // CHECK: ret <vscale x 2 x double> undef
  return svundef_f64();
}
