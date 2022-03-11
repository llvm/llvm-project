// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN: -fallow-half-arguments-and-returns \
// RUN:  -O1 -emit-llvm -o - %s 2>&1 | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

// ADDITION

svint8_t add_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: add_i8
  // CHECK: %add = add <vscale x 16 x i8> %b, %a
  // CHECK-NEXT: ret <vscale x 16 x i8> %add
  return a + b;
}

svint16_t add_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: add_i16
  // CHECK: %add = add <vscale x 8 x i16> %b, %a
  // CHECK-NEXT: ret <vscale x 8 x i16> %add
  return a + b;
}

svint32_t add_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: add_i32
  // CHECK: %add = add <vscale x 4 x i32> %b, %a
  // CHECK-NEXT: ret <vscale x 4 x i32> %add
  return a + b;
}

svint64_t add_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: add_i64
  // CHECK: %add = add <vscale x 2 x i64> %b, %a
  // CHECK-NEXT: ret <vscale x 2 x i64> %add
  return a + b;
}

svuint8_t add_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: add_u8
  // CHECK: %add = add <vscale x 16 x i8> %b, %a
  // CHECK-NEXT: ret <vscale x 16 x i8> %add
  return a + b;
}

svuint16_t add_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: add_u16
  // CHECK: %add = add <vscale x 8 x i16> %b, %a
  // CHECK-NEXT: ret <vscale x 8 x i16> %add
  return a + b;
}

svuint32_t add_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: add_u32
  // CHECK: %add = add <vscale x 4 x i32> %b, %a
  // CHECK-NEXT: ret <vscale x 4 x i32> %add
  return a + b;
}

svuint64_t add_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: add_u64
  // CHECK: %add = add <vscale x 2 x i64> %b, %a
  // CHECK-NEXT: ret <vscale x 2 x i64> %add
  return a + b;
}

svfloat16_t add_f16(svfloat16_t a, svfloat16_t b) {
  // CHECK-LABEL: add_f16
  // CHECK: %add = fadd <vscale x 8 x half> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x half> %add
  return a + b;
}

svfloat32_t add_f32(svfloat32_t a, svfloat32_t b) {
  // CHECK-LABEL: add_f32
  // CHECK: %add = fadd <vscale x 4 x float> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x float> %add
  return a + b;
}

svfloat64_t add_f64(svfloat64_t a, svfloat64_t b) {
  // CHECK-LABEL: add_f64
  // CHECK: %add = fadd <vscale x 2 x double> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x double> %add
  return a + b;
}

svint8_t add_inplace_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: add_inplace_i8
  // CHECK: %add = add <vscale x 16 x i8> %b, %a
  // CHECK-NEXT: ret <vscale x 16 x i8> %add
  return a += b;
}

svint16_t add_inplace_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: add_inplace_i16
  // CHECK: %add = add <vscale x 8 x i16> %b, %a
  // CHECK-NEXT: ret <vscale x 8 x i16> %add
  return a += b;
}

svint32_t add_inplace_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: add_inplace_i32
  // CHECK: %add = add <vscale x 4 x i32> %b, %a
  // CHECK-NEXT: ret <vscale x 4 x i32> %add
  return a += b;
}

svint64_t add_inplace_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: add_inplace_i64
  // CHECK: %add = add <vscale x 2 x i64> %b, %a
  // CHECK-NEXT: ret <vscale x 2 x i64> %add
  return a += b;
}

svuint8_t add_inplace_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: add_inplace_u8
  // CHECK: %add = add <vscale x 16 x i8> %b, %a
  // CHECK-NEXT: ret <vscale x 16 x i8> %add
  return a += b;
}

svuint16_t add_inplace_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: add_inplace_u16
  // CHECK: %add = add <vscale x 8 x i16> %b, %a
  // CHECK-NEXT: ret <vscale x 8 x i16> %add
  return a += b;
}

svuint32_t add_inplace_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: add_inplace_u32
  // CHECK: %add = add <vscale x 4 x i32> %b, %a
  // CHECK-NEXT: ret <vscale x 4 x i32> %add
  return a += b;
}

svuint64_t add_inplace_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: add_inplace_u64
  // CHECK: %add = add <vscale x 2 x i64> %b, %a
  // CHECK-NEXT: ret <vscale x 2 x i64> %add
  return a += b;
}

svfloat16_t add_inplace_f16(svfloat16_t a, svfloat16_t b) {
  // CHECK-LABEL: add_inplace_f16
  // CHECK: %add = fadd <vscale x 8 x half> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x half> %add
  return a += b;
}

svfloat32_t add_inplace_f32(svfloat32_t a, svfloat32_t b) {
  // CHECK-LABEL: add_inplace_f32
  // CHECK: %add = fadd <vscale x 4 x float> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x float> %add
  return a += b;
}

svfloat64_t add_inplace_f64(svfloat64_t a, svfloat64_t b) {
  // CHECK-LABEL: add_inplace_f64
  // CHECK: %add = fadd <vscale x 2 x double> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x double> %add
  return a += b;
}

// SUBTRACTION

svint8_t sub_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: sub_i8
  // CHECK: %sub = sub <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %sub
  return a - b;
}

svint16_t sub_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: sub_i16
  // CHECK: %sub = sub <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %sub
  return a - b;
}

svint32_t sub_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: sub_i32
  // CHECK: %sub = sub <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %sub
  return a - b;
}

svint64_t sub_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: sub_i64
  // CHECK: %sub = sub <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %sub
  return a - b;
}

svuint8_t sub_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: sub_u8
  // CHECK: %sub = sub <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %sub
  return a - b;
}

svuint16_t sub_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: sub_u16
  // CHECK: %sub = sub <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %sub
  return a - b;
}

svuint32_t sub_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: sub_u32
  // CHECK: %sub = sub <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %sub
  return a - b;
}

svuint64_t sub_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: sub_u64
  // CHECK: %sub = sub <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %sub
  return a - b;
}

svfloat16_t sub_f16(svfloat16_t a, svfloat16_t b) {
  // CHECK-LABEL: sub_f16
  // CHECK: %sub = fsub <vscale x 8 x half> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x half> %sub
  return a - b;
}

svfloat32_t sub_f32(svfloat32_t a, svfloat32_t b) {
  // CHECK-LABEL: sub_f32
  // CHECK: %sub = fsub <vscale x 4 x float> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x float> %sub
  return a - b;
}

svfloat64_t sub_f64(svfloat64_t a, svfloat64_t b) {
  // CHECK-LABEL: sub_f64
  // CHECK: %sub = fsub <vscale x 2 x double> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x double> %sub
  return a - b;
}

svint8_t sub_inplace_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: sub_inplace_i8
  // CHECK: %sub = sub <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %sub
  return a - b;
}

svint16_t sub_inplace_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: sub_inplace_i16
  // CHECK: %sub = sub <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %sub
  return a - b;
}

svint32_t sub_inplace_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: sub_inplace_i32
  // CHECK: %sub = sub <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %sub
  return a - b;
}

svint64_t sub_inplace_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: sub_inplace_i64
  // CHECK: %sub = sub <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %sub
  return a - b;
}

svuint8_t sub_inplace_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: sub_inplace_u8
  // CHECK: %sub = sub <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %sub
  return a - b;
}

svuint16_t sub_inplace_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: sub_inplace_u16
  // CHECK: %sub = sub <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %sub
  return a - b;
}

svuint32_t sub_inplace_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: sub_inplace_u32
  // CHECK: %sub = sub <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %sub
  return a - b;
}

svuint64_t sub_inplace_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: sub_inplace_u64
  // CHECK: %sub = sub <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %sub
  return a - b;
}

svfloat16_t sub_inplace_f16(svfloat16_t a, svfloat16_t b) {
  // CHECK-LABEL: sub_inplace_f16
  // CHECK: %sub = fsub <vscale x 8 x half> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x half> %sub
  return a - b;
}

svfloat32_t sub_inplace_f32(svfloat32_t a, svfloat32_t b) {
  // CHECK-LABEL: sub_inplace_f32
  // CHECK: %sub = fsub <vscale x 4 x float> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x float> %sub
  return a - b;
}

svfloat64_t sub_inplace_f64(svfloat64_t a, svfloat64_t b) {
  // CHECK-LABEL: sub_inplace_f64
  // CHECK: %sub = fsub <vscale x 2 x double> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x double> %sub
  return a - b;
}

// MULTIPLICATION

svint8_t mul_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: mul_i8
  // CHECK: %mul = mul <vscale x 16 x i8> %b, %a
  // CHECK-NEXT: ret <vscale x 16 x i8> %mul
  return a * b;
}

svint16_t mul_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: mul_i16
  // CHECK: %mul = mul <vscale x 8 x i16> %b, %a
  // CHECK-NEXT: ret <vscale x 8 x i16> %mul
  return a * b;
}

svint32_t mul_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: mul_i32
  // CHECK: %mul = mul <vscale x 4 x i32> %b, %a
  // CHECK-NEXT: ret <vscale x 4 x i32> %mul
  return a * b;
}

svint64_t mul_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: mul_i64
  // CHECK: %mul = mul <vscale x 2 x i64> %b, %a
  // CHECK-NEXT: ret <vscale x 2 x i64> %mul
  return a * b;
}

svuint8_t mul_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: mul_u8
  // CHECK: %mul = mul <vscale x 16 x i8> %b, %a
  // CHECK-NEXT: ret <vscale x 16 x i8> %mul
  return a * b;
}

svuint16_t mul_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: mul_u16
  // CHECK: %mul = mul <vscale x 8 x i16> %b, %a
  // CHECK-NEXT: ret <vscale x 8 x i16> %mul
  return a * b;
}

svuint32_t mul_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: mul_u32
  // CHECK: %mul = mul <vscale x 4 x i32> %b, %a
  // CHECK-NEXT: ret <vscale x 4 x i32> %mul
  return a * b;
}

svuint64_t mul_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: mul_u64
  // CHECK: %mul = mul <vscale x 2 x i64> %b, %a
  // CHECK-NEXT: ret <vscale x 2 x i64> %mul
  return a * b;
}

svfloat16_t mul_f16(svfloat16_t a, svfloat16_t b) {
  // CHECK-LABEL: mul_f16
  // CHECK: %mul = fmul <vscale x 8 x half> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x half> %mul
  return a * b;
}

svfloat32_t mul_f32(svfloat32_t a, svfloat32_t b) {
  // CHECK-LABEL: mul_f32
  // CHECK: %mul = fmul <vscale x 4 x float> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x float> %mul
  return a * b;
}

svfloat64_t mul_f64(svfloat64_t a, svfloat64_t b) {
  // CHECK-LABEL: mul_f64
  // CHECK: %mul = fmul <vscale x 2 x double> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x double> %mul
  return a * b;
}

svint8_t mul_inplace_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: mul_inplace_i8
  // CHECK: %mul = mul <vscale x 16 x i8> %b, %a
  // CHECK-NEXT: ret <vscale x 16 x i8> %mul
  return a * b;
}

svint16_t mul_inplace_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: mul_inplace_i16
  // CHECK: %mul = mul <vscale x 8 x i16> %b, %a
  // CHECK-NEXT: ret <vscale x 8 x i16> %mul
  return a * b;
}

svint32_t mul_inplace_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: mul_inplace_i32
  // CHECK: %mul = mul <vscale x 4 x i32> %b, %a
  // CHECK-NEXT: ret <vscale x 4 x i32> %mul
  return a * b;
}

svint64_t mul_inplace_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: mul_inplace_i64
  // CHECK: %mul = mul <vscale x 2 x i64> %b, %a
  // CHECK-NEXT: ret <vscale x 2 x i64> %mul
  return a * b;
}

svuint8_t mul_inplace_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: mul_inplace_u8
  // CHECK: %mul = mul <vscale x 16 x i8> %b, %a
  // CHECK-NEXT: ret <vscale x 16 x i8> %mul
  return a * b;
}

svuint16_t mul_inplace_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: mul_inplace_u16
  // CHECK: %mul = mul <vscale x 8 x i16> %b, %a
  // CHECK-NEXT: ret <vscale x 8 x i16> %mul
  return a * b;
}

svuint32_t mul_inplace_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: mul_inplace_u32
  // CHECK: %mul = mul <vscale x 4 x i32> %b, %a
  // CHECK-NEXT: ret <vscale x 4 x i32> %mul
  return a * b;
}

svuint64_t mul_inplace_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: mul_inplace_u64
  // CHECK: %mul = mul <vscale x 2 x i64> %b, %a
  // CHECK-NEXT: ret <vscale x 2 x i64> %mul
  return a * b;
}

svfloat16_t mul_inplace_f16(svfloat16_t a, svfloat16_t b) {
  // CHECK-LABEL: mul_inplace_f16
  // CHECK: %mul = fmul <vscale x 8 x half> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x half> %mul
  return a * b;
}

svfloat32_t mul_inplace_f32(svfloat32_t a, svfloat32_t b) {
  // CHECK-LABEL: mul_inplace_f32
  // CHECK: %mul = fmul <vscale x 4 x float> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x float> %mul
  return a * b;
}

svfloat64_t mul_inplace_f64(svfloat64_t a, svfloat64_t b) {
  // CHECK-LABEL: mul_inplace_f64
  // CHECK: %mul = fmul <vscale x 2 x double> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x double> %mul
  return a * b;
}

// DIVISION

svint8_t div_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: div_i8
  // CHECK: %div = sdiv <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %div
  return a / b;
}

svint16_t div_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: div_i16
  // CHECK: %div = sdiv <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %div
  return a / b;
}

svint32_t div_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: div_i32
  // CHECK: %div = sdiv <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %div
  return a / b;
}

svint64_t div_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: div_i64
  // CHECK: %div = sdiv <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %div
  return a / b;
}

svuint8_t div_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: div_u8
  // CHECK: %div = udiv <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %div
  return a / b;
}

svuint16_t div_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: div_u16
  // CHECK: %div = udiv <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %div
  return a / b;
}

svuint32_t div_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: div_u32
  // CHECK: %div = udiv <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %div
  return a / b;
}

svuint64_t div_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: div_u64
  // CHECK: %div = udiv <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %div
  return a / b;
}

svfloat16_t div_f16(svfloat16_t a, svfloat16_t b) {
  // CHECK-LABEL: div_f16
  // CHECK: %div = fdiv <vscale x 8 x half> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x half> %div
  return a / b;
}

svfloat32_t div_f32(svfloat32_t a, svfloat32_t b) {
  // CHECK-LABEL: div_f32
  // CHECK: %div = fdiv <vscale x 4 x float> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x float> %div
  return a / b;
}

svfloat64_t div_f64(svfloat64_t a, svfloat64_t b) {
  // CHECK-LABEL: div_f64
  // CHECK: %div = fdiv <vscale x 2 x double> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x double> %div
  return a / b;
}

svint8_t div_inplace_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: div_inplace_i8
  // CHECK: %div = sdiv <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %div
  return a / b;
}

svint16_t div_inplace_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: div_inplace_i16
  // CHECK: %div = sdiv <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %div
  return a / b;
}

svint32_t div_inplace_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: div_inplace_i32
  // CHECK: %div = sdiv <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %div
  return a / b;
}

svint64_t div_inplace_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: div_inplace_i64
  // CHECK: %div = sdiv <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %div
  return a / b;
}

svuint8_t div_inplace_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: div_inplace_u8
  // CHECK: %div = udiv <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %div
  return a / b;
}

svuint16_t div_inplace_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: div_inplace_u16
  // CHECK: %div = udiv <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %div
  return a / b;
}

svuint32_t div_inplace_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: div_inplace_u32
  // CHECK: %div = udiv <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %div
  return a / b;
}

svuint64_t div_inplace_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: div_inplace_u64
  // CHECK: %div = udiv <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %div
  return a / b;
}

svfloat16_t div_inplace_f16(svfloat16_t a, svfloat16_t b) {
  // CHECK-LABEL: div_inplace_f16
  // CHECK: %div = fdiv <vscale x 8 x half> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x half> %div
  return a / b;
}

svfloat32_t div_inplace_f32(svfloat32_t a, svfloat32_t b) {
  // CHECK-LABEL: div_inplace_f32
  // CHECK: %div = fdiv <vscale x 4 x float> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x float> %div
  return a / b;
}

svfloat64_t div_inplace_f64(svfloat64_t a, svfloat64_t b) {
  // CHECK-LABEL: div_inplace_f64
  // CHECK: %div = fdiv <vscale x 2 x double> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x double> %div
  return a / b;
}

// REMAINDER

svint8_t rem_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: rem_i8
  // CHECK: %rem = srem <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %rem
  return a % b;
}

svint16_t rem_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: rem_i16
  // CHECK: %rem = srem <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %rem
  return a % b;
}

svint32_t rem_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: rem_i32
  // CHECK: %rem = srem <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %rem
  return a % b;
}

svint64_t rem_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: rem_i64
  // CHECK: %rem = srem <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %rem
  return a % b;
}

svuint8_t rem_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: rem_u8
  // CHECK: %rem = urem <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %rem
  return a % b;
}

svuint16_t rem_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: rem_u16
  // CHECK: %rem = urem <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %rem
  return a % b;
}

svuint32_t rem_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: rem_u32
  // CHECK: %rem = urem <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %rem
  return a % b;
}

svuint64_t rem_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: rem_u64
  // CHECK: %rem = urem <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %rem
  return a % b;
}

svint8_t rem_inplace_i8(svint8_t a, svint8_t b) {
  // CHECK-LABEL: rem_inplace_i8
  // CHECK: %rem = srem <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %rem
  return a % b;
}

svint16_t rem_inplace_i16(svint16_t a, svint16_t b) {
  // CHECK-LABEL: rem_inplace_i16
  // CHECK: %rem = srem <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %rem
  return a % b;
}

svint32_t rem_inplace_i32(svint32_t a, svint32_t b) {
  // CHECK-LABEL: rem_inplace_i32
  // CHECK: %rem = srem <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %rem
  return a % b;
}

svint64_t rem_inplace_i64(svint64_t a, svint64_t b) {
  // CHECK-LABEL: rem_inplace_i64
  // CHECK: %rem = srem <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %rem
  return a % b;
}

svuint8_t rem_inplace_u8(svuint8_t a, svuint8_t b) {
  // CHECK-LABEL: rem_inplace_u8
  // CHECK: %rem = urem <vscale x 16 x i8> %a, %b
  // CHECK-NEXT: ret <vscale x 16 x i8> %rem
  return a % b;
}

svuint16_t rem_inplace_u16(svuint16_t a, svuint16_t b) {
  // CHECK-LABEL: rem_inplace_u16
  // CHECK: %rem = urem <vscale x 8 x i16> %a, %b
  // CHECK-NEXT: ret <vscale x 8 x i16> %rem
  return a % b;
}

svuint32_t rem_inplace_u32(svuint32_t a, svuint32_t b) {
  // CHECK-LABEL: rem_inplace_u32
  // CHECK: %rem = urem <vscale x 4 x i32> %a, %b
  // CHECK-NEXT: ret <vscale x 4 x i32> %rem
  return a % b;
}

svuint64_t rem_inplace_u64(svuint64_t a, svuint64_t b) {
  // CHECK-LABEL: rem_inplace_u64
  // CHECK: %rem = urem <vscale x 2 x i64> %a, %b
  // CHECK-NEXT: ret <vscale x 2 x i64> %rem
  return a % b;
}
