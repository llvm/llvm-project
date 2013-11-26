// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int8x8_t test_vget_high_s8(int8x16_t a) {
  // CHECK-LABEL: test_vget_high_s8:
  return vget_high_s8(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

int16x4_t test_vget_high_s16(int16x8_t a) {
  // CHECK-LABEL: test_vget_high_s16:
  return vget_high_s16(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

int32x2_t test_vget_high_s32(int32x4_t a) {
  // CHECK-LABEL: test_vget_high_s32:
  return vget_high_s32(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

int64x1_t test_vget_high_s64(int64x2_t a) {
  // CHECK-LABEL: test_vget_high_s64:
  return vget_high_s64(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

uint8x8_t test_vget_high_u8(uint8x16_t a) {
  // CHECK-LABEL: test_vget_high_u8:
  return vget_high_u8(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

uint16x4_t test_vget_high_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vget_high_u16:
  return vget_high_u16(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

uint32x2_t test_vget_high_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vget_high_u32:
  return vget_high_u32(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

uint64x1_t test_vget_high_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vget_high_u64:
  return vget_high_u64(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

poly64x1_t test_vget_high_p64(poly64x2_t a) {
  // CHECK-LABEL: test_vget_high_p64:
  return vget_high_p64(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

float16x4_t test_vget_high_f16(float16x8_t a) {
  // CHECK-LABEL: test_vget_high_f16:
  return vget_high_f16(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

float32x2_t test_vget_high_f32(float32x4_t a) {
  // CHECK-LABEL: test_vget_high_f32:
  return vget_high_f32(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

poly8x8_t test_vget_high_p8(poly8x16_t a) {
  // CHECK-LABEL: test_vget_high_p8:
  return vget_high_p8(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

poly16x4_t test_vget_high_p16(poly16x8_t a) {
  // CHECK-LABEL: test_vget_high_p16
  return vget_high_p16(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

float64x1_t test_vget_high_f64(float64x2_t a) {
  // CHECK-LABEL: test_vget_high_f64
  return vget_high_f64(a);
  // CHECK: dup d0, {{v[0-9]+}}.d[1]
}

int8x8_t test_vget_low_s8(int8x16_t a) {
  // CHECK-LABEL: test_vget_low_s8:
  return vget_low_s8(a);
  // CHECK-NEXT: ret
}

int16x4_t test_vget_low_s16(int16x8_t a) {
  // CHECK-LABEL: test_vget_low_s16:
  return vget_low_s16(a);
  // CHECK-NEXT: ret
}

int32x2_t test_vget_low_s32(int32x4_t a) {
  // CHECK-LABEL: test_vget_low_s32:
  return vget_low_s32(a);
  // CHECK-NEXT: ret
}

int64x1_t test_vget_low_s64(int64x2_t a) {
  // CHECK-LABEL: test_vget_low_s64:
  return vget_low_s64(a);
  // CHECK-NEXT: ret
}

uint8x8_t test_vget_low_u8(uint8x16_t a) {
  // CHECK-LABEL: test_vget_low_u8:
  return vget_low_u8(a);
  // CHECK-NEXT: ret
}

uint16x4_t test_vget_low_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vget_low_u16:
  return vget_low_u16(a);
  // CHECK-NEXT: ret
}

uint32x2_t test_vget_low_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vget_low_u32:
  return vget_low_u32(a);
  // CHECK-NEXT: ret
}

uint64x1_t test_vget_low_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vget_low_u64:
  return vget_low_u64(a);
  // CHECK-NEXT: ret
}

poly64x1_t test_vget_low_p64(poly64x2_t a) {
  // CHECK-LABEL: test_vget_low_p64:
  return vget_low_p64(a);
  // CHECK-NEXT: ret
}

float16x4_t test_vget_low_f16(float16x8_t a) {
  // CHECK-LABEL: test_vget_low_f16:
  return vget_low_f16(a);
  // CHECK-NEXT: ret
}

float32x2_t test_vget_low_f32(float32x4_t a) {
  // CHECK-LABEL: test_vget_low_f32:
  return vget_low_f32(a);
  // CHECK-NEXT: ret
}

poly8x8_t test_vget_low_p8(poly8x16_t a) {
  // CHECK-LABEL: test_vget_low_p8:
  return vget_low_p8(a);
  // CHECK-NEXT: ret
}

poly16x4_t test_vget_low_p16(poly16x8_t a) {
  // CHECK-LABEL: test_vget_low_p16:
  return vget_low_p16(a);
  // CHECK-NEXT: ret
}

float64x1_t test_vget_low_f64(float64x2_t a) {
  // CHECK-LABEL: test_vget_low_f64:
  return vget_low_f64(a);
  // CHECK-NEXT: ret
}

