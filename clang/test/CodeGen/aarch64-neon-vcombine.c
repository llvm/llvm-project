// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int8x16_t test_vcombine_s8(int8x8_t low, int8x8_t high) {
  // CHECK-LABEL: test_vcombine_s8:
  return vcombine_s8(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

int16x8_t test_vcombine_s16(int16x4_t low, int16x4_t high) {
  // CHECK-LABEL: test_vcombine_s16:
  return vcombine_s16(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

int32x4_t test_vcombine_s32(int32x2_t low, int32x2_t high) {
  // CHECK-LABEL: test_vcombine_s32:
  return vcombine_s32(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

int64x2_t test_vcombine_s64(int64x1_t low, int64x1_t high) {
  // CHECK-LABEL: test_vcombine_s64:
  return vcombine_s64(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

uint8x16_t test_vcombine_u8(uint8x8_t low, uint8x8_t high) {
  // CHECK-LABEL: test_vcombine_u8:
  return vcombine_u8(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

uint16x8_t test_vcombine_u16(uint16x4_t low, uint16x4_t high) {
  // CHECK-LABEL: test_vcombine_u16:
  return vcombine_u16(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

uint32x4_t test_vcombine_u32(uint32x2_t low, uint32x2_t high) {
  // CHECK-LABEL: test_vcombine_u32:
  return vcombine_u32(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

uint64x2_t test_vcombine_u64(uint64x1_t low, uint64x1_t high) {
  // CHECK-LABEL: test_vcombine_u64:
  return vcombine_u64(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

poly64x2_t test_vcombine_p64(poly64x1_t low, poly64x1_t high) {
  // CHECK-LABEL: test_vcombine_p64:
  return vcombine_p64(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

float16x8_t test_vcombine_f16(float16x4_t low, float16x4_t high) {
  // CHECK-LABEL: test_vcombine_f16:
  return vcombine_f16(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

float32x4_t test_vcombine_f32(float32x2_t low, float32x2_t high) {
  // CHECK-LABEL: test_vcombine_f32:
  return vcombine_f32(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

poly8x16_t test_vcombine_p8(poly8x8_t low, poly8x8_t high) {
  // CHECK-LABEL: test_vcombine_p8:
  return vcombine_p8(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

poly16x8_t test_vcombine_p16(poly16x4_t low, poly16x4_t high) {
  // CHECK-LABEL: test_vcombine_p16:
  return vcombine_p16(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}

float64x2_t test_vcombine_f64(float64x1_t low, float64x1_t high) {
  // CHECK-LABEL: test_vcombine_f64:
  return vcombine_f64(low, high);
  // CHECK: ins	v0.d[1], v1.d[0]
}
