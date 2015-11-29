// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon \
// RUN:  -target-feature +v8.1a -O3 -S -o - %s \
// RUN:  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AARCH64

 #include <arm_neon.h>

// CHECK-AARCH64-LABEL: test_vqrdmlah_laneq_s16
int16x4_t test_vqrdmlah_laneq_s16(int16x4_t a, int16x4_t b, int16x8_t v) {
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
  return vqrdmlah_laneq_s16(a, b, v, 7);
}

// CHECK-AARCH64-LABEL: test_vqrdmlah_laneq_s32
int32x2_t test_vqrdmlah_laneq_s32(int32x2_t a, int32x2_t b, int32x4_t v) {
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
  return vqrdmlah_laneq_s32(a, b, v, 3);
}

// CHECK-AARCH64-LABEL: test_vqrdmlahq_laneq_s16
int16x8_t test_vqrdmlahq_laneq_s16(int16x8_t a, int16x8_t b, int16x8_t v) {
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
  return vqrdmlahq_laneq_s16(a, b, v, 7);
}

// CHECK-AARCH64-LABEL: test_vqrdmlahq_laneq_s32
int32x4_t test_vqrdmlahq_laneq_s32(int32x4_t a, int32x4_t b, int32x4_t v) {
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
  return vqrdmlahq_laneq_s32(a, b, v, 3);
}

// CHECK-AARCH64-LABEL: test_vqrdmlahh_s16
int16_t test_vqrdmlahh_s16(int16_t a, int16_t b, int16_t c) {
// CHECK-AARCH64: sqrdmlah {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
  return vqrdmlahh_s16(a, b, c);
}

// CHECK-AARCH64-LABEL: test_vqrdmlahs_s32
int32_t test_vqrdmlahs_s32(int32_t a, int32_t b, int32_t c) {
// CHECK-AARCH64: sqrdmlah {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  return vqrdmlahs_s32(a, b, c);
}

// CHECK-AARCH64-LABEL: test_vqrdmlahh_lane_s16
int16_t test_vqrdmlahh_lane_s16(int16_t a, int16_t b, int16x4_t c) {
// CHECK-AARCH64: sqrdmlah {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[3]
  return vqrdmlahh_lane_s16(a, b, c, 3);
}

// CHECK-AARCH64-LABEL: test_vqrdmlahs_lane_s32
int32_t test_vqrdmlahs_lane_s32(int32_t a, int32_t b, int32x2_t c) {
// CHECK-AARCH64: sqrdmlah {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  return vqrdmlahs_lane_s32(a, b, c, 1);
}

// CHECK-AARCH64-LABEL: test_vqrdmlahh_laneq_s16
int16_t test_vqrdmlahh_laneq_s16(int16_t a, int16_t b, int16x8_t c) {
// CHECK-AARCH64: sqrdmlah {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[7]
  return vqrdmlahh_laneq_s16(a, b, c, 7);
}

// CHECK-AARCH64-LABEL: test_vqrdmlahs_laneq_s32
int32_t test_vqrdmlahs_laneq_s32(int32_t a, int32_t b, int32x4_t c) {
// CHECK-AARCH64: sqrdmlah {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  return vqrdmlahs_laneq_s32(a, b, c, 3);
}

// CHECK-AARCH64-LABEL: test_vqrdmlsh_laneq_s16
int16x4_t test_vqrdmlsh_laneq_s16(int16x4_t a, int16x4_t b, int16x8_t v) {
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
  return vqrdmlsh_laneq_s16(a, b, v, 7);
}

// CHECK-AARCH64-LABEL: test_vqrdmlsh_laneq_s32
int32x2_t test_vqrdmlsh_laneq_s32(int32x2_t a, int32x2_t b, int32x4_t v) {
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
  return vqrdmlsh_laneq_s32(a, b, v, 3);
}

// CHECK-AARCH64-LABEL: test_vqrdmlshq_laneq_s16
int16x8_t test_vqrdmlshq_laneq_s16(int16x8_t a, int16x8_t b, int16x8_t v) {
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
  return vqrdmlshq_laneq_s16(a, b, v, 7);
}

// CHECK-AARCH64-LABEL: test_vqrdmlshq_laneq_s32
int32x4_t test_vqrdmlshq_laneq_s32(int32x4_t a, int32x4_t b, int32x4_t v) {
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
  return vqrdmlshq_laneq_s32(a, b, v, 3);
}

// CHECK-AARCH64-LABEL: test_vqrdmlshh_s16
int16_t test_vqrdmlshh_s16(int16_t a, int16_t b, int16_t c) {
// CHECK-AARCH64: sqrdmlsh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
  return vqrdmlshh_s16(a, b, c);
}

// CHECK-AARCH64-LABEL: test_vqrdmlshs_s32
int32_t test_vqrdmlshs_s32(int32_t a, int32_t b, int32_t c) {
// CHECK-AARCH64: sqrdmlsh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  return vqrdmlshs_s32(a, b, c);
}

// CHECK-AARCH64-LABEL: test_vqrdmlshh_lane_s16
int16_t test_vqrdmlshh_lane_s16(int16_t a, int16_t b, int16x4_t c) {
// CHECK-AARCH64: sqrdmlsh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[3]
  return vqrdmlshh_lane_s16(a, b, c, 3);
}

// CHECK-AARCH64-LABEL: test_vqrdmlshs_lane_s32
int32_t test_vqrdmlshs_lane_s32(int32_t a, int32_t b, int32x2_t c) {
// CHECK-AARCH64: sqrdmlsh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  return vqrdmlshs_lane_s32(a, b, c, 1);
}

// CHECK-AARCH64-LABEL: test_vqrdmlshh_laneq_s16
int16_t test_vqrdmlshh_laneq_s16(int16_t a, int16_t b, int16x8_t c) {
// CHECK-AARCH64: sqrdmlsh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[7]
  return vqrdmlshh_laneq_s16(a, b, c, 7);
}

// CHECK-AARCH64-LABEL: test_vqrdmlshs_laneq_s32
int32_t test_vqrdmlshs_laneq_s32(int32_t a, int32_t b, int32x4_t c) {
// CHECK-AARCH64: sqrdmlsh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  return vqrdmlshs_laneq_s32(a, b, c, 3);
}

