// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s


#include <arm_neon.h>

// CHECK: test_vdups_lane_f32
float32_t test_vdups_lane_f32(float32x2_t a) {
  return vdups_lane_f32(a, 1);
// CHECK: ret
// CHECK-NOT: dup {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}


// CHECK: test_vdupd_lane_f64
float64_t test_vdupd_lane_f64(float64x1_t a) {
  return vdupd_lane_f64(a, 0);
// CHECK: ret
// CHECK-NOT: dup {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}


// CHECK: test_vdups_laneq_f32
float32_t test_vdups_laneq_f32(float32x4_t a) {
  return vdups_laneq_f32(a, 3);
// CHECK: ret
// CHECK-NOT: dup {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}


// CHECK: test_vdupd_laneq_f64
float64_t test_vdupd_laneq_f64(float64x2_t a) {
  return vdupd_laneq_f64(a, 1);
// CHECK: ret
// CHECK-NOT: dup {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}


// CHECK: test_vdupb_lane_s8
int8_t test_vdupb_lane_s8(int8x8_t a) {
  return vdupb_lane_s8(a, 7);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[7]
}


// CHECK: test_vduph_lane_s16
int16_t test_vduph_lane_s16(int16x4_t a) {
  return vduph_lane_s16(a, 3);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[3]
}


// CHECK: test_vdups_lane_s32
int32_t test_vdups_lane_s32(int32x2_t a) {
  return vdups_lane_s32(a, 1);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.s[1]
}


// CHECK: test_vdupd_lane_s64
int64_t test_vdupd_lane_s64(int64x1_t a) {
  return vdupd_lane_s64(a, 0);
// CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
}


// CHECK: test_vdupb_lane_u8
uint8_t test_vdupb_lane_u8(uint8x8_t a) {
  return vdupb_lane_u8(a, 7);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[7]
}


// CHECK: test_vduph_lane_u16
uint16_t test_vduph_lane_u16(uint16x4_t a) {
  return vduph_lane_u16(a, 3);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[3]
}


// CHECK: test_vdups_lane_u32
uint32_t test_vdups_lane_u32(uint32x2_t a) {
  return vdups_lane_u32(a, 1);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.s[1]
}


// CHECK: test_vdupd_lane_u64
uint64_t test_vdupd_lane_u64(uint64x1_t a) {
  return vdupd_lane_u64(a, 0);
// CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vdupb_laneq_s8
int8_t test_vdupb_laneq_s8(int8x16_t a) {
  return vdupb_laneq_s8(a, 15);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[15]
}


// CHECK: test_vduph_laneq_s16
int16_t test_vduph_laneq_s16(int16x8_t a) {
  return vduph_laneq_s16(a, 7);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[7]
}


// CHECK: test_vdups_laneq_s32
int32_t test_vdups_laneq_s32(int32x4_t a) {
  return vdups_laneq_s32(a, 3);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.s[3]
}


// CHECK: test_vdupd_laneq_s64
int64_t test_vdupd_laneq_s64(int64x2_t a) {
  return vdupd_laneq_s64(a, 1);
// CHECK: umov {{x[0-9]+}}, {{v[0-9]+}}.d[1]
}


// CHECK: test_vdupb_laneq_u8
uint8_t test_vdupb_laneq_u8(uint8x16_t a) {
  return vdupb_laneq_u8(a, 15);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[15]
}


// CHECK: test_vduph_laneq_u16
uint16_t test_vduph_laneq_u16(uint16x8_t a) {
  return vduph_laneq_u16(a, 7);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[7]
}


// CHECK: test_vdups_laneq_u32
uint32_t test_vdups_laneq_u32(uint32x4_t a) {
  return vdups_laneq_u32(a, 3);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.s[3]
}


// CHECK: test_vdupd_laneq_u64
uint64_t test_vdupd_laneq_u64(uint64x2_t a) {
  return vdupd_laneq_u64(a, 1);
// CHECK: umov {{x[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK: test_vdupb_lane_p8
poly8_t test_vdupb_lane_p8(poly8x8_t a) {
  return vdupb_lane_p8(a, 7);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[7]
}

// CHECK: test_vduph_lane_p16
poly16_t test_vduph_lane_p16(poly16x4_t a) {
  return vduph_lane_p16(a, 3);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[3]
}

// CHECK: test_vdupb_laneq_p8
poly8_t test_vdupb_laneq_p8(poly8x16_t a) {
  return vdupb_laneq_p8(a, 15);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[15]
}

// CHECK: test_vduph_laneq_p16
poly16_t test_vduph_laneq_p16(poly16x8_t a) {
  return vduph_laneq_p16(a, 7);
// CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[7]
}

