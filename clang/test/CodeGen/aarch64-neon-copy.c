// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

uint8x8_t test_vset_lane_u8(uint8_t v1, uint8x8_t v2) {
   // CHECK: test_vset_lane_u8
  return vset_lane_u8(v1, v2, 6);
  // CHECK: ins {{v[0-9]+}}.b[6], {{w[0-9]+}}
}

uint16x4_t test_vset_lane_u16(uint16_t v1, uint16x4_t v2) {
   // CHECK: test_vset_lane_u16
  return vset_lane_u16(v1, v2, 2);
  // CHECK: ins {{v[0-9]+}}.h[2], {{w[0-9]+}}
}

uint32x2_t test_vset_lane_u32(uint32_t v1, uint32x2_t v2) {
   // CHECK: test_vset_lane_u32
  return vset_lane_u32(v1, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[1], {{w[0-9]+}}
}
uint64x1_t test_vset_lane_u64(uint64_t v1, uint64x1_t v2) {
   // CHECK: test_vset_lane_u64
  return vset_lane_u64(v1, v2, 0);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

int8x8_t test_vset_lane_s8(int8_t v1, int8x8_t v2) {
   // CHECK: test_vset_lane_s8
  return vset_lane_s8(v1, v2, 6);
  // CHECK: ins {{v[0-9]+}}.b[6], {{w[0-9]+}}
}

int16x4_t test_vset_lane_s16(int16_t v1, int16x4_t v2) {
   // CHECK: test_vset_lane_s16
  return vset_lane_s16(v1, v2, 2);
  // CHECK: ins {{v[0-9]+}}.h[2], {{w[0-9]+}}
}

int32x2_t test_vset_lane_s32(int32_t v1, int32x2_t v2) {
   // CHECK: test_vset_lane_s32
  return vset_lane_s32(v1, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[1], {{w[0-9]+}}
}

  int64x1_t test_vset_lane_s64(int64_t v1, int64x1_t v2) {
   // CHECK: test_vset_lane_s64
  return vset_lane_s64(v1, v2, 0);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

uint8x16_t test_vsetq_lane_u8(uint8_t v1, uint8x16_t v2) {
   // CHECK: test_vsetq_lane_u8
  return vsetq_lane_u8(v1, v2, 6);
  // CHECK: ins {{v[0-9]+}}.b[6], {{w[0-9]+}}
}

uint16x8_t test_vsetq_lane_u16(uint16_t v1, uint16x8_t v2) {
   // CHECK: test_vsetq_lane_u16
  return vsetq_lane_u16(v1, v2, 2);
  // CHECK: ins {{v[0-9]+}}.h[2], {{w[0-9]+}}
}

uint32x4_t test_vsetq_lane_u32(uint32_t v1, uint32x4_t v2) {
   // CHECK: test_vsetq_lane_u32
  return vsetq_lane_u32(v1, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[1], {{w[0-9]+}}
}

  uint64x2_t test_vsetq_lane_u64(uint64_t v1, uint64x2_t v2) {
   // CHECK: test_vsetq_lane_u64
  return vsetq_lane_u64(v1, v2, 1);
  // CHECK: ins {{v[0-9]+}}.d[1], {{x[0-9]+}}
}

int8x16_t test_vsetq_lane_s8(int8_t v1, int8x16_t v2) {
   // CHECK: test_vsetq_lane_s8
  return vsetq_lane_s8(v1, v2, 6);
  // CHECK: ins {{v[0-9]+}}.b[6], {{w[0-9]+}}
}

int16x8_t test_vsetq_lane_s16(int16_t v1, int16x8_t v2) {
   // CHECK: test_vsetq_lane_s16
  return vsetq_lane_s16(v1, v2, 2);
  // CHECK: ins {{v[0-9]+}}.h[2], {{w[0-9]+}}
}

int32x4_t test_vsetq_lane_s32(int32_t v1, int32x4_t v2) {
   // CHECK: test_vsetq_lane_s32
  return vsetq_lane_s32(v1, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[1], {{w[0-9]+}}
}

int64x2_t test_vsetq_lane_s64(int64_t v1, int64x2_t v2) {
   // CHECK: test_vsetq_lane_s64
  return vsetq_lane_s64(v1, v2, 0);
  // CHECK: ins {{v[0-9]+}}.d[0], {{x[0-9]+}}
}

poly8x8_t test_vset_lane_p8(poly8_t v1, poly8x8_t v2) {
   // CHECK: test_vset_lane_p8
  return vset_lane_p8(v1, v2, 6);
  // CHECK: ins {{v[0-9]+}}.b[6], {{w[0-9]+}}
}

poly16x4_t test_vset_lane_p16(poly16_t v1, poly16x4_t v2) {
   // CHECK: test_vset_lane_p16
  return vset_lane_p16(v1, v2, 2);
  // CHECK: ins {{v[0-9]+}}.h[2], {{w[0-9]+}}
}

poly8x16_t test_vsetq_lane_p8(poly8_t v1, poly8x16_t v2) {
   // CHECK: test_vsetq_lane_p8
  return vsetq_lane_p8(v1, v2, 6);
  // CHECK: ins {{v[0-9]+}}.b[6], {{w[0-9]+}}
}

poly16x8_t test_vsetq_lane_p16(poly16_t v1, poly16x8_t v2) {
   // CHECK: test_vsetq_lane_p16
  return vsetq_lane_p16(v1, v2, 2);
  // CHECK: ins {{v[0-9]+}}.h[2], {{w[0-9]+}}
}

float32x2_t test_vset_lane_f32(float32_t v1, float32x2_t v2) {
   // CHECK: test_vset_lane_f32
  return vset_lane_f32(v1, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
}

float32x4_t test_vsetq_lane_f32(float32_t v1, float32x4_t v2) {
   // CHECK: test_vsetq_lane_f32
  return vsetq_lane_f32(v1, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
}

float64x1_t test_vset_lane_f64(float64_t v1, float64x1_t v2) {
   // CHECK: test_vset_lane_f64
  return vset_lane_f64(v1, v2, 0);
  // CHECK: ret
}

float64x2_t test_vsetq_lane_f64(float64_t v1, float64x2_t v2) {
   // CHECK: test_vsetq_lane_f64
  return vsetq_lane_f64(v1, v2, 0);
  // CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[0]
}

uint8_t test_vget_lane_u8(uint8x8_t v1) {
  // CHECK: test_vget_lane_u8
  return vget_lane_u8(v1, 7);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[7]
}

uint16_t test_vget_lane_u16(uint16x4_t v1) {
  // CHECK: test_vget_lane_u16
  return vget_lane_u16(v1, 3);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[3]
}

uint32_t test_vget_lane_u32(uint32x2_t v1) {
  // CHECK: test_vget_lane_u32
  return vget_lane_u32(v1, 1);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.s[1]
}

uint64_t test_vget_lane_u64(uint64x1_t v1) {
  // CHECK: test_vget_lane_u64
  return vget_lane_u64(v1, 0);
  // CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
}

uint8_t test_vgetq_lane_u8(uint8x16_t v1) {
  // CHECK: test_vgetq_lane_u8
  return vgetq_lane_u8(v1, 15);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[15]
}

uint16_t test_vgetq_lane_u16(uint16x8_t v1) {
  // CHECK: test_vgetq_lane_u16
  return vgetq_lane_u16(v1, 6);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[6]
}

uint32_t test_vgetq_lane_u32(uint32x4_t v1) {
  // CHECK: test_vgetq_lane_u32
  return vgetq_lane_u32(v1, 2);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.s[2]
}

uint64_t test_vgetq_lane_u64(uint64x2_t v1) {
  // CHECK: test_vgetq_lane_u64
  return vgetq_lane_u64(v1, 1);
  // CHECK: umov {{x[0-9]+}}, {{v[0-9]+}}.d[1]
}

poly8_t test_vget_lane_p8(poly8x8_t v1) {
  // CHECK: test_vget_lane_p8
  return vget_lane_p8(v1, 7);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[7]
}

poly16_t test_vget_lane_p16(poly16x4_t v1) {
  // CHECK: test_vget_lane_p16
  return vget_lane_p16(v1, 3);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[3]
}

poly8_t test_vgetq_lane_p8(poly8x16_t v1) {
  // CHECK: test_vgetq_lane_p8
  return vgetq_lane_p8(v1, 14);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[14]
}

poly16_t test_vgetq_lane_p16(poly16x8_t v1) {
  // CHECK: test_vgetq_lane_p16
  return vgetq_lane_p16(v1, 6);
  // CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[6]
}

int32_t test_vget_lane_s8(int8x8_t v1) {
  // CHECK: test_vget_lane_s8
  return vget_lane_s8(v1, 7)+1;
  // CHECK: smov {{w[0-9]+}}, {{v[0-9]+}}.b[7]
}

int32_t test_vget_lane_s16(int16x4_t v1) {
  // CHECK: test_vget_lane_s16
  return vget_lane_s16(v1, 3)+1;
  // CHECK: smov {{w[0-9]+}}, {{v[0-9]+}}.h[3]
}

int64_t test_vget_lane_s32(int32x2_t v1) {
  // CHECK: test_vget_lane_s32
  return vget_lane_s32(v1, 1);
  // CHECK: smov {{x[0-9]+}}, {{v[0-9]+}}.s[1]
}

int64_t test_vget_lane_s64(int64x1_t v1) {
  // CHECK: test_vget_lane_s64
  return vget_lane_s64(v1, 0);
  // CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
}

int32_t test_vgetq_lane_s8(int8x16_t v1) {
  // CHECK: test_vgetq_lane_s8
  return vgetq_lane_s8(v1, 15)+1;
  // CHECK: smov {{w[0-9]+}}, {{v[0-9]+}}.b[15]
}

int32_t test_vgetq_lane_s16(int16x8_t v1) {
  // CHECK: test_vgetq_lane_s16
  return vgetq_lane_s16(v1, 6)+1;
  // CHECK: smov {{w[0-9]+}}, {{v[0-9]+}}.h[6]
}

int64_t test_vgetq_lane_s32(int32x4_t v1) {
  // CHECK: test_vgetq_lane_s32
  return vgetq_lane_s32(v1, 2);
  // CHECK: smov {{x[0-9]+}}, {{v[0-9]+}}.s[2]
}

int64_t test_vgetq_lane_s64(int64x2_t v1) {
  // CHECK: test_vgetq_lane_s64
  return vgetq_lane_s64(v1, 1);
  // CHECK: umov {{x[0-9]+}}, {{v[0-9]+}}.d[1]
}

int8x8_t test_vcopy_lane_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK: test_vcopy_lane_s8
  return vcopy_lane_s8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

int16x4_t test_vcopy_lane_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK: test_vcopy_lane_s16
  return vcopy_lane_s16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

poly8x8_t test_vcopy_lane_p8(poly8x8_t v1, poly8x8_t v2) {
  // CHECK: test_vcopy_lane_p8
  return vcopy_lane_p8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

poly16x4_t test_vcopy_lane_p16(poly16x4_t v1, poly16x4_t v2) {
  // CHECK: test_vcopy_lane_p16
  return vcopy_lane_p16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

int32x2_t test_vcopy_lane_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK: test_vcopy_lane_s32
  return vcopy_lane_s32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

float32x2_t test_vcopy_lane_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vcopy_lane_f32
  return vcopy_lane_f32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

uint8x8_t test_vcopy_lane_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK: test_vcopy_lane_u8
  return vcopy_lane_u8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

uint16x4_t test_vcopy_lane_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK: test_vcopy_lane_u16
  return vcopy_lane_u16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

uint32x2_t test_vcopy_lane_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK: test_vcopy_lane_u32
  return vcopy_lane_u32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

int8x8_t test_vcopy_laneq_s8(int8x8_t v1, int8x16_t v2) {
  // CHECK: test_vcopy_laneq_s8
  return vcopy_laneq_s8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

int16x4_t test_vcopy_laneq_s16(int16x4_t v1, int16x8_t v2) {
  // CHECK: test_vcopy_laneq_s16
  return vcopy_laneq_s16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

poly8x8_t test_vcopy_laneq_p8(poly8x8_t v1, poly8x16_t v2) {
  // CHECK: test_vcopy_laneq_p8
  return vcopy_laneq_p8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

poly16x4_t test_vcopy_laneq_p16(poly16x4_t v1, poly16x8_t v2) {
  // CHECK: test_vcopy_laneq_p16
  return vcopy_laneq_p16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

int32x2_t test_vcopy_laneq_s32(int32x2_t v1, int32x4_t v2) {
  // CHECK: test_vcopy_laneq_s32
  return vcopy_laneq_s32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

float32x2_t test_vcopy_laneq_f32(float32x2_t v1, float32x4_t v2) {
  // CHECK: test_vcopy_laneq_f32
  return vcopy_laneq_f32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

uint8x8_t test_vcopy_laneq_u8(uint8x8_t v1, uint8x16_t v2) {
  // CHECK: test_vcopy_laneq_u8
  return vcopy_laneq_u8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

uint16x4_t test_vcopy_laneq_u16(uint16x4_t v1, uint16x8_t v2) {
  // CHECK: test_vcopy_laneq_u16
  return vcopy_laneq_u16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

uint32x2_t test_vcopy_laneq_u32(uint32x2_t v1, uint32x4_t v2) {
  // CHECK: test_vcopy_laneq_u32
  return vcopy_laneq_u32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

int8x16_t test_vcopyq_lane_s8(int8x16_t v1, int8x8_t v2) {
  // CHECK: test_vcopyq_lane_s8
  return vcopyq_lane_s8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

int16x8_t test_vcopyq_lane_s16(int16x8_t v1, int16x4_t v2) {
  // CHECK: test_vcopyq_lane_s16
  return vcopyq_lane_s16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

poly8x16_t test_vcopyq_lane_p8(poly8x16_t v1, poly8x8_t v2) {
  // CHECK: test_vcopyq_lane_p8
  return vcopyq_lane_p8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

poly16x8_t test_vcopyq_lane_p16(poly16x8_t v1, poly16x4_t v2) {
  // CHECK: test_vcopyq_lane_p16
  return vcopyq_lane_p16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

int32x4_t test_vcopyq_lane_s32(int32x4_t v1, int32x2_t v2) {
  // CHECK: test_vcopyq_lane_s32
  return vcopyq_lane_s32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

int64x2_t test_vcopyq_lane_s64(int64x2_t v1, int64x1_t v2) {
  // CHECK: test_vcopyq_lane_s64
  return vcopyq_lane_s64(v1, 1, v2, 0);
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
}

float32x4_t test_vcopyq_lane_f32(float32x4_t v1, float32x2_t v2) {
  // CHECK: test_vcopyq_lane_f32
  return vcopyq_lane_f32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

float64x2_t test_vcopyq_lane_f64(float64x2_t v1, float64x1_t v2) {
  // CHECK: test_vcopyq_lane_f64
  return vcopyq_lane_f64(v1, 1, v2, 0);
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
}

uint8x16_t test_vcopyq_lane_u8(uint8x16_t v1, uint8x8_t v2) {
  // CHECK: test_vcopyq_lane_u8
  return vcopyq_lane_u8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

uint16x8_t test_vcopyq_lane_u16(uint16x8_t v1, uint16x4_t v2) {
  // CHECK: test_vcopyq_lane_u16
  return vcopyq_lane_u16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

uint32x4_t test_vcopyq_lane_u32(uint32x4_t v1, uint32x2_t v2) {
  // CHECK: test_vcopyq_lane_u32
  return vcopyq_lane_u32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

uint64x2_t test_vcopyq_lane_u64(uint64x2_t v1, uint64x1_t v2) {
  // CHECK: test_vcopyq_lane_u64
  return vcopyq_lane_u64(v1, 1, v2, 0);
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
}

int8x16_t test_vcopyq_laneq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK: test_vcopyq_laneq_s8
  return vcopyq_laneq_s8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

int16x8_t test_vcopyq_laneq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK: test_vcopyq_laneq_s16
  return vcopyq_laneq_s16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

poly8x16_t test_vcopyq_laneq_p8(poly8x16_t v1, poly8x16_t v2) {
  // CHECK: test_vcopyq_laneq_p8
  return vcopyq_laneq_p8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

poly16x8_t test_vcopyq_laneq_p16(poly16x8_t v1, poly16x8_t v2) {
  // CHECK: test_vcopyq_laneq_p16
  return vcopyq_laneq_p16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

int32x4_t test_vcopyq_laneq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK: test_vcopyq_laneq_s32
  return vcopyq_laneq_s32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

float32x4_t test_vcopyq_laneq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vcopyq_laneq_f32
  return vcopyq_laneq_f32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

int64x2_t test_vcopyq_laneq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK: test_vcopyq_laneq_s64
  return vcopyq_laneq_s64(v1, 1, v2, 1);
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[1]
}

uint8x16_t test_vcopyq_laneq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK: test_vcopyq_laneq_u8
  return vcopyq_laneq_u8(v1, 5, v2, 3);
  // CHECK: ins {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
}

uint16x8_t test_vcopyq_laneq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK: test_vcopyq_laneq_u16
  return vcopyq_laneq_u16(v1, 2, v2, 3);
  // CHECK: ins {{v[0-9]+}}.h[2], {{v[0-9]+}}.h[3]
}

uint32x4_t test_vcopyq_laneq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK: test_vcopyq_laneq_u32
  return vcopyq_laneq_u32(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.s[0], {{v[0-9]+}}.s[1]
}

uint64x2_t test_vcopyq_laneq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK: test_vcopyq_laneq_u64
  return vcopyq_laneq_u64(v1, 0, v2, 1);
  // CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[1]
}

int8x8_t test_vcreate_s8(uint64_t v1) {
  // CHECK: test_vcreate_s8
  return vcreate_s8(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

int16x4_t test_vcreate_s16(uint64_t v1) {
  // CHECK: test_vcreate_s16
  return vcreate_s16(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

int32x2_t test_vcreate_s32(uint64_t v1) {
  // CHECK: test_vcreate_s32
  return vcreate_s32(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

int64x1_t test_vcreate_s64(uint64_t v1) {
  // CHECK: test_vcreate_s64
  return vcreate_s64(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

uint8x8_t test_vcreate_u8(uint64_t v1) {
  // CHECK: test_vcreate_u8
  return vcreate_u8(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

uint16x4_t test_vcreate_u16(uint64_t v1) {
  // CHECK: test_vcreate_u16
  return vcreate_u16(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

uint32x2_t test_vcreate_u32(uint64_t v1) {
  // CHECK: test_vcreate_u32
  return vcreate_u32(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

uint64x1_t test_vcreate_u64(uint64_t v1) {
  // CHECK: test_vcreate_u64
  return vcreate_u64(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

poly8x8_t test_vcreate_p8(uint64_t v1) {
  // CHECK: test_vcreate_p8
  return vcreate_p8(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

poly16x4_t test_vcreate_p16(uint64_t v1) {
  // CHECK: test_vcreate_p16
  return vcreate_p16(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

float16x4_t test_vcreate_f16(uint64_t v1) {
  // CHECK: test_vcreate_f16
  return vcreate_f16(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

float32x2_t test_vcreate_f32(uint64_t v1) {
  // CHECK: test_vcreate_f32
  return vcreate_f32(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

float64x1_t test_vcreate_f64(uint64_t v1) {
  // CHECK: test_vcreate_f64
  return vcreate_f64(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

uint8x8_t test_vdup_n_u8(uint8_t v1) {
  // CHECK: test_vdup_n_u8
  return vdup_n_u8(v1);
  // CHECK: dup {{v[0-9]+}}.8b, {{w[0-9]+}}
}

uint16x4_t test_vdup_n_u16(uint16_t v1) {
  // CHECK: test_vdup_n_u16
  return vdup_n_u16(v1);
  // CHECK: dup {{v[0-9]+}}.4h, {{w[0-9]+}}
}

uint32x2_t test_vdup_n_u32(uint32_t v1) {
  // CHECK: test_vdup_n_u32
  return vdup_n_u32(v1);
  // CHECK: dup {{v[0-9]+}}.2s, {{w[0-9]+}}
}

uint64x1_t test_vdup_n_u64(uint64_t v1) {
  // CHECK: test_vdup_n_u64
  return vdup_n_u64(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

uint8x16_t test_vdupq_n_u8(uint8_t v1) {
  // CHECK: test_vdupq_n_u8
  return vdupq_n_u8(v1);
  // CHECK: dup {{v[0-9]+}}.16b, {{w[0-9]+}}
}

uint16x8_t test_vdupq_n_u16(uint16_t v1) {
  // CHECK: test_vdupq_n_u16
  return vdupq_n_u16(v1);
  // CHECK: dup {{v[0-9]+}}.8h, {{w[0-9]+}}
}

uint32x4_t test_vdupq_n_u32(uint32_t v1) {
  // CHECK: test_vdupq_n_u32
  return vdupq_n_u32(v1);
  // CHECK: dup {{v[0-9]+}}.4s, {{w[0-9]+}}
}

uint64x2_t test_vdupq_n_u64(uint64_t v1) {
  // CHECK: test_vdupq_n_u64
  return vdupq_n_u64(v1);
  // CHECK: dup {{v[0-9]+}}.2d, {{x[0-9]+}}
}

int8x8_t test_vdup_n_s8(int8_t v1) {
  // CHECK: test_vdup_n_s8
  return vdup_n_s8(v1);
  // CHECK: dup {{v[0-9]+}}.8b, {{w[0-9]+}}
}

int16x4_t test_vdup_n_s16(int16_t v1) {
  // CHECK: test_vdup_n_s16
  return vdup_n_s16(v1);
  // CHECK: dup {{v[0-9]+}}.4h, {{w[0-9]+}}
}

int32x2_t test_vdup_n_s32(int32_t v1) {
  // CHECK: test_vdup_n_s32
  return vdup_n_s32(v1);
  // CHECK: dup {{v[0-9]+}}.2s, {{w[0-9]+}}
}

int64x1_t test_vdup_n_s64(int64_t v1) {
  // CHECK: test_vdup_n_s64
  return vdup_n_s64(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

int8x16_t test_vdupq_n_s8(int8_t v1) {
  // CHECK: test_vdupq_n_s8
  return vdupq_n_s8(v1);
  // CHECK: dup {{v[0-9]+}}.16b, {{w[0-9]+}}
}

int16x8_t test_vdupq_n_s16(int16_t v1) {
  // CHECK: test_vdupq_n_s16
  return vdupq_n_s16(v1);
  // CHECK: dup {{v[0-9]+}}.8h, {{w[0-9]+}}
}

int32x4_t test_vdupq_n_s32(int32_t v1) {
  // CHECK: test_vdupq_n_s32
  return vdupq_n_s32(v1);
  // CHECK: dup {{v[0-9]+}}.4s, {{w[0-9]+}}
}

int64x2_t test_vdupq_n_s64(int64_t v1) {
  // CHECK: test_vdupq_n_s64
  return vdupq_n_s64(v1);
  // CHECK: dup {{v[0-9]+}}.2d, {{x[0-9]+}}
}

poly8x8_t test_vdup_n_p8(poly8_t v1) {
  // CHECK: test_vdup_n_p8
  return vdup_n_p8(v1);
  // CHECK: dup {{v[0-9]+}}.8b, {{w[0-9]+}}
}

poly16x4_t test_vdup_n_p16(poly16_t v1) {
  // CHECK: test_vdup_n_p16
  return vdup_n_p16(v1);
  // CHECK: dup {{v[0-9]+}}.4h, {{w[0-9]+}}
}

poly8x16_t test_vdupq_n_p8(poly8_t v1) {
  // CHECK: test_vdupq_n_p8
  return vdupq_n_p8(v1);
  // CHECK: dup {{v[0-9]+}}.16b, {{w[0-9]+}}
}

poly16x8_t test_vdupq_n_p16(poly16_t v1) {
  // CHECK: test_vdupq_n_p16
  return vdupq_n_p16(v1);
  // CHECK: dup {{v[0-9]+}}.8h, {{w[0-9]+}}
}

float32x2_t test_vdup_n_f32(float32_t v1) {
  // CHECK: test_vdup_n_f32
  return vdup_n_f32(v1);
  // CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float64x1_t test_vdup_n_f64(float64_t v1) {
  // CHECK: test_vdup_n_f64
  return vdup_n_f64(v1);
  // CHECK: ret
}

float32x4_t test_vdupq_n_f32(float32_t v1) {
  // CHECK: test_vdupq_n_f32
  return vdupq_n_f32(v1);
  // CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float64x2_t test_vdupq_n_f64(float64_t v1) {
  // CHECK: test_vdupq_n_f64
  return vdupq_n_f64(v1);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

int8x8_t test_vdup_lane_s8(int8x8_t v1) {
  // CHECK: test_vdup_lane_s8
  return vdup_lane_s8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
}

int16x4_t test_vdup_lane_s16(int16x4_t v1) {
  // CHECK: test_vdup_lane_s16
  return vdup_lane_s16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
}

int32x2_t test_vdup_lane_s32(int32x2_t v1) {
  // CHECK: test_vdup_lane_s32
  return vdup_lane_s32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int64x1_t test_vdup_lane_s64(int64x1_t v1) {
  // CHECK: test_vdup_lane_s64
  return vdup_lane_s64(v1, 0);
  // CHECK: ret
}

int8x16_t test_vdupq_lane_s8(int8x8_t v1) {
  // CHECK: test_vdupq_lane_s8
  return vdupq_lane_s8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
}

int16x8_t test_vdupq_lane_s16(int16x4_t v1) {
  // CHECK: test_vdupq_lane_s16
  return vdupq_lane_s16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
}

int32x4_t test_vdupq_lane_s32(int32x2_t v1) {
  // CHECK: test_vdupq_lane_s32
  return vdupq_lane_s32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int64x2_t test_vdupq_lane_s64(int64x1_t v1) {
  // CHECK: test_vdupq_lane_s64
  return vdupq_lane_s64(v1, 0);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

uint8x8_t test_vdup_lane_u8(uint8x8_t v1) {
  // CHECK: test_vdup_lane_u8
  return vdup_lane_u8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
}

uint16x4_t test_vdup_lane_u16(uint16x4_t v1) {
  // CHECK: test_vdup_lane_u16
  return vdup_lane_u16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
}

uint32x2_t test_vdup_lane_u32(uint32x2_t v1) {
  // CHECK: test_vdup_lane_u32
  return vdup_lane_u32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

uint64x1_t test_vdup_lane_u64(uint64x1_t v1) {
  // CHECK: test_vdup_lane_u64
  return vdup_lane_u64(v1, 0);
  // CHECK: ret
}

uint8x16_t test_vdupq_lane_u8(uint8x8_t v1) {
  // CHECK: test_vdupq_lane_u8
  return vdupq_lane_u8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
}

uint16x8_t test_vdupq_lane_u16(uint16x4_t v1) {
  // CHECK: test_vdupq_lane_u16
  return vdupq_lane_u16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
}

uint32x4_t test_vdupq_lane_u32(uint32x2_t v1) {
  // CHECK: test_vdupq_lane_u32
  return vdupq_lane_u32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

uint64x2_t test_vdupq_lane_u64(uint64x1_t v1) {
  // CHECK: test_vdupq_lane_u64
  return vdupq_lane_u64(v1, 0);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

int8x8_t test_vdup_laneq_s8(int8x16_t v1) {
  // CHECK: test_vdup_laneq_s8
  return vdup_laneq_s8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
}

int16x4_t test_vdup_laneq_s16(int16x8_t v1) {
  // CHECK: test_vdup_laneq_s16
  return vdup_laneq_s16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
}

int32x2_t test_vdup_laneq_s32(int32x4_t v1) {
  // CHECK: test_vdup_laneq_s32
  return vdup_laneq_s32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int64x1_t test_vdup_laneq_s64(int64x2_t v1) {
  // CHECK: test_vdup_laneq_s64
  return vdup_laneq_s64(v1, 0);
  // CHECK: ret
}

int8x16_t test_vdupq_laneq_s8(int8x16_t v1) {
  // CHECK: test_vdupq_laneq_s8
  return vdupq_laneq_s8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
}

int16x8_t test_vdupq_laneq_s16(int16x8_t v1) {
  // CHECK: test_vdupq_laneq_s16
  return vdupq_laneq_s16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
}

int32x4_t test_vdupq_laneq_s32(int32x4_t v1) {
  // CHECK: test_vdupq_laneq_s32
  return vdupq_laneq_s32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int64x2_t test_vdupq_laneq_s64(int64x2_t v1) {
  // CHECK: test_vdupq_laneq_s64
  return vdupq_laneq_s64(v1, 0);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

uint8x8_t test_vdup_laneq_u8(uint8x16_t v1) {
  // CHECK: test_vdup_laneq_u8
  return vdup_laneq_u8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
}

uint16x4_t test_vdup_laneq_u16(uint16x8_t v1) {
  // CHECK: test_vdup_laneq_u16
  return vdup_laneq_u16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
}

uint32x2_t test_vdup_laneq_u32(uint32x4_t v1) {
  // CHECK: test_vdup_laneq_u32
  return vdup_laneq_u32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

uint64x1_t test_vdup_laneq_u64(uint64x2_t v1) {
  // CHECK: test_vdup_laneq_u64
  return vdup_laneq_u64(v1, 0);
  // CHECK: ret
}

uint8x16_t test_vdupq_laneq_u8(uint8x16_t v1) {
  // CHECK: test_vdupq_laneq_u8
  return vdupq_laneq_u8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
}

uint16x8_t test_vdupq_laneq_u16(uint16x8_t v1) {
  // CHECK: test_vdupq_laneq_u16
  return vdupq_laneq_u16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
}

uint32x4_t test_vdupq_laneq_u32(uint32x4_t v1) {
  // CHECK: test_vdupq_laneq_u32
  return vdupq_laneq_u32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

uint64x2_t test_vdupq_laneq_u64(uint64x2_t v1) {
  // CHECK: test_vdupq_laneq_u64
  return vdupq_laneq_u64(v1, 0);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

poly8x8_t test_vdup_lane_p8(poly8x8_t v1) {
  // CHECK: test_vdup_lane_p8
  return vdup_lane_p8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
}

poly16x4_t test_vdup_lane_p16(poly16x4_t v1) {
  // CHECK: test_vdup_lane_p16
  return vdup_lane_p16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
}

poly8x16_t test_vdupq_lane_p8(poly8x8_t v1) {
  // CHECK: test_vdupq_lane_p8
  return vdupq_lane_p8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
}

poly16x8_t test_vdupq_lane_p16(poly16x4_t v1) {
  // CHECK: test_vdupq_lane_p16
  return vdupq_lane_p16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
}

poly8x8_t test_vdup_laneq_p8(poly8x16_t v1) {
  // CHECK: test_vdup_laneq_p8
  return vdup_laneq_p8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
}

poly16x4_t test_vdup_laneq_p16(poly16x8_t v1) {
  // CHECK: test_vdup_laneq_p16
  return vdup_laneq_p16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
}

poly8x16_t test_vdupq_laneq_p8(poly8x16_t v1) {
  // CHECK: test_vdupq_laneq_p8
  return vdupq_laneq_p8(v1, 5);
  // CHECK: dup {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
}

poly16x8_t test_vdupq_laneq_p16(poly16x8_t v1) {
  // CHECK: test_vdupq_laneq_p16
  return vdupq_laneq_p16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
}

float16x4_t test_vdup_lane_f16(float16x4_t v1) {
  // CHECK: test_vdup_lane_f16
  return vdup_lane_f16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
}

float32x2_t test_vdup_lane_f32(float32x2_t v1) {
  // CHECK: test_vdup_lane_f32
  return vdup_lane_f32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

float64x1_t test_vdup_lane_f64(float64x1_t v1) {
  // CHECK: test_vdup_lane_f64
  return vdup_lane_f64(v1, 0);
  // CHECK: ret
}

float16x4_t test_vdup_laneq_f16(float16x8_t v1) {
  // CHECK: test_vdup_laneq_f16
  return vdup_laneq_f16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
}

float32x2_t test_vdup_laneq_f32(float32x4_t v1) {
  // CHECK: test_vdup_laneq_f32
  return vdup_laneq_f32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

float64x1_t test_vdup_laneq_f64(float64x2_t v1) {
  // CHECK: test_vdup_laneq_f64
  return vdup_laneq_f64(v1, 0);
  // CHECK: ret
}

float16x8_t test_vdupq_lane_f16(float16x4_t v1) {
  // CHECK: test_vdupq_lane_f16
  return vdupq_lane_f16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
}

float32x4_t test_vdupq_lane_f32(float32x2_t v1) {
  // CHECK: test_vdupq_lane_f32
  return vdupq_lane_f32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

float64x2_t test_vdupq_lane_f64(float64x1_t v1) {
  // CHECK: test_vdupq_lane_f64
  return vdupq_lane_f64(v1, 0);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float16x8_t test_vdupq_laneq_f16(float16x8_t v1) {
  // CHECK: test_vdupq_laneq_f16
  return vdupq_laneq_f16(v1, 2);
  // CHECK: dup {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
}

float32x4_t test_vdupq_laneq_f32(float32x4_t v1) {
  // CHECK: test_vdupq_laneq_f32
  return vdupq_laneq_f32(v1, 1);
  // CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

float64x2_t test_vdupq_laneq_f64(float64x2_t v1) {
  // CHECK: test_vdupq_laneq_f64
  return vdupq_laneq_f64(v1, 0);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

int8x8_t test_vmov_n_s8(int8_t v1) {
  // CHECK: test_vmov_n_s8
  return vmov_n_s8(v1);
  // CHECK: dup {{v[0-9]+}}.8b, {{w[0-9]+}}
}

int16x4_t test_vmov_n_s16(int16_t v1) {
  // CHECK: test_vmov_n_s16
  return vmov_n_s16(v1);
  // CHECK: dup {{v[0-9]+}}.4h, {{w[0-9]+}}
}

int32x2_t test_vmov_n_s32(int32_t v1) {
  // CHECK: test_vmov_n_s32
  return vmov_n_s32(v1);
  // CHECK: dup {{v[0-9]+}}.2s, {{w[0-9]+}}
}

int64x1_t test_vmov_n_s64(int64_t v1) {
  // CHECK: test_vmov_n_s64
  return vmov_n_s64(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

int8x16_t test_vmovq_n_s8(int8_t v1) {
  // CHECK: test_vmovq_n_s8
  return vmovq_n_s8(v1);
  // CHECK: dup {{v[0-9]+}}.16b, {{w[0-9]+}}
}

int16x8_t test_vmovq_n_s16(int16_t v1) {
  // CHECK: test_vmovq_n_s16
  return vmovq_n_s16(v1);
  // CHECK: dup {{v[0-9]+}}.8h, {{w[0-9]+}}
}

int32x4_t test_vmovq_n_s32(int32_t v1) {
  // CHECK: test_vmovq_n_s32
  return vmovq_n_s32(v1);
  // CHECK: dup {{v[0-9]+}}.4s, {{w[0-9]+}}
}

int64x2_t test_vmovq_n_s64(int64_t v1) {
  // CHECK: test_vmovq_n_s64
  return vmovq_n_s64(v1);
  // CHECK: dup {{v[0-9]+}}.2d, {{x[0-9]+}}
}

uint8x8_t test_vmov_n_u8(uint8_t v1) {
  // CHECK: test_vmov_n_u8
  return vmov_n_u8(v1);
  // CHECK: dup {{v[0-9]+}}.8b, {{w[0-9]+}}
}

uint16x4_t test_vmov_n_u16(uint16_t v1) {
  // CHECK: test_vmov_n_u16
  return vmov_n_u16(v1);
  // CHECK: dup {{v[0-9]+}}.4h, {{w[0-9]+}}
}

uint32x2_t test_vmov_n_u32(uint32_t v1) {
  // CHECK: test_vmov_n_u32
  return vmov_n_u32(v1);
  // CHECK: dup {{v[0-9]+}}.2s, {{w[0-9]+}}
}

uint64x1_t test_vmov_n_u64(uint64_t v1) {
  // CHECK: test_vmov_n_u64
  return vmov_n_u64(v1);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

uint8x16_t test_vmovq_n_u8(uint8_t v1) {
  // CHECK: test_vmovq_n_u8
  return vmovq_n_u8(v1);
  // CHECK: dup {{v[0-9]+}}.16b, {{w[0-9]+}}
}

uint16x8_t test_vmovq_n_u16(uint16_t v1) {
  // CHECK: test_vmovq_n_u16
  return vmovq_n_u16(v1);
  // CHECK: dup {{v[0-9]+}}.8h, {{w[0-9]+}}
}

uint32x4_t test_vmovq_n_u32(uint32_t v1) {
  // CHECK: test_vmovq_n_u32
  return vmovq_n_u32(v1);
  // CHECK: dup {{v[0-9]+}}.4s, {{w[0-9]+}}
}

uint64x2_t test_vmovq_n_u64(uint64_t v1) {
  // CHECK: test_vmovq_n_u64
  return vmovq_n_u64(v1);
  // CHECK: dup {{v[0-9]+}}.2d, {{x[0-9]+}}
}

poly8x8_t test_vmov_n_p8(poly8_t v1) {
  // CHECK: test_vmov_n_p8
  return vmov_n_p8(v1);
  // CHECK: dup {{v[0-9]+}}.8b, {{w[0-9]+}}
}

poly16x4_t test_vmov_n_p16(poly16_t v1) {
  // CHECK: test_vmov_n_p16
  return vmov_n_p16(v1);
  // CHECK: dup {{v[0-9]+}}.4h, {{w[0-9]+}}
}

poly8x16_t test_vmovq_n_p8(poly8_t v1) {
  // CHECK: test_vmovq_n_p8
  return vmovq_n_p8(v1);
  // CHECK: dup {{v[0-9]+}}.16b, {{w[0-9]+}}
}

poly16x8_t test_vmovq_n_p16(poly16_t v1) {
  // CHECK: test_vmovq_n_p16
  return vmovq_n_p16(v1);
  // CHECK: dup {{v[0-9]+}}.8h, {{w[0-9]+}}
}

float32x2_t test_vmov_n_f32(float32_t v1) {
  // CHECK: test_vmov_n_f32
  return vmov_n_f32(v1);
  // CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float64x1_t test_vmov_n_f64(float64_t v1) {
  // CHECK: test_vmov_n_f64
  return vmov_n_f64(v1);
  // CHECK: ret
}

float32x4_t test_vmovq_n_f32(float32_t v1) {
  // CHECK: test_vmovq_n_f32
  return vmovq_n_f32(v1);
  // CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float64x2_t test_vmovq_n_f64(float64_t v1) {
  // CHECK: test_vmovq_n_f64
  return vmovq_n_f64(v1);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

// CHECK: test_vcopy_lane_s64
int64x1_t test_vcopy_lane_s64(int64x1_t a, int64x1_t c) {
  return vcopy_lane_s64(a, 0, c, 0);
// CHECK: fmov {{d[0-9]+}}, {{d[0-9]+}}
// CHECK-NOT: dup {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vcopy_lane_u64
uint64x1_t test_vcopy_lane_u64(uint64x1_t a, uint64x1_t c) {
  return vcopy_lane_u64(a, 0, c, 0);
// CHECK: fmov {{d[0-9]+}}, {{d[0-9]+}}
// CHECK-NOT: dup {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vcopy_lane_f64
float64x1_t test_vcopy_lane_f64(float64x1_t a, float64x1_t c) {
  return vcopy_lane_f64(a, 0, c, 0);
// CHECK: fmov {{d[0-9]+}}, {{d[0-9]+}}
// CHECK-NOT: dup {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vcopy_laneq_s64
int64x1_t test_vcopy_laneq_s64(int64x1_t a, int64x2_t c) {
  return vcopy_laneq_s64(a, 0, c, 1);
// CHECK: dup {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK: test_vcopy_laneq_u64
uint64x1_t test_vcopy_laneq_u64(uint64x1_t a, uint64x2_t c) {
  return vcopy_laneq_u64(a, 0, c, 1);
// CHECK: dup {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK: test_vcopy_laneq_f64
float64x1_t test_vcopy_laneq_f64(float64x1_t a, float64x2_t c) {
  return vcopy_laneq_f64(a, 0, c, 1);
// CHECK: dup {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK: test_vcopy_laneq_p64
poly64x1_t test_vcopy_laneq_p64(poly64x1_t a, poly64x2_t c) {
  return vcopy_laneq_p64(a, 0, c, 1);
// CHECK: dup {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK: test_vcopyq_laneq_f64
float64x2_t test_vcopyq_laneq_f64(float64x2_t a, float64x2_t c) {
// CHECK: ins  {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[1]
  return vcopyq_laneq_f64(a, 1, c, 1);
}

// CHECK: test_vget_lane_f16
int test_vget_lane_f16(float16x4_t v1) {
  float16_t a = vget_lane_f16(v1, 3);
  return (int)a;
// CHECK: dup {{h[0-9]+}}, {{v[0-9]+}}.h[3]
}

// CHECK: test_vgetq_lane_f16
int test_vgetq_lane_f16(float16x8_t v1) {
  float16_t a = vgetq_lane_f16(v1, 7);
  return (int)a;
// CHECK: dup {{h[0-9]+}}, {{v[0-9]+}}.h[7]
}

// CHECK: test_vget_lane_f16_2
float test_vget_lane_f16_2(float16x4_t v1) {
  float16_t a = vget_lane_f16(v1, 3);
  return (float)a;
// CHECK: dup {{h[0-9]+}}, {{v[0-9]+}}.h[3]
}

// CHECK: test_vgetq_lane_f16_2
float test_vgetq_lane_f16_2(float16x8_t v1) {
  float16_t a = vgetq_lane_f16(v1, 7);
  return (float)a;
// CHECK: dup {{h[0-9]+}}, {{v[0-9]+}}.h[7]
}

// CHECK: test_vset_lane_f16
float16x4_t test_vset_lane_f16(float16x4_t v1) {
  float16_t a;
  return vset_lane_f16(a, v1, 3);
// CHECK: fmov  {{s[0-9]+}}, wzr
// CHECK-NEXT: ins {{v[0-9]+}}.h[3],  {{v[0-9]+}}.h[0]
}

// CHECK: test_vsetq_lane_f16
float16x8_t test_vsetq_lane_f16(float16x8_t v1) {
  float16_t a;
  return vsetq_lane_f16(a, v1, 7);
// CHECK: fmov  {{s[0-9]+}}, wzr
// CHECK-NEXT: ins {{v[0-9]+}}.h[7],  {{v[0-9]+}}.h[0]
}

// CHECK: test_vset_lane_f16_2
float16x4_t test_vset_lane_f16_2(float16x4_t v1) {
  float16_t a = vget_lane_f16(v1, 0);
  return vset_lane_f16(a, v1, 3);
// CHECK: ins {{v[0-9]+}}.h[3],  {{v[0-9]+}}.h[0]
}

// CHECK: test_vsetq_lane_f16_2
float16x8_t test_vsetq_lane_f16_2(float16x8_t v1) {
  float16_t a = vgetq_lane_f16(v1, 0);
  return vsetq_lane_f16(a, v1, 7);
// CHECK: ins {{v[0-9]+}}.h[7],  {{v[0-9]+}}.h[0]
}


// CHECK: test_vsetq_lane_f16_3
float16x8_t test_vsetq_lane_f16_3(float16x8_t v1, float b, float c) {
  float16_t a = (float16_t)b;
  return vsetq_lane_f16(a, v1, 7);
// CHECK: ins {{v[0-9]+}}.h[7],  {{w[0-9]+}}
}

// CHECK: test_vsetq_lane_f16_4
float16x8_t test_vsetq_lane_f16_4(float16x8_t v1, float b, float c) {
  float16_t a = (float16_t)b + 1.0;
  return vsetq_lane_f16(a, v1, 7);
// CHECK: ins {{v[0-9]+}}.h[7],  {{w[0-9]+}}
}

