// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

// CHECK: test_vceqz_s8
// CHECK: cmeq  {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
uint8x8_t test_vceqz_s8(int8x8_t a) {
  return vceqz_s8(a);
}

// CHECK: test_vceqz_s16
// CHECK: cmeq  {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
uint16x4_t test_vceqz_s16(int16x4_t a) {
  return vceqz_s16(a);
}

// CHECK: test_vceqz_s32
// CHECK: cmeq  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
uint32x2_t test_vceqz_s32(int32x2_t a) {
  return vceqz_s32(a);
}

// CHECK: test_vceqz_s64
// CHECK: cmeq {{d[0-9]+}}, {{d[0-9]+}}, #0x0
uint64x1_t test_vceqz_s64(int64x1_t a) {
  return vceqz_s64(a);
}

// CHECK: test_vceqz_u64
// CHECK: cmeq {{d[0-9]+}}, {{d[0-9]+}}, #0x0
uint64x1_t test_vceqz_u64(uint64x1_t a) {
  return vceqz_u64(a);
}

// CHECK: test_vceqz_p64
// CHECK: cmeq {{d[0-9]+}}, {{d[0-9]+}}, #0x0
uint64x1_t test_vceqz_p64(poly64x1_t a) {
  return vceqz_p64(a);
}

// CHECK: test_vceqzq_s8
// CHECK: cmeq  {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
uint8x16_t test_vceqzq_s8(int8x16_t a) {
  return vceqzq_s8(a);
}

// CHECK: test_vceqzq_s16
// CHECK: cmeq  {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
uint16x8_t test_vceqzq_s16(int16x8_t a) {
  return vceqzq_s16(a);
}

// CHECK: test_vceqzq_s32
// CHECK: cmeq  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
uint32x4_t test_vceqzq_s32(int32x4_t a) {
  return vceqzq_s32(a);
}

// CHECK: test_vceqzq_s64
// CHECK: cmeq  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
uint64x2_t test_vceqzq_s64(int64x2_t a) {
  return vceqzq_s64(a);
}

// CHECK: test_vceqz_u8
// CHECK: cmeq  {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
uint8x8_t test_vceqz_u8(uint8x8_t a) {
  return vceqz_u8(a);
}

// CHECK: test_vceqz_u16
// CHECK: cmeq  {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
uint16x4_t test_vceqz_u16(uint16x4_t a) {
  return vceqz_u16(a);
}

// CHECK: test_vceqz_u32
// CHECK: cmeq  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
uint32x2_t test_vceqz_u32(uint32x2_t a) {
  return vceqz_u32(a);
}

// CHECK: test_vceqzq_u8
// CHECK: cmeq  {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
uint8x16_t test_vceqzq_u8(uint8x16_t a) {
  return vceqzq_u8(a);
}

// CHECK: test_vceqzq_u16
// CHECK: cmeq  {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
uint16x8_t test_vceqzq_u16(uint16x8_t a) {
  return vceqzq_u16(a);
}

// CHECK: test_vceqzq_u32
// CHECK: cmeq  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
uint32x4_t test_vceqzq_u32(uint32x4_t a) {
  return vceqzq_u32(a);
}

// CHECK: test_vceqzq_u64
// CHECK: cmeq  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
uint64x2_t test_vceqzq_u64(uint64x2_t a) {
  return vceqzq_u64(a);
}

// CHECK: test_vceqz_f32
// CHECK: fcmeq  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
uint32x2_t test_vceqz_f32(float32x2_t a) {
  return vceqz_f32(a);
}

// CHECK: test_vceqz_f64
// CHECK: fcmeq  {{d[0-9]+}}, {{d[0-9]+}}, #0
uint64x1_t test_vceqz_f64(float64x1_t a) {
  return vceqz_f64(a);
}

// CHECK: test_vceqzq_f32
// CHECK: fcmeq  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
uint32x4_t test_vceqzq_f32(float32x4_t a) {
  return vceqzq_f32(a);
}

// CHECK: test_vceqz_p16
// CHECK: cmeq  {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
uint16x4_t test_vceqz_p16(poly16x4_t a) {
  return vceqz_p16(a);
}

// CHECK: test_vceqzq_p16
// CHECK: cmeq  {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
uint16x8_t test_vceqzq_p16(poly16x8_t a) {
  return vceqzq_p16(a);
}

// CHECK: test_vceqzq_f64
// CHECK: fcmeq  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
uint64x2_t test_vceqzq_f64(float64x2_t a) {
  return vceqzq_f64(a);
}

// CHECK: test_vceqzq_p64
// CHECK: cmeq  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
uint64x2_t test_vceqzq_p64(poly64x2_t a) {
  return vceqzq_p64(a);
}

// CHECK: test_vcgez_s8
// CHECK: cmge  {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
uint8x8_t test_vcgez_s8(int8x8_t a) {
  return vcgez_s8(a);
}

// CHECK: test_vcgez_s16
// CHECK: cmge  {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
uint16x4_t test_vcgez_s16(int16x4_t a) {
  return vcgez_s16(a);
}

// CHECK: test_vcgez_s32
// CHECK: cmge  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
uint32x2_t test_vcgez_s32(int32x2_t a) {
  return vcgez_s32(a);
}

// CHECK: test_vcgez_s64
// CHECK: cmge {{d[0-9]+}}, {{d[0-9]+}}, #0x0
uint64x1_t test_vcgez_s64(int64x1_t a) {
  return vcgez_s64(a);
}

// CHECK: test_vcgezq_s8
// CHECK: cmge  {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
uint8x16_t test_vcgezq_s8(int8x16_t a) {
  return vcgezq_s8(a);
}

// CHECK: test_vcgezq_s16
// CHECK: cmge  {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
uint16x8_t test_vcgezq_s16(int16x8_t a) {
  return vcgezq_s16(a);
}

// CHECK: test_vcgezq_s32
// CHECK: cmge  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
uint32x4_t test_vcgezq_s32(int32x4_t a) {
  return vcgezq_s32(a);
}

// CHECK: test_vcgezq_s64
// CHECK: cmge  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
uint64x2_t test_vcgezq_s64(int64x2_t a) {
  return vcgezq_s64(a);
}

// CHECK: test_vcgez_f32
// CHECK: fcmge  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
uint32x2_t test_vcgez_f32(float32x2_t a) {
  return vcgez_f32(a);
}

// CHECK: test_vcgezq_f32
// CHECK: fcmge  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
uint32x4_t test_vcgezq_f32(float32x4_t a) {
  return vcgezq_f32(a);
}

// CHECK: test_vcgezq_f64
// CHECK: fcmge  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
uint64x2_t test_vcgezq_f64(float64x2_t a) {
  return vcgezq_f64(a);
}

// CHECK: test_vclez_s8
// CHECK: cmle  {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
uint8x8_t test_vclez_s8(int8x8_t a) {
  return vclez_s8(a);
}

// CHECK: test_vclez_s16
// CHECK: cmle  {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
uint16x4_t test_vclez_s16(int16x4_t a) {
  return vclez_s16(a);
}

// CHECK: test_vclez_s32
// CHECK: cmle  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
uint32x2_t test_vclez_s32(int32x2_t a) {
  return vclez_s32(a);
}

// CHECK: test_vclez_s64
// CHECK: cmle {{d[0-9]+}}, {{d[0-9]+}}, #0x0
uint64x1_t test_vclez_s64(int64x1_t a) {
  return vclez_s64(a);
}

// CHECK: test_vclezq_s8
// CHECK: cmle  {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
uint8x16_t test_vclezq_s8(int8x16_t a) {
  return vclezq_s8(a);
}

// CHECK: test_vclezq_s16
// CHECK: cmle  {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
uint16x8_t test_vclezq_s16(int16x8_t a) {
  return vclezq_s16(a);
}

// CHECK: test_vclezq_s32
// CHECK: cmle  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
uint32x4_t test_vclezq_s32(int32x4_t a) {
  return vclezq_s32(a);
}

// CHECK: test_vclezq_s64
// CHECK: cmle  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
uint64x2_t test_vclezq_s64(int64x2_t a) {
  return vclezq_s64(a);
}

// CHECK: test_vclez_f32
// CHECK: fcmle  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
uint32x2_t test_vclez_f32(float32x2_t a) {
  return vclez_f32(a);
}

// CHECK: test_vclezq_f32
// CHECK: fcmle  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
uint32x4_t test_vclezq_f32(float32x4_t a) {
  return vclezq_f32(a);
}

// CHECK: test_vclezq_f64
// CHECK: fcmle  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
uint64x2_t test_vclezq_f64(float64x2_t a) {
  return vclezq_f64(a);
}

// CHECK: test_vcgtz_s8
// CHECK: cmgt  {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
uint8x8_t test_vcgtz_s8(int8x8_t a) {
  return vcgtz_s8(a);
}

// CHECK: test_vcgtz_s16
// CHECK: cmgt  {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
uint16x4_t test_vcgtz_s16(int16x4_t a) {
  return vcgtz_s16(a);
}

// CHECK: test_vcgtz_s32
// CHECK: cmgt  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
uint32x2_t test_vcgtz_s32(int32x2_t a) {
  return vcgtz_s32(a);
}

// CHECK: test_vcgtz_s64
// CHECK: cmgt {{d[0-9]+}}, {{d[0-9]+}}, #0x0
uint64x1_t test_vcgtz_s64(int64x1_t a) {
  return vcgtz_s64(a);
}

// CHECK: test_vcgtzq_s8
// CHECK: cmgt  {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
uint8x16_t test_vcgtzq_s8(int8x16_t a) {
  return vcgtzq_s8(a);
}

// CHECK: test_vcgtzq_s16
// CHECK: cmgt  {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
uint16x8_t test_vcgtzq_s16(int16x8_t a) {
  return vcgtzq_s16(a);
}

// CHECK: test_vcgtzq_s32
// CHECK: cmgt  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
uint32x4_t test_vcgtzq_s32(int32x4_t a) {
  return vcgtzq_s32(a);
}

// CHECK: test_vcgtzq_s64
// CHECK: cmgt  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
uint64x2_t test_vcgtzq_s64(int64x2_t a) {
  return vcgtzq_s64(a);
}

// CHECK: test_vcgtz_f32
// CHECK: fcmgt  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
uint32x2_t test_vcgtz_f32(float32x2_t a) {
  return vcgtz_f32(a);
}

// CHECK: test_vcgtzq_f32
// CHECK: fcmgt  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
uint32x4_t test_vcgtzq_f32(float32x4_t a) {
  return vcgtzq_f32(a);
}

// CHECK: test_vcgtzq_f64
// CHECK: fcmgt  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
uint64x2_t test_vcgtzq_f64(float64x2_t a) {
  return vcgtzq_f64(a);
}

// CHECK: test_vcltz_s8
// CHECK: cmlt  {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
uint8x8_t test_vcltz_s8(int8x8_t a) {
  return vcltz_s8(a);
}

// CHECK: test_vcltz_s16
// CHECK: cmlt  {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
uint16x4_t test_vcltz_s16(int16x4_t a) {
  return vcltz_s16(a);
}

// CHECK: test_vcltz_s32
// CHECK: cmlt  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
uint32x2_t test_vcltz_s32(int32x2_t a) {
  return vcltz_s32(a);
}

// CHECK: test_vcltz_s64
// CHECK: cmlt {{d[0-9]+}}, {{d[0-9]+}}, #0
uint64x1_t test_vcltz_s64(int64x1_t a) {
  return vcltz_s64(a);
}

// CHECK: test_vcltzq_s8
// CHECK: cmlt  {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
uint8x16_t test_vcltzq_s8(int8x16_t a) {
  return vcltzq_s8(a);
}

// CHECK: test_vcltzq_s16
// CHECK: cmlt  {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
uint16x8_t test_vcltzq_s16(int16x8_t a) {
  return vcltzq_s16(a);
}

// CHECK: test_vcltzq_s32
// CHECK: cmlt  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
uint32x4_t test_vcltzq_s32(int32x4_t a) {
  return vcltzq_s32(a);
}

// CHECK: test_vcltzq_s64
// CHECK: cmlt  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
uint64x2_t test_vcltzq_s64(int64x2_t a) {
  return vcltzq_s64(a);
}

// CHECK: test_vcltz_f32
// CHECK: fcmlt  {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
uint32x2_t test_vcltz_f32(float32x2_t a) {
  return vcltz_f32(a);
}

// CHECK: test_vcltzq_f32
// CHECK: fcmlt  {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
uint32x4_t test_vcltzq_f32(float32x4_t a) {
  return vcltzq_f32(a);
}

// CHECK: test_vcltzq_f64
// CHECK: fcmlt  {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
uint64x2_t test_vcltzq_f64(float64x2_t a) {
  return vcltzq_f64(a);
}

// CHECK: test_vrev16_s8
// CHECK: rev16 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
int8x8_t test_vrev16_s8(int8x8_t a) {
  return vrev16_s8(a);
}

// CHECK: test_vrev16_u8
// CHECK: rev16 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
uint8x8_t test_vrev16_u8(uint8x8_t a) {
  return vrev16_u8(a);
}

// CHECK: test_vrev16_p8
// CHECK: rev16 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
poly8x8_t test_vrev16_p8(poly8x8_t a) {
  return vrev16_p8(a);
}

// CHECK: test_vrev16q_s8
// CHECK: rev16 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
int8x16_t test_vrev16q_s8(int8x16_t a) {
  return vrev16q_s8(a);
}

// CHECK: test_vrev16q_u8
// CHECK: rev16 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
uint8x16_t test_vrev16q_u8(uint8x16_t a) {
  return vrev16q_u8(a);
}

// CHECK: test_vrev16q_p8
// CHECK: rev16 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
poly8x16_t test_vrev16q_p8(poly8x16_t a) {
  return vrev16q_p8(a);
}

// CHECK: test_vrev32_s8
// CHECK: rev32 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
int8x8_t test_vrev32_s8(int8x8_t a) {
  return vrev32_s8(a);
}

// CHECK: test_vrev32_s16
// CHECK: rev32 v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
int16x4_t test_vrev32_s16(int16x4_t a) {
  return vrev32_s16(a);
}

// CHECK: test_vrev32_u8
// CHECK: rev32 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
uint8x8_t test_vrev32_u8(uint8x8_t a) {
  return vrev32_u8(a);
}

// CHECK: test_vrev32_u16
// CHECK: rev32 v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
uint16x4_t test_vrev32_u16(uint16x4_t a) {
  return vrev32_u16(a);
}

// CHECK: test_vrev32_p8
// CHECK: rev32 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
poly8x8_t test_vrev32_p8(poly8x8_t a) {
  return vrev32_p8(a);
}

// CHECK: test_vrev32_p16
// CHECK: rev32 v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
poly16x4_t test_vrev32_p16(poly16x4_t a) {
  return vrev32_p16(a);
}

// CHECK: test_vrev32q_s8
// CHECK: rev32 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
int8x16_t test_vrev32q_s8(int8x16_t a) {
  return vrev32q_s8(a);
}

// CHECK: test_vrev32q_s16
// CHECK: rev32 v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
int16x8_t test_vrev32q_s16(int16x8_t a) {
  return vrev32q_s16(a);
}

// CHECK: test_vrev32q_u8
// CHECK: rev32 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
uint8x16_t test_vrev32q_u8(uint8x16_t a) {
  return vrev32q_u8(a);
}

// CHECK: test_vrev32q_u16
// CHECK: rev32 v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
uint16x8_t test_vrev32q_u16(uint16x8_t a) {
  return vrev32q_u16(a);
}

// CHECK: test_vrev32q_p8
// CHECK: rev32 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
poly8x16_t test_vrev32q_p8(poly8x16_t a) {
  return vrev32q_p8(a);
}

// CHECK: test_vrev32q_p16
// CHECK: rev32 v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
poly16x8_t test_vrev32q_p16(poly16x8_t a) {
  return vrev32q_p16(a);
}

// CHECK: test_vrev64_s8
// CHECK: rev64 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
int8x8_t test_vrev64_s8(int8x8_t a) {
  return vrev64_s8(a);
}

// CHECK: test_vrev64_s16
// CHECK: rev64 v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
int16x4_t test_vrev64_s16(int16x4_t a) {
  return vrev64_s16(a);
}

// CHECK: test_vrev64_s32
// CHECK: rev64 v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
int32x2_t test_vrev64_s32(int32x2_t a) {
  return vrev64_s32(a);
}

// CHECK: test_vrev64_u8
// CHECK: rev64 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
uint8x8_t test_vrev64_u8(uint8x8_t a) {
  return vrev64_u8(a);
}

// CHECK: test_vrev64_u16
// CHECK: rev64 v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
uint16x4_t test_vrev64_u16(uint16x4_t a) {
  return vrev64_u16(a);
}

// CHECK: test_vrev64_u32
// CHECK: rev64 v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
uint32x2_t test_vrev64_u32(uint32x2_t a) {
  return vrev64_u32(a);
}

// CHECK: test_vrev64_p8
// CHECK: rev64 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
poly8x8_t test_vrev64_p8(poly8x8_t a) {
  return vrev64_p8(a);
}

// CHECK: test_vrev64_p16
// CHECK: rev64 v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
poly16x4_t test_vrev64_p16(poly16x4_t a) {
  return vrev64_p16(a);
}

// CHECK: test_vrev64_f32
// CHECK: rev64 v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
float32x2_t test_vrev64_f32(float32x2_t a) {
  return vrev64_f32(a);
}

// CHECK: test_vrev64q_s8
// CHECK: rev64 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
int8x16_t test_vrev64q_s8(int8x16_t a) {
  return vrev64q_s8(a);
}

// CHECK: test_vrev64q_s16
// CHECK: rev64 v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
int16x8_t test_vrev64q_s16(int16x8_t a) {
  return vrev64q_s16(a);
}

// CHECK: test_vrev64q_s32
// CHECK: rev64 v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
int32x4_t test_vrev64q_s32(int32x4_t a) {
  return vrev64q_s32(a);
}

// CHECK: test_vrev64q_u8
// CHECK: rev64 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
uint8x16_t test_vrev64q_u8(uint8x16_t a) {
  return vrev64q_u8(a);
}

// CHECK: test_vrev64q_u16
// CHECK: rev64 v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
uint16x8_t test_vrev64q_u16(uint16x8_t a) {
  return vrev64q_u16(a);
}

// CHECK: test_vrev64q_u32
// CHECK: rev64 v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
uint32x4_t test_vrev64q_u32(uint32x4_t a) {
  return vrev64q_u32(a);
}

// CHECK: test_vrev64q_p8
// CHECK: rev64 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
poly8x16_t test_vrev64q_p8(poly8x16_t a) {
  return vrev64q_p8(a);
}

// CHECK: test_vrev64q_p16
// CHECK: rev64 v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
poly16x8_t test_vrev64q_p16(poly16x8_t a) {
  return vrev64q_p16(a);
}

// CHECK: test_vrev64q_f32
// CHECK: rev64 v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
float32x4_t test_vrev64q_f32(float32x4_t a) {
  return vrev64q_f32(a);
}

int16x4_t test_vpaddl_s8(int8x8_t a) {
  // CHECK: test_vpaddl_s8
  return vpaddl_s8(a);
  // CHECK: saddlp v{{[0-9]+}}.4h, v{{[0-9]+}}.8b
}

int32x2_t test_vpaddl_s16(int16x4_t a) {
  // CHECK: test_vpaddl_s16
  return vpaddl_s16(a);
  // CHECK: saddlp v{{[0-9]+}}.2s, v{{[0-9]+}}.4h
}

int64x1_t test_vpaddl_s32(int32x2_t a) {
  // CHECK: test_vpaddl_s32
  return vpaddl_s32(a);
  // CHECK: saddlp v{{[0-9]+}}.1d, v{{[0-9]+}}.2s
}

uint16x4_t test_vpaddl_u8(uint8x8_t a) {
  // CHECK: test_vpaddl_u8
  return vpaddl_u8(a);
  // CHECK: uaddlp v{{[0-9]+}}.4h, v{{[0-9]+}}.8b
}

uint32x2_t test_vpaddl_u16(uint16x4_t a) {
  // CHECK: test_vpaddl_u16
  return vpaddl_u16(a);
  // CHECK: uaddlp v{{[0-9]+}}.2s, v{{[0-9]+}}.4h
}

uint64x1_t test_vpaddl_u32(uint32x2_t a) {
  // CHECK: test_vpaddl_u32
  return vpaddl_u32(a);
  // CHECK: uaddlp v{{[0-9]+}}.1d, v{{[0-9]+}}.2s
}

int16x8_t test_vpaddlq_s8(int8x16_t a) {
  // CHECK: test_vpaddlq_s8
  return vpaddlq_s8(a);
  // CHECK: saddlp v{{[0-9]+}}.8h, v{{[0-9]+}}.16b
}

int32x4_t test_vpaddlq_s16(int16x8_t a) {
  // CHECK: test_vpaddlq_s16
  return vpaddlq_s16(a);
  // CHECK: saddlp v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
}

int64x2_t test_vpaddlq_s32(int32x4_t a) {
  // CHECK: test_vpaddlq_s32
  return vpaddlq_s32(a);
  // CHECK: saddlp v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
}

uint16x8_t test_vpaddlq_u8(uint8x16_t a) {
  // CHECK: test_vpaddlq_u8
  return vpaddlq_u8(a);
  // CHECK: uaddlp v{{[0-9]+}}.8h, v{{[0-9]+}}.16b
}

uint32x4_t test_vpaddlq_u16(uint16x8_t a) {
  // CHECK: test_vpaddlq_u16
  return vpaddlq_u16(a);
  // CHECK: uaddlp v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
}

uint64x2_t test_vpaddlq_u32(uint32x4_t a) {
  // CHECK: test_vpaddlq_u32
  return vpaddlq_u32(a);
  // CHECK: uaddlp v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
}

int16x4_t test_vpadal_s8(int16x4_t a, int8x8_t b) {
  // CHECK: test_vpadal_s8
  return vpadal_s8(a, b);
  // CHECK: sadalp v{{[0-9]+}}.4h, v{{[0-9]+}}.8b
}

int32x2_t test_vpadal_s16(int32x2_t a, int16x4_t b) {
  // CHECK: test_vpadal_s16
  return vpadal_s16(a, b);
  // CHECK: sadalp v{{[0-9]+}}.2s, v{{[0-9]+}}.4h
}

int64x1_t test_vpadal_s32(int64x1_t a, int32x2_t b) {
  // CHECK: test_vpadal_s32
  return vpadal_s32(a, b);
  // CHECK: sadalp v{{[0-9]+}}.1d, v{{[0-9]+}}.2s
}

uint16x4_t test_vpadal_u8(uint16x4_t a, uint8x8_t b) {
  // CHECK: test_vpadal_u8
  return vpadal_u8(a, b);
  // CHECK: uadalp v{{[0-9]+}}.4h, v{{[0-9]+}}.8b
}

uint32x2_t test_vpadal_u16(uint32x2_t a, uint16x4_t b) {
  // CHECK: test_vpadal_u16
  return vpadal_u16(a, b);
  // CHECK: uadalp v{{[0-9]+}}.2s, v{{[0-9]+}}.4h
}

uint64x1_t test_vpadal_u32(uint64x1_t a, uint32x2_t b) {
  // CHECK: test_vpadal_u32
  return vpadal_u32(a, b);
  // CHECK: uadalp v{{[0-9]+}}.1d, v{{[0-9]+}}.2s
}

int16x8_t test_vpadalq_s8(int16x8_t a, int8x16_t b) {
  // CHECK: test_vpadalq_s8
  return vpadalq_s8(a, b);
  // CHECK: sadalp v{{[0-9]+}}.8h, v{{[0-9]+}}.16b
}

int32x4_t test_vpadalq_s16(int32x4_t a, int16x8_t b) {
  // CHECK: test_vpadalq_s16
  return vpadalq_s16(a, b);
  // CHECK: sadalp v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
}

int64x2_t test_vpadalq_s32(int64x2_t a, int32x4_t b) {
  // CHECK: test_vpadalq_s32
  return vpadalq_s32(a, b);
  // CHECK: sadalp v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
}

uint16x8_t test_vpadalq_u8(uint16x8_t a, uint8x16_t b) {
  // CHECK: test_vpadalq_u8
  return vpadalq_u8(a, b);
  // CHECK: uadalp v{{[0-9]+}}.8h, v{{[0-9]+}}.16b
}

uint32x4_t test_vpadalq_u16(uint32x4_t a, uint16x8_t b) {
  // CHECK: test_vpadalq_u16
  return vpadalq_u16(a, b);
  // CHECK: uadalp v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
}

uint64x2_t test_vpadalq_u32(uint64x2_t a, uint32x4_t b) {
  // CHECK: test_vpadalq_u32
  return vpadalq_u32(a, b);
  // CHECK: uadalp v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
}

int8x8_t test_vqabs_s8(int8x8_t a) {
  // CHECK: test_vqabs_s8
  return vqabs_s8(a);
  // CHECK: sqabs v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vqabsq_s8(int8x16_t a) {
  // CHECK: test_vqabsq_s8
  return vqabsq_s8(a);
  // CHECK: sqabs v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int16x4_t test_vqabs_s16(int16x4_t a) {
  // CHECK: test_vqabs_s16
  return vqabs_s16(a);
  // CHECK: sqabs v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
}

int16x8_t test_vqabsq_s16(int16x8_t a) {
  // CHECK: test_vqabsq_s16
  return vqabsq_s16(a);
  // CHECK: sqabs v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
}

int32x2_t test_vqabs_s32(int32x2_t a) {
  // CHECK: test_vqabs_s32
  return vqabs_s32(a);
  // CHECK: sqabs v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vqabsq_s32(int32x4_t a) {
  // CHECK: test_vqabsq_s32
  return vqabsq_s32(a);
  // CHECK: sqabs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vqabsq_s64(int64x2_t a) {
  // CHECK: test_vqabsq_s64
  return vqabsq_s64(a);
  // CHECK: sqabs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int8x8_t test_vqneg_s8(int8x8_t a) {
  // CHECK: test_vqneg_s8
  return vqneg_s8(a);
  // CHECK: sqneg v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vqnegq_s8(int8x16_t a) {
  // CHECK: test_vqnegq_s8
  return vqnegq_s8(a);
  // CHECK: sqneg v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int16x4_t test_vqneg_s16(int16x4_t a) {
  // CHECK: test_vqneg_s16
  return vqneg_s16(a);
  // CHECK: sqneg v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
}

int16x8_t test_vqnegq_s16(int16x8_t a) {
  // CHECK: test_vqnegq_s16
  return vqnegq_s16(a);
  // CHECK: sqneg v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
}

int32x2_t test_vqneg_s32(int32x2_t a) {
  // CHECK: test_vqneg_s32
  return vqneg_s32(a);
  // CHECK: sqneg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vqnegq_s32(int32x4_t a) {
  // CHECK: test_vqnegq_s32
  return vqnegq_s32(a);
  // CHECK: sqneg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vqnegq_s64(int64x2_t a) {
  // CHECK: test_vqnegq_s64
  return vqnegq_s64(a);
  // CHECK: sqneg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int8x8_t test_vneg_s8(int8x8_t a) {
  // CHECK: test_vneg_s8
  return vneg_s8(a);
  // CHECK: neg v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vnegq_s8(int8x16_t a) {
  // CHECK: test_vnegq_s8
  return vnegq_s8(a);
  // CHECK: neg v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int16x4_t test_vneg_s16(int16x4_t a) {
  // CHECK: test_vneg_s16
  return vneg_s16(a);
  // CHECK: neg v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
}

int16x8_t test_vnegq_s16(int16x8_t a) {
  // CHECK: test_vnegq_s16
  return vnegq_s16(a);
  // CHECK: neg v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
}

int32x2_t test_vneg_s32(int32x2_t a) {
  // CHECK: test_vneg_s32
  return vneg_s32(a);
  // CHECK: neg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vnegq_s32(int32x4_t a) {
  // CHECK: test_vnegq_s32
  return vnegq_s32(a);
  // CHECK: neg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vnegq_s64(int64x2_t a) {
  // CHECK: test_vnegq_s64
  return vnegq_s64(a);
  // CHECK: neg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vneg_f32(float32x2_t a) {
  // CHECK: test_vneg_f32
  return vneg_f32(a);
  // CHECK: fneg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vnegq_f32(float32x4_t a) {
  // CHECK: test_vnegq_f32
  return vnegq_f32(a);
  // CHECK: fneg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vnegq_f64(float64x2_t a) {
  // CHECK: test_vnegq_f64
  return vnegq_f64(a);
  // CHECK: fneg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int8x8_t test_vabs_s8(int8x8_t a) {
  // CHECK: test_vabs_s8
  return vabs_s8(a);
  // CHECK: abs v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vabsq_s8(int8x16_t a) {
  // CHECK: test_vabsq_s8
  return vabsq_s8(a);
  // CHECK: abs v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int16x4_t test_vabs_s16(int16x4_t a) {
  // CHECK: test_vabs_s16
  return vabs_s16(a);
  // CHECK: abs v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
}

int16x8_t test_vabsq_s16(int16x8_t a) {
  // CHECK: test_vabsq_s16
  return vabsq_s16(a);
  // CHECK: abs v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
}

int32x2_t test_vabs_s32(int32x2_t a) {
  // CHECK: test_vabs_s32
  return vabs_s32(a);
  // CHECK: abs v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vabsq_s32(int32x4_t a) {
  // CHECK: test_vabsq_s32
  return vabsq_s32(a);
  // CHECK: abs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vabsq_s64(int64x2_t a) {
  // CHECK: test_vabsq_s64
  return vabsq_s64(a);
  // CHECK: abs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vabs_f32(float32x2_t a) {
  // CHECK: test_vabs_f32
  return vabs_f32(a);
  // CHECK: fabs v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vabsq_f32(float32x4_t a) {
  // CHECK: test_vabsq_f32
  return vabsq_f32(a);
  // CHECK: fabs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vabsq_f64(float64x2_t a) {
  // CHECK: test_vabsq_f64
  return vabsq_f64(a);
  // CHECK: fabs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int8x8_t test_vuqadd_s8(int8x8_t a, int8x8_t b) {
  // CHECK: test_vuqadd_s8
  return vuqadd_s8(a, b);
  // CHECK: suqadd v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vuqaddq_s8(int8x16_t a, int8x16_t b) {
  // CHECK: test_vuqaddq_s8
  return vuqaddq_s8(a, b);
  // CHECK: suqadd v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int16x4_t test_vuqadd_s16(int16x4_t a, int16x4_t b) {
  // CHECK: test_vuqadd_s16
  return vuqadd_s16(a, b);
  // CHECK: suqadd v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
}

int16x8_t test_vuqaddq_s16(int16x8_t a, int16x8_t b) {
  // CHECK: test_vuqaddq_s16
  return vuqaddq_s16(a, b);
  // CHECK: suqadd v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
}

int32x2_t test_vuqadd_s32(int32x2_t a, int32x2_t b) {
  // CHECK: test_vuqadd_s32
  return vuqadd_s32(a, b);
  // CHECK: suqadd v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vuqaddq_s32(int32x4_t a, int32x4_t b) {
  // CHECK: test_vuqaddq_s32
  return vuqaddq_s32(a, b);
  // CHECK: suqadd v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vuqaddq_s64(int64x2_t a, int64x2_t b) {
  // CHECK: test_vuqaddq_s64
  return vuqaddq_s64(a, b);
  // CHECK: suqadd v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int8x8_t test_vcls_s8(int8x8_t a) {
  // CHECK: test_vcls_s8
  return vcls_s8(a);
  // CHECK: cls v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vclsq_s8(int8x16_t a) {
  // CHECK: test_vclsq_s8
  return vclsq_s8(a);
  // CHECK: cls v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int16x4_t test_vcls_s16(int16x4_t a) {
  // CHECK: test_vcls_s16
  return vcls_s16(a);
  // CHECK: cls v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
}

int16x8_t test_vclsq_s16(int16x8_t a) {
  // CHECK: test_vclsq_s16
  return vclsq_s16(a);
  // CHECK: cls v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
}

int32x2_t test_vcls_s32(int32x2_t a) {
  // CHECK: test_vcls_s32
  return vcls_s32(a);
  // CHECK: cls v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vclsq_s32(int32x4_t a) {
  // CHECK: test_vclsq_s32
  return vclsq_s32(a);
  // CHECK: cls v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int8x8_t test_vclz_s8(int8x8_t a) {
  // CHECK: test_vclz_s8
  return vclz_s8(a);
  // CHECK: clz v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vclzq_s8(int8x16_t a) {
  // CHECK: test_vclzq_s8
  return vclzq_s8(a);
  // CHECK: clz v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int16x4_t test_vclz_s16(int16x4_t a) {
  // CHECK: test_vclz_s16
  return vclz_s16(a);
  // CHECK: clz v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
}

int16x8_t test_vclzq_s16(int16x8_t a) {
  // CHECK: test_vclzq_s16
  return vclzq_s16(a);
  // CHECK: clz v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
}

int32x2_t test_vclz_s32(int32x2_t a) {
  // CHECK: test_vclz_s32
  return vclz_s32(a);
  // CHECK: clz v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vclzq_s32(int32x4_t a) {
  // CHECK: test_vclzq_s32
  return vclzq_s32(a);
  // CHECK: clz v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

uint8x8_t test_vclz_u8(uint8x8_t a) {
  // CHECK: test_vclz_u8
  return vclz_u8(a);
  // CHECK: clz v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

uint8x16_t test_vclzq_u8(uint8x16_t a) {
  // CHECK: test_vclzq_u8
  return vclzq_u8(a);
  // CHECK: clz v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

uint16x4_t test_vclz_u16(uint16x4_t a) {
  // CHECK: test_vclz_u16
  return vclz_u16(a);
  // CHECK: clz v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
}

uint16x8_t test_vclzq_u16(uint16x8_t a) {
  // CHECK: test_vclzq_u16
  return vclzq_u16(a);
  // CHECK: clz v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
}

uint32x2_t test_vclz_u32(uint32x2_t a) {
  // CHECK: test_vclz_u32
  return vclz_u32(a);
  // CHECK: clz v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

uint32x4_t test_vclzq_u32(uint32x4_t a) {
  // CHECK: test_vclzq_u32
  return vclzq_u32(a);
  // CHECK: clz v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int8x8_t test_vcnt_s8(int8x8_t a) {
  // CHECK: test_vcnt_s8
  return vcnt_s8(a);
  // CHECK: cnt v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vcntq_s8(int8x16_t a) {
  // CHECK: test_vcntq_s8
  return vcntq_s8(a);
  // CHECK: cnt v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

uint8x8_t test_vcnt_u8(uint8x8_t a) {
  // CHECK: test_vcnt_u8
  return vcnt_u8(a);
  // CHECK: cnt v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

uint8x16_t test_vcntq_u8(uint8x16_t a) {
  // CHECK: test_vcntq_u8
  return vcntq_u8(a);
  // CHECK: cnt v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

poly8x8_t test_vcnt_p8(poly8x8_t a) {
  // CHECK: test_vcnt_p8
  return vcnt_p8(a);
  // CHECK: cnt v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

poly8x16_t test_vcntq_p8(poly8x16_t a) {
  // CHECK: test_vcntq_p8
  return vcntq_p8(a);
  // CHECK: cnt v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int8x8_t test_vmvn_s8(int8x8_t a) {
  // CHECK: test_vmvn_s8
  return vmvn_s8(a);
  // CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vmvnq_s8(int8x16_t a) {
  // CHECK: test_vmvnq_s8
  return vmvnq_s8(a);
  // CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int16x4_t test_vmvn_s16(int16x4_t a) {
  // CHECK: test_vmvn_s16
  return vmvn_s16(a);
  // CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int16x8_t test_vmvnq_s16(int16x8_t a) {
  // CHECK: test_vmvnq_s16
  return vmvnq_s16(a);
  // CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int32x2_t test_vmvn_s32(int32x2_t a) {
  // CHECK: test_vmvn_s32
  return vmvn_s32(a);
  // CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int32x4_t test_vmvnq_s32(int32x4_t a) {
  // CHECK: test_vmvnq_s32
  return vmvnq_s32(a);
  // CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

uint8x8_t test_vmvn_u8(uint8x8_t a) {
  // CHECK: test_vmvn_u8
  return vmvn_u8(a);
  // CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

uint8x16_t test_vmvnq_u8(uint8x16_t a) {
  // CHECK: test_vmvnq_u8
  return vmvnq_u8(a);
  // CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

uint16x4_t test_vmvn_u16(uint16x4_t a) {
  // CHECK: test_vmvn_u16
  return vmvn_u16(a);
  // CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

uint16x8_t test_vmvnq_u16(uint16x8_t a) {
  // CHECK: test_vmvnq_u16
  return vmvnq_u16(a);
  // CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

uint32x2_t test_vmvn_u32(uint32x2_t a) {
  // CHECK: test_vmvn_u32
  return vmvn_u32(a);
  // CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

uint32x4_t test_vmvnq_u32(uint32x4_t a) {
  // CHECK: test_vmvnq_u32
  return vmvnq_u32(a);
  // CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

poly8x8_t test_vmvn_p8(poly8x8_t a) {
  // CHECK: test_vmvn_p8
  return vmvn_p8(a);
  // CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

poly8x16_t test_vmvnq_p8(poly8x16_t a) {
  // CHECK: test_vmvnq_p8
  return vmvnq_p8(a);
  // CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int8x8_t test_vrbit_s8(int8x8_t a) {
  // CHECK: test_vrbit_s8
  return vrbit_s8(a);
  // CHECK: rbit v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

int8x16_t test_vrbitq_s8(int8x16_t a) {
  // CHECK: test_vrbitq_s8
  return vrbitq_s8(a);
  // CHECK: rbit v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

uint8x8_t test_vrbit_u8(uint8x8_t a) {
  // CHECK: test_vrbit_u8
  return vrbit_u8(a);
  // CHECK: rbit v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

uint8x16_t test_vrbitq_u8(uint8x16_t a) {
  // CHECK: test_vrbitq_u8
  return vrbitq_u8(a);
  // CHECK: rbit v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

poly8x8_t test_vrbit_p8(poly8x8_t a) {
  // CHECK: test_vrbit_p8
  return vrbit_p8(a);
  // CHECK: rbit v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
}

poly8x16_t test_vrbitq_p8(poly8x16_t a) {
  // CHECK: test_vrbitq_p8
  return vrbitq_p8(a);
  // CHECK: rbit v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
}

int8x8_t test_vmovn_s16(int16x8_t a) {
  // CHECK: test_vmovn_s16
  return vmovn_s16(a);
  // CHECK: xtn v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
}

int16x4_t test_vmovn_s32(int32x4_t a) {
  // CHECK: test_vmovn_s32
  return vmovn_s32(a);
  // CHECK: xtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
}

int32x2_t test_vmovn_s64(int64x2_t a) {
  // CHECK: test_vmovn_s64
  return vmovn_s64(a);
  // CHECK: xtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
}

uint8x8_t test_vmovn_u16(uint16x8_t a) {
  // CHECK: test_vmovn_u16
  return vmovn_u16(a);
  // CHECK: xtn v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
}

uint16x4_t test_vmovn_u32(uint32x4_t a) {
  // CHECK: test_vmovn_u32
  return vmovn_u32(a);
  // CHECK: xtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
}

uint32x2_t test_vmovn_u64(uint64x2_t a) {
  // CHECK: test_vmovn_u64
  return vmovn_u64(a);
  // CHECK: xtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
}

int8x16_t test_vmovn_high_s16(int8x8_t a, int16x8_t b) {
  // CHECK: test_vmovn_high_s16
  return vmovn_high_s16(a, b);
  // CHECK: xtn2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
}

int16x8_t test_vmovn_high_s32(int16x4_t a, int32x4_t b) {
  // CHECK: test_vmovn_high_s32
  return vmovn_high_s32(a, b);
  // CHECK: xtn2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
}

int32x4_t test_vmovn_high_s64(int32x2_t a, int64x2_t b) {
  // CHECK: test_vmovn_high_s64
  return vmovn_high_s64(a, b);
  // CHECK: xtn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
}

int8x16_t test_vmovn_high_u16(int8x8_t a, int16x8_t b) {
  // CHECK: test_vmovn_high_u16
  return vmovn_high_u16(a, b);
  // CHECK: xtn2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
}

int16x8_t test_vmovn_high_u32(int16x4_t a, int32x4_t b) {
  // CHECK: test_vmovn_high_u32
  return vmovn_high_u32(a, b);
  // CHECK: xtn2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
}

int32x4_t test_vmovn_high_u64(int32x2_t a, int64x2_t b) {
  // CHECK: test_vmovn_high_u64
  return vmovn_high_u64(a, b);
  // CHECK: xtn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
}

int8x8_t test_vqmovun_s16(int16x8_t a) {
  // CHECK: test_vqmovun_s16
  return vqmovun_s16(a);
  // CHECK: sqxtun v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
}

int16x4_t test_vqmovun_s32(int32x4_t a) {
  // CHECK: test_vqmovun_s32
  return vqmovun_s32(a);
  // CHECK: sqxtun v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
}

int32x2_t test_vqmovun_s64(int64x2_t a) {
  // CHECK: test_vqmovun_s64
  return vqmovun_s64(a);
  // CHECK: sqxtun v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
}

int8x16_t test_vqmovun_high_s16(int8x8_t a, int16x8_t b) {
  // CHECK: test_vqmovun_high_s16
  return vqmovun_high_s16(a, b);
  // CHECK: sqxtun2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
}

int16x8_t test_vqmovun_high_s32(int16x4_t a, int32x4_t b) {
  // CHECK: test_vqmovun_high_s32
  return vqmovun_high_s32(a, b);
  // CHECK: sqxtun2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
}

int32x4_t test_vqmovun_high_s64(int32x2_t a, int64x2_t b) {
  // CHECK: test_vqmovun_high_s64
  return vqmovun_high_s64(a, b);
  // CHECK: sqxtun2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
}

int8x8_t test_vqmovn_s16(int16x8_t a) {
  // CHECK: test_vqmovn_s16
  return vqmovn_s16(a);
  // CHECK: sqxtn v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
}

int16x4_t test_vqmovn_s32(int32x4_t a) {
  // CHECK: test_vqmovn_s32
  return vqmovn_s32(a);
  // CHECK: sqxtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
}

int32x2_t test_vqmovn_s64(int64x2_t a) {
  // CHECK: test_vqmovn_s64
  return vqmovn_s64(a);
  // CHECK: sqxtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
}

int8x16_t test_vqmovn_high_s16(int8x8_t a, int16x8_t b) {
  // CHECK: test_vqmovn_high_s16
  return vqmovn_high_s16(a, b);
  // CHECK: sqxtn2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
}

int16x8_t test_vqmovn_high_s32(int16x4_t a, int32x4_t b) {
  // CHECK: test_vqmovn_high_s32
  return vqmovn_high_s32(a, b);
  // CHECK: sqxtn2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
}

int32x4_t test_vqmovn_high_s64(int32x2_t a, int64x2_t b) {
  // CHECK: test_vqmovn_high_s64
  return vqmovn_high_s64(a, b);
  // CHECK: sqxtn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
}

uint8x8_t test_vqmovn_u16(uint16x8_t a) {
  // CHECK: test_vqmovn_u16
  return vqmovn_u16(a);
  // CHECK: uqxtn v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
}

uint16x4_t test_vqmovn_u32(uint32x4_t a) {
  // CHECK: test_vqmovn_u32
  return vqmovn_u32(a);
  // CHECK: uqxtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
}

uint32x2_t test_vqmovn_u64(uint64x2_t a) {
  // CHECK: test_vqmovn_u64
  return vqmovn_u64(a);
  // CHECK: uqxtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
}

uint8x16_t test_vqmovn_high_u16(uint8x8_t a, uint16x8_t b) {
  // CHECK: test_vqmovn_high_u16
  return vqmovn_high_u16(a, b);
  // CHECK: uqxtn2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
}

uint16x8_t test_vqmovn_high_u32(uint16x4_t a, uint32x4_t b) {
  // CHECK: test_vqmovn_high_u32
  return vqmovn_high_u32(a, b);
  // CHECK: uqxtn2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
}

uint32x4_t test_vqmovn_high_u64(uint32x2_t a, uint64x2_t b) {
  // CHECK: test_vqmovn_high_u64
  return vqmovn_high_u64(a, b);
  // CHECK: uqxtn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
}

int16x8_t test_vshll_n_s8(int8x8_t a) {
  // CHECK: test_vshll_n_s8
  return vshll_n_s8(a, 8);
  // CHECK: shll {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, #8
}

int32x4_t test_vshll_n_s16(int16x4_t a) {
  // CHECK: test_vshll_n_s16
  return vshll_n_s16(a, 16);
  // CHECK: shll {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, #16
}

int64x2_t test_vshll_n_s32(int32x2_t a) {
  // CHECK: test_vshll_n_s32
  return vshll_n_s32(a, 32);
  // CHECK: shll {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, #32
}

uint16x8_t test_vshll_n_u8(uint8x8_t a) {
  // CHECK: test_vshll_n_u8
  return vshll_n_u8(a, 8);
  // CHECK: shll {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, #8
}

uint32x4_t test_vshll_n_u16(uint16x4_t a) {
  // CHECK: test_vshll_n_u16
  return vshll_n_u16(a, 16);
  // CHECK: shll {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, #16
}

uint64x2_t test_vshll_n_u32(uint32x2_t a) {
  // CHECK: test_vshll_n_u32
  return vshll_n_u32(a, 32);
  // CHECK: shll {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, #32
}

int16x8_t test_vshll_high_n_s8(int8x16_t a) {
  // CHECK: test_vshll_high_n_s8
  return vshll_high_n_s8(a, 8);
  // CHECK: shll2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, #8
}

int32x4_t test_vshll_high_n_s16(int16x8_t a) {
  // CHECK: test_vshll_high_n_s16
  return vshll_high_n_s16(a, 16);
  // CHECK: shll2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, #16
}

int64x2_t test_vshll_high_n_s32(int32x4_t a) {
  // CHECK: test_vshll_high_n_s32
  return vshll_high_n_s32(a, 32);
  // CHECK: shll2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, #32
}

uint16x8_t test_vshll_high_n_u8(uint8x16_t a) {
  // CHECK: test_vshll_high_n_u8
  return vshll_high_n_u8(a, 8);
  // CHECK: shll2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, #8
}

uint32x4_t test_vshll_high_n_u16(uint16x8_t a) {
  // CHECK: test_vshll_high_n_u16
  return vshll_high_n_u16(a, 16);
  // CHECK: shll2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, #16
}

uint64x2_t test_vshll_high_n_u32(uint32x4_t a) {
  // CHECK: test_vshll_high_n_u32
  return vshll_high_n_u32(a, 32);
  // CHECK: shll2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, #32
}

float16x4_t test_vcvt_f16_f32(float32x4_t a) {
  //CHECK: test_vcvt_f16_f32
  return vcvt_f16_f32(a);
  // CHECK: fcvtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
}

float16x8_t test_vcvt_high_f16_f32(float16x4_t a, float32x4_t b) {
  //CHECK: test_vcvt_high_f16_f32
  return vcvt_high_f16_f32(a, b);
  // CHECK: fcvtn2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
}

float32x2_t test_vcvt_f32_f64(float64x2_t a) {
  //CHECK: test_vcvt_f32_f64
  return vcvt_f32_f64(a);
  // CHECK: fcvtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
}

float32x4_t test_vcvt_high_f32_f64(float32x2_t a, float64x2_t b) {
  //CHECK: test_vcvt_high_f32_f64
  return vcvt_high_f32_f64(a, b);
  // CHECK: fcvtn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
}

float32x2_t test_vcvtx_f32_f64(float64x2_t a) {
  //CHECK: test_vcvtx_f32_f64
  return vcvtx_f32_f64(a);
  // CHECK: fcvtxn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
}

float32x4_t test_vcvtx_high_f32_f64(float32x2_t a, float64x2_t b) {
  //CHECK: test_vcvtx_high_f32_f64
  return vcvtx_high_f32_f64(a, b);
  // CHECK: fcvtxn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
}

float32x4_t test_vcvt_f32_f16(float16x4_t a) {
  //CHECK: test_vcvt_f32_f16
  return vcvt_f32_f16(a);
  // CHECK: fcvtl v{{[0-9]+}}.4s, v{{[0-9]+}}.4h
}

float32x4_t test_vcvt_high_f32_f16(float16x8_t a) {
  //CHECK: test_vcvt_high_f32_f16
  return vcvt_high_f32_f16(a);
  // CHECK: fcvtl2 v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
}

float64x2_t test_vcvt_f64_f32(float32x2_t a) {
  //CHECK: test_vcvt_f64_f32
  return vcvt_f64_f32(a);
  // CHECK: fcvtl v{{[0-9]+}}.2d, v{{[0-9]+}}.2s
}

float64x2_t test_vcvt_high_f64_f32(float32x4_t a) {
  //CHECK: test_vcvt_high_f64_f32
  return vcvt_high_f64_f32(a);
  // CHECK: fcvtl2 v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
}

float32x2_t test_vrndn_f32(float32x2_t a) {
  //CHECK: test_vrndn_f32
  return vrndn_f32(a);
  // CHECK: frintn v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrndnq_f32(float32x4_t a) {
  //CHECK: test_vrndnq_f32
  return vrndnq_f32(a);
  // CHECK: frintn v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrndnq_f64(float64x2_t a) {
  //CHECK: test_vrndnq_f64
  return vrndnq_f64(a);
  // CHECK: frintn v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vrnda_f32(float32x2_t a) {
  //CHECK: test_vrnda_f32
  return vrnda_f32(a);
  // CHECK: frinta v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrndaq_f32(float32x4_t a) {
  //CHECK: test_vrndaq_f32
  return vrndaq_f32(a);
  // CHECK: frinta v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrndaq_f64(float64x2_t a) {
  //CHECK: test_vrndaq_f64
  return vrndaq_f64(a);
  // CHECK: frinta v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vrndp_f32(float32x2_t a) {
  //CHECK: test_vrndp_f32
  return vrndp_f32(a);
  // CHECK: frintp v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrndpq_f32(float32x4_t a) {
  //CHECK: test_vrndpq_f32
  return vrndpq_f32(a);
  // CHECK: frintp v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrndpq_f64(float64x2_t a) {
  //CHECK: test_vrndpq_f64
  return vrndpq_f64(a);
  // CHECK: frintp v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vrndm_f32(float32x2_t a) {
  //CHECK: test_vrndm_f32
  return vrndm_f32(a);
  // CHECK: frintm v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrndmq_f32(float32x4_t a) {
  //CHECK: test_vrndmq_f32
  return vrndmq_f32(a);
  // CHECK: frintm v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrndmq_f64(float64x2_t a) {
  //CHECK: test_vrndmq_f64
  return vrndmq_f64(a);
  // CHECK: frintm v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vrndx_f32(float32x2_t a) {
  //CHECK: test_vrndx_f32
  return vrndx_f32(a);
  // CHECK: frintx v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrndxq_f32(float32x4_t a) {
  //CHECK: test_vrndxq_f32
  return vrndxq_f32(a);
  // CHECK: frintx v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrndxq_f64(float64x2_t a) {
  //CHECK: test_vrndxq_f64
  return vrndxq_f64(a);
  // CHECK: frintx v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vrnd_f32(float32x2_t a) {
  //CHECK: test_vrnd_f32
  return vrnd_f32(a);
  // CHECK: frintz v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrndq_f32(float32x4_t a) {
  //CHECK: test_vrndq_f32
  return vrndq_f32(a);
  // CHECK: frintz v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrndq_f64(float64x2_t a) {
  //CHECK: test_vrndq_f64
  return vrndq_f64(a);
  // CHECK: frintz v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vrndi_f32(float32x2_t a) {
  //CHECK: test_vrndi_f32
  return vrndi_f32(a);
  // CHECK: frinti v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrndiq_f32(float32x4_t a) {
  //CHECK: test_vrndiq_f32
  return vrndiq_f32(a);
  // CHECK: frinti v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrndiq_f64(float64x2_t a) {
  //CHECK: test_vrndiq_f64
  return vrndiq_f64(a);
  // CHECK: frinti v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int32x2_t test_vcvt_s32_f32(float32x2_t a) {
  //CHECK: test_vcvt_s32_f32
  return vcvt_s32_f32(a);
  // CHECK: fcvtzs v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vcvtq_s32_f32(float32x4_t a) {
  //CHECK: test_vcvtq_s32_f32
  return vcvtq_s32_f32(a);
  // CHECK: fcvtzs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vcvtq_s64_f64(float64x2_t a) {
  //CHECK: test_vcvtq_s64_f64
  return vcvtq_s64_f64(a);
  // CHECK: fcvtzs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

uint32x2_t test_vcvt_u32_f32(float32x2_t a) {
  //CHECK: test_vcvt_u32_f32
  return vcvt_u32_f32(a);
  // CHECK: fcvtzu v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

uint32x4_t test_vcvtq_u32_f32(float32x4_t a) {
  //CHECK: test_vcvtq_u32_f32
  return vcvtq_u32_f32(a);
  // CHECK: fcvtzu v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

uint64x2_t test_vcvtq_u64_f64(float64x2_t a) {
  //CHECK: test_vcvtq_u64_f64
  return vcvtq_u64_f64(a);
  // CHECK: fcvtzu v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int32x2_t test_vcvtn_s32_f32(float32x2_t a) {
  //CHECK: test_vcvtn_s32_f32
  return vcvtn_s32_f32(a);
  // CHECK: fcvtns v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vcvtnq_s32_f32(float32x4_t a) {
  //CHECK: test_vcvtnq_s32_f32
  return vcvtnq_s32_f32(a);
  // CHECK: fcvtns v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vcvtnq_s64_f64(float64x2_t a) {
  //CHECK: test_vcvtnq_s64_f64
  return vcvtnq_s64_f64(a);
  // CHECK: fcvtns v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

uint32x2_t test_vcvtn_u32_f32(float32x2_t a) {
  //CHECK: test_vcvtn_u32_f32
  return vcvtn_u32_f32(a);
  // CHECK: fcvtnu v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

uint32x4_t test_vcvtnq_u32_f32(float32x4_t a) {
  //CHECK: test_vcvtnq_u32_f32
  return vcvtnq_u32_f32(a);
  // CHECK: fcvtnu v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

uint64x2_t test_vcvtnq_u64_f64(float64x2_t a) {
  //CHECK: test_vcvtnq_u64_f64
  return vcvtnq_u64_f64(a);
  // CHECK: fcvtnu v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int32x2_t test_vcvtp_s32_f32(float32x2_t a) {
  //CHECK: test_vcvtp_s32_f32
  return vcvtp_s32_f32(a);
  // CHECK: fcvtps v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vcvtpq_s32_f32(float32x4_t a) {
  //CHECK: test_vcvtpq_s32_f32
  return vcvtpq_s32_f32(a);
  // CHECK: fcvtps v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vcvtpq_s64_f64(float64x2_t a) {
  //CHECK: test_vcvtpq_s64_f64
  return vcvtpq_s64_f64(a);
  // CHECK: fcvtps v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

uint32x2_t test_vcvtp_u32_f32(float32x2_t a) {
  //CHECK: test_vcvtp_u32_f32
  return vcvtp_u32_f32(a);
  // CHECK: fcvtpu v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

uint32x4_t test_vcvtpq_u32_f32(float32x4_t a) {
  //CHECK: test_vcvtpq_u32_f32
  return vcvtpq_u32_f32(a);
  // CHECK: fcvtpu v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

uint64x2_t test_vcvtpq_u64_f64(float64x2_t a) {
  //CHECK: test_vcvtpq_u64_f64
  return vcvtpq_u64_f64(a);
  // CHECK: fcvtpu v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int32x2_t test_vcvtm_s32_f32(float32x2_t a) {
  //CHECK: test_vcvtm_s32_f32
  return vcvtm_s32_f32(a);
  // CHECK: fcvtms v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vcvtmq_s32_f32(float32x4_t a) {
  //CHECK: test_vcvtmq_s32_f32
  return vcvtmq_s32_f32(a);
  // CHECK: fcvtms v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vcvtmq_s64_f64(float64x2_t a) {
  //CHECK: test_vcvtmq_s64_f64
  return vcvtmq_s64_f64(a);
  // CHECK: fcvtms v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

uint32x2_t test_vcvtm_u32_f32(float32x2_t a) {
  //CHECK: test_vcvtm_u32_f32
  return vcvtm_u32_f32(a);
  // CHECK: fcvtmu v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

uint32x4_t test_vcvtmq_u32_f32(float32x4_t a) {
  //CHECK: test_vcvtmq_u32_f32
  return vcvtmq_u32_f32(a);
  // CHECK: fcvtmu v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

uint64x2_t test_vcvtmq_u64_f64(float64x2_t a) {
  //CHECK: test_vcvtmq_u64_f64
  return vcvtmq_u64_f64(a);
  // CHECK: fcvtmu v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

int32x2_t test_vcvta_s32_f32(float32x2_t a) {
  //CHECK: test_vcvta_s32_f32
  return vcvta_s32_f32(a);
  // CHECK: fcvtas v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

int32x4_t test_vcvtaq_s32_f32(float32x4_t a) {
  //CHECK: test_vcvtaq_s32_f32
  return vcvtaq_s32_f32(a);
  // CHECK: fcvtas v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

int64x2_t test_vcvtaq_s64_f64(float64x2_t a) {
  //CHECK: test_vcvtaq_s64_f64
  return vcvtaq_s64_f64(a);
  // CHECK: fcvtas v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

uint32x2_t test_vcvta_u32_f32(float32x2_t a) {
  //CHECK: test_vcvta_u32_f32
  return vcvta_u32_f32(a);
  // CHECK: fcvtau v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

uint32x4_t test_vcvtaq_u32_f32(float32x4_t a) {
  //CHECK: test_vcvtaq_u32_f32
  return vcvtaq_u32_f32(a);
  // CHECK: fcvtau v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

uint64x2_t test_vcvtaq_u64_f64(float64x2_t a) {
  //CHECK: test_vcvtaq_u64_f64
  return vcvtaq_u64_f64(a);
  // CHECK: fcvtau v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vrsqrte_f32(float32x2_t a) {
  //CHECK: test_vrsqrte_f32
  return vrsqrte_f32(a);
  // CHECK: frsqrte v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrsqrteq_f32(float32x4_t a) {
  //CHECK: test_vrsqrteq_f32
  return vrsqrteq_f32(a);
  // CHECK: frsqrte v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrsqrteq_f64(float64x2_t a) {
  //CHECK: test_vrsqrteq_f64
  return vrsqrteq_f64(a);
  // CHECK: frsqrte v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vrecpe_f32(float32x2_t a) {
  //CHECK: test_vrecpe_f32
  return vrecpe_f32(a);
  // CHECK: frecpe v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vrecpeq_f32(float32x4_t a) {
  //CHECK: test_vrecpeq_f32
  return vrecpeq_f32(a);
  // CHECK: frecpe v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vrecpeq_f64(float64x2_t a) {
  //CHECK: test_vrecpeq_f64
  return vrecpeq_f64(a);
  // CHECK: frecpe v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

uint32x2_t test_vrecpe_u32(uint32x2_t a) {
  //CHECK: test_vrecpe_u32
  return vrecpe_u32(a);
  // CHECK: urecpe v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

uint32x4_t test_vrecpeq_u32(uint32x4_t a) {
  //CHECK: test_vrecpeq_u32
  return vrecpeq_u32(a);
  // CHECK: urecpe v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float32x2_t test_vsqrt_f32(float32x2_t a) {
  //CHECK: test_vsqrt_f32
  return vsqrt_f32(a);
  // CHECK: fsqrt v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vsqrtq_f32(float32x4_t a) {
  //CHECK: test_vsqrtq_f32
  return vsqrtq_f32(a);
  // CHECK: fsqrt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vsqrtq_f64(float64x2_t a) {
  //CHECK: test_vsqrtq_f64
  return vsqrtq_f64(a);
  // CHECK: fsqrt v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float32x2_t test_vcvt_f32_s32(int32x2_t a) {
  //CHECK: test_vcvt_f32_s32
  return vcvt_f32_s32(a);
  //CHECK: scvtf v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x2_t test_vcvt_f32_u32(uint32x2_t a) {
  //CHECK: test_vcvt_f32_u32
  return vcvt_f32_u32(a);
  //CHECK: ucvtf v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
}

float32x4_t test_vcvtq_f32_s32(int32x4_t a) {
  //CHECK: test_vcvtq_f32_s32
  return vcvtq_f32_s32(a);
  //CHECK: scvtf v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float32x4_t test_vcvtq_f32_u32(uint32x4_t a) {
  //CHECK: test_vcvtq_f32_u32
  return vcvtq_f32_u32(a);
  //CHECK: ucvtf v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
}

float64x2_t test_vcvtq_f64_s64(int64x2_t a) {
  //CHECK: test_vcvtq_f64_s64
  return vcvtq_f64_s64(a);
  //CHECK: scvtf v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}

float64x2_t test_vcvtq_f64_u64(uint64x2_t a) {
  //CHECK: test_vcvtq_f64_u64
  return vcvtq_f64_u64(a);
  //CHECK: ucvtf v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
}
