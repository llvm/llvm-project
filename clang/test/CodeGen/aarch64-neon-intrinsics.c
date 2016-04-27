// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ARM64

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int8x8_t test_vadd_s8(int8x8_t v1, int8x8_t v2) {
   // CHECK-LABEL: test_vadd_s8
  return vadd_s8(v1, v2);
  // CHECK: add {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vadd_s16(int16x4_t v1, int16x4_t v2) {
   // CHECK-LABEL: test_vadd_s16
  return vadd_s16(v1, v2);
  // CHECK: add {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vadd_s32(int32x2_t v1, int32x2_t v2) {
   // CHECK-LABEL: test_vadd_s32
  return vadd_s32(v1, v2);
  // CHECK: add {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vadd_s64(int64x1_t v1, int64x1_t v2) {
  // CHECK-LABEL: test_vadd_s64
  return vadd_s64(v1, v2);
  // CHECK: add {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

float32x2_t test_vadd_f32(float32x2_t v1, float32x2_t v2) {
   // CHECK-LABEL: test_vadd_f32
  return vadd_f32(v1, v2);
  // CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vadd_u8(uint8x8_t v1, uint8x8_t v2) {
   // CHECK-LABEL: test_vadd_u8
  return vadd_u8(v1, v2);
  // CHECK: add {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vadd_u16(uint16x4_t v1, uint16x4_t v2) {
   // CHECK-LABEL: test_vadd_u16
  return vadd_u16(v1, v2);
  // CHECK: add {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vadd_u32(uint32x2_t v1, uint32x2_t v2) {
   // CHECK-LABEL: test_vadd_u32
  return vadd_u32(v1, v2);
  // CHECK: add {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vadd_u64(uint64x1_t v1, uint64x1_t v2) {
   // CHECK-LABEL: test_vadd_u64
  return vadd_u64(v1, v2);
  // CHECK: add {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vaddq_s8(int8x16_t v1, int8x16_t v2) {
   // CHECK-LABEL: test_vaddq_s8
  return vaddq_s8(v1, v2);
  // CHECK: add {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vaddq_s16(int16x8_t v1, int16x8_t v2) {
   // CHECK-LABEL: test_vaddq_s16
  return vaddq_s16(v1, v2);
  // CHECK: add {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vaddq_s32(int32x4_t v1,int32x4_t  v2) {
   // CHECK-LABEL: test_vaddq_s32
  return vaddq_s32(v1, v2);
  // CHECK: add {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vaddq_s64(int64x2_t v1, int64x2_t v2) {
   // CHECK-LABEL: test_vaddq_s64
  return vaddq_s64(v1, v2);
  // CHECK: add {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x4_t test_vaddq_f32(float32x4_t v1, float32x4_t v2) {
   // CHECK-LABEL: test_vaddq_f32
  return vaddq_f32(v1, v2);
  // CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vaddq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vaddq_f64
  return vaddq_f64(v1, v2);
  // CHECK: fadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vaddq_u8(uint8x16_t v1, uint8x16_t v2) {
   // CHECK-LABEL: test_vaddq_u8
  return vaddq_u8(v1, v2);
  // CHECK: add {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vaddq_u16(uint16x8_t v1, uint16x8_t v2) {
   // CHECK-LABEL: test_vaddq_u16
  return vaddq_u16(v1, v2);
  // CHECK: add {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vaddq_u32(uint32x4_t v1, uint32x4_t v2) {
   // CHECK: vaddq_u32
  return vaddq_u32(v1, v2);
  // CHECK: add {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vaddq_u64(uint64x2_t v1, uint64x2_t v2) {
   // CHECK-LABEL: test_vaddq_u64
  return vaddq_u64(v1, v2);
  // CHECK: add {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vsub_s8(int8x8_t v1, int8x8_t v2) {
   // CHECK-LABEL: test_vsub_s8
  return vsub_s8(v1, v2);
  // CHECK: sub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
int16x4_t test_vsub_s16(int16x4_t v1, int16x4_t v2) {
   // CHECK-LABEL: test_vsub_s16
  return vsub_s16(v1, v2);
  // CHECK: sub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int32x2_t test_vsub_s32(int32x2_t v1, int32x2_t v2) {
   // CHECK-LABEL: test_vsub_s32
  return vsub_s32(v1, v2);
  // CHECK: sub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vsub_s64(int64x1_t v1, int64x1_t v2) {
   // CHECK-LABEL: test_vsub_s64
  return vsub_s64(v1, v2);
  // CHECK: sub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

float32x2_t test_vsub_f32(float32x2_t v1, float32x2_t v2) {
   // CHECK-LABEL: test_vsub_f32
  return vsub_f32(v1, v2);
  // CHECK: fsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vsub_u8(uint8x8_t v1, uint8x8_t v2) {
   // CHECK-LABEL: test_vsub_u8
  return vsub_u8(v1, v2);
  // CHECK: sub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vsub_u16(uint16x4_t v1, uint16x4_t v2) {
   // CHECK-LABEL: test_vsub_u16
  return vsub_u16(v1, v2);
  // CHECK: sub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vsub_u32(uint32x2_t v1, uint32x2_t v2) {
   // CHECK-LABEL: test_vsub_u32
  return vsub_u32(v1, v2);
  // CHECK: sub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vsub_u64(uint64x1_t v1, uint64x1_t v2) {
   // CHECK-LABEL: test_vsub_u64
  return vsub_u64(v1, v2);
  // CHECK: sub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vsubq_s8(int8x16_t v1, int8x16_t v2) {
   // CHECK-LABEL: test_vsubq_s8
  return vsubq_s8(v1, v2);
  // CHECK: sub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vsubq_s16(int16x8_t v1, int16x8_t v2) {
   // CHECK-LABEL: test_vsubq_s16
  return vsubq_s16(v1, v2);
  // CHECK: sub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vsubq_s32(int32x4_t v1,int32x4_t  v2) {
   // CHECK-LABEL: test_vsubq_s32
  return vsubq_s32(v1, v2);
  // CHECK: sub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vsubq_s64(int64x2_t v1, int64x2_t v2) {
   // CHECK-LABEL: test_vsubq_s64
  return vsubq_s64(v1, v2);
  // CHECK: sub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x4_t test_vsubq_f32(float32x4_t v1, float32x4_t v2) {
   // CHECK-LABEL: test_vsubq_f32
  return vsubq_f32(v1, v2);
  // CHECK: fsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vsubq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vsubq_f64
  return vsubq_f64(v1, v2);
  // CHECK: fsub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vsubq_u8(uint8x16_t v1, uint8x16_t v2) {
   // CHECK-LABEL: test_vsubq_u8
  return vsubq_u8(v1, v2);
  // CHECK: sub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vsubq_u16(uint16x8_t v1, uint16x8_t v2) {
   // CHECK-LABEL: test_vsubq_u16
  return vsubq_u16(v1, v2);
  // CHECK: sub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vsubq_u32(uint32x4_t v1, uint32x4_t v2) {
   // CHECK: vsubq_u32
  return vsubq_u32(v1, v2);
  // CHECK: sub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vsubq_u64(uint64x2_t v1, uint64x2_t v2) {
   // CHECK-LABEL: test_vsubq_u64
  return vsubq_u64(v1, v2);
  // CHECK: sub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vmul_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK-LABEL: test_vmul_s8
  return vmul_s8(v1, v2);
  // CHECK: mul {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vmul_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK-LABEL: test_vmul_s16
  return vmul_s16(v1, v2);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmul_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK-LABEL: test_vmul_s32
  return vmul_s32(v1, v2);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmul_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vmul_f32
  return vmul_f32(v1, v2);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}


uint8x8_t test_vmul_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK-LABEL: test_vmul_u8
  return vmul_u8(v1, v2);
  // CHECK: mul {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmul_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK-LABEL: test_vmul_u16
  return vmul_u16(v1, v2);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmul_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK-LABEL: test_vmul_u32
  return vmul_u32(v1, v2);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vmulq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK-LABEL: test_vmulq_s8
  return vmulq_s8(v1, v2);
  // CHECK: mul {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vmulq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK-LABEL: test_vmulq_s16
  return vmulq_s16(v1, v2);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vmulq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK-LABEL: test_vmulq_s32
  return vmulq_s32(v1, v2);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
    
uint8x16_t test_vmulq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK-LABEL: test_vmulq_u8
  return vmulq_u8(v1, v2);
  // CHECK: mul {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vmulq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK-LABEL: test_vmulq_u16
  return vmulq_u16(v1, v2);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vmulq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK-LABEL: test_vmulq_u32
  return vmulq_u32(v1, v2);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vmulq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vmulq_f32
  return vmulq_f32(v1, v2);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmulq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vmulq_f64
  return vmulq_f64(v1, v2);
  // CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly8x8_t test_vmul_p8(poly8x8_t v1, poly8x8_t v2) {
  //  test_vmul_p8
  return vmul_p8(v1, v2);
  //  pmul {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x16_t test_vmulq_p8(poly8x16_t v1, poly8x16_t v2) {
  // test_vmulq_p8
  return vmulq_p8(v1, v2);
  // pmul {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}


int8x8_t test_vmla_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CHECK-LABEL: test_vmla_s8
  return vmla_s8(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x8_t test_vmla_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CHECK-LABEL: test_vmla_s16
  return vmla_s16(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmla_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CHECK-LABEL: test_vmla_s32
  return vmla_s32(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmla_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK-LABEL: test_vmla_f32
  return vmla_f32(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vmla_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CHECK-LABEL: test_vmla_u8
  return vmla_u8(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmla_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CHECK-LABEL: test_vmla_u16
  return vmla_u16(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmla_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CHECK-LABEL: test_vmla_u32
  return vmla_u32(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vmlaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CHECK-LABEL: test_vmlaq_s8
  return vmlaq_s8(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vmlaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CHECK-LABEL: test_vmlaq_s16
  return vmlaq_s16(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vmlaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK-LABEL: test_vmlaq_s32
  return vmlaq_s32(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
} 

float32x4_t test_vmlaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK-LABEL: test_vmlaq_f32
  return vmlaq_f32(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vmlaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
   // CHECK-LABEL: test_vmlaq_u8
  return vmlaq_u8(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vmlaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CHECK-LABEL: test_vmlaq_u16
  return vmlaq_u16(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vmlaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  // CHECK-LABEL: test_vmlaq_u32
  return vmlaq_u32(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmlaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK-LABEL: test_vmlaq_f64
  return vmlaq_f64(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vmls_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CHECK-LABEL: test_vmls_s8
  return vmls_s8(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x8_t test_vmls_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CHECK-LABEL: test_vmls_s16
  return vmls_s16(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmls_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CHECK-LABEL: test_vmls_s32
  return vmls_s32(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmls_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK-LABEL: test_vmls_f32
  return vmls_f32(v1, v2, v3);
  // CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vmls_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CHECK-LABEL: test_vmls_u8
  return vmls_u8(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmls_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CHECK-LABEL: test_vmls_u16
  return vmls_u16(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmls_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CHECK-LABEL: test_vmls_u32
  return vmls_u32(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}
int8x16_t test_vmlsq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CHECK-LABEL: test_vmlsq_s8
  return vmlsq_s8(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vmlsq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CHECK-LABEL: test_vmlsq_s16
  return vmlsq_s16(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vmlsq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK-LABEL: test_vmlsq_s32
  return vmlsq_s32(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vmlsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK-LABEL: test_vmlsq_f32
  return vmlsq_f32(v1, v2, v3);
  // CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint8x16_t test_vmlsq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  // CHECK-LABEL: test_vmlsq_u8
  return vmlsq_u8(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vmlsq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CHECK-LABEL: test_vmlsq_u16
  return vmlsq_u16(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vmlsq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  // CHECK-LABEL: test_vmlsq_u32
  return vmlsq_u32(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmlsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK-LABEL: test_vmlsq_f64
  return vmlsq_f64(v1, v2, v3);
  // CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}
float32x2_t test_vfma_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK-LABEL: test_vfma_f32
  return vfma_f32(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vfmaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK-LABEL: test_vfmaq_f32
  return vfmaq_f32(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vfmaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK-LABEL: test_vfmaq_f64
  return vfmaq_f64(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}
float32x2_t test_vfms_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK-LABEL: test_vfms_f32
  return vfms_f32(v1, v2, v3);
  // CHECK: fmls v0.2s, {{v1.2s, v2.2s|v2.2s, v1.2s}}
}

float32x4_t test_vfmsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK-LABEL: test_vfmsq_f32
  return vfmsq_f32(v1, v2, v3);
  // CHECK: fmls v0.4s, {{v1.4s, v2.4s|v2.4s, v1.4s}}
}

float64x2_t test_vfmsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK: vfmsq_f64
  return vfmsq_f64(v1, v2, v3);
  // CHECK: fmls v0.2d, {{v1.2d, v2.2d|v2.2d, v1.2d}}
}

float64x2_t test_vdivq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vdivq_f64
  return vdivq_f64(v1, v2);
  // CHECK: fdiv {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x4_t test_vdivq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vdivq_f32
  return vdivq_f32(v1, v2);
  // CHECK: fdiv {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x2_t test_vdiv_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vdiv_f32
  return vdiv_f32(v1, v2);
  // CHECK: fdiv {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x8_t test_vaba_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CHECK-LABEL: test_vaba_s8
  return vaba_s8(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vaba_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CHECK-LABEL: test_vaba_s16
  return vaba_s16(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vaba_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CHECK-LABEL: test_vaba_s32
  return vaba_s32(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vaba_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CHECK-LABEL: test_vaba_u8
  return vaba_u8(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vaba_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CHECK-LABEL: test_vaba_u16
  return vaba_u16(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vaba_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CHECK-LABEL: test_vaba_u32
  return vaba_u32(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vabaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CHECK-LABEL: test_vabaq_s8
  return vabaq_s8(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vabaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CHECK-LABEL: test_vabaq_s16
  return vabaq_s16(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vabaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK-LABEL: test_vabaq_s32
  return vabaq_s32(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vabaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  // CHECK-LABEL: test_vabaq_u8
  return vabaq_u8(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vabaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CHECK-LABEL: test_vabaq_u16
  return vabaq_u16(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vabaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  // CHECK-LABEL: test_vabaq_u32
  return vabaq_u32(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int8x8_t test_vabd_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK-LABEL: test_vabd_s8
  return vabd_s8(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vabd_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK-LABEL: test_vabd_s16
  return vabd_s16(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vabd_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK-LABEL: test_vabd_s32
  return vabd_s32(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vabd_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK-LABEL: test_vabd_u8
  return vabd_u8(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vabd_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK-LABEL: test_vabd_u16
  return vabd_u16(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vabd_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK-LABEL: test_vabd_u32
  return vabd_u32(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vabd_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vabd_f32
  return vabd_f32(v1, v2);
  // CHECK: fabd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vabdq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK-LABEL: test_vabdq_s8
  return vabdq_s8(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vabdq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK-LABEL: test_vabdq_s16
  return vabdq_s16(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vabdq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK-LABEL: test_vabdq_s32
  return vabdq_s32(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vabdq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK-LABEL: test_vabdq_u8
  return vabdq_u8(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vabdq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK-LABEL: test_vabdq_u16
  return vabdq_u16(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vabdq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK-LABEL: test_vabdq_u32
  return vabdq_u32(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vabdq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vabdq_f32
  return vabdq_f32(v1, v2);
  // CHECK: fabd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vabdq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vabdq_f64
  return vabdq_f64(v1, v2);
  // CHECK: fabd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vbsl_s8(uint8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CHECK-LABEL: test_vbsl_s8
  return vbsl_s8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x8_t test_vbsl_s16(uint16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CHECK-LABEL: test_vbsl_s16
  return vbsl_s16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x2_t test_vbsl_s32(uint32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CHECK-LABEL: test_vbsl_s32
  return vbsl_s32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x1_t test_vbsl_s64(uint64x1_t v1, uint64x1_t v2, uint64x1_t v3) {
  // CHECK-LABEL: test_vbsl_s64
  return vbsl_s64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x8_t test_vbsl_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CHECK-LABEL: test_vbsl_u8
  return vbsl_u8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vbsl_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CHECK-LABEL: test_vbsl_u16
  return vbsl_u16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x2_t test_vbsl_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CHECK-LABEL: test_vbsl_u32
  return vbsl_u32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x1_t test_vbsl_u64(uint64x1_t v1, uint64x1_t v2, uint64x1_t v3) {
  // CHECK-LABEL: test_vbsl_u64
  return vbsl_u64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

float32x2_t test_vbsl_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK-LABEL: test_vbsl_f32
  return vbsl_f32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

float64x1_t test_vbsl_f64(uint64x1_t v1, float64x1_t v2, float64x1_t v3) {
  // CHECK-LABEL: test_vbsl_f64
  return vbsl_f64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x8_t test_vbsl_p8(uint8x8_t v1, poly8x8_t v2, poly8x8_t v3) {
  // CHECK-LABEL: test_vbsl_p8
  return vbsl_p8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly16x4_t test_vbsl_p16(uint16x4_t v1, poly16x4_t v2, poly16x4_t v3) {
  // CHECK-LABEL: test_vbsl_p16
  return vbsl_p16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vbslq_s8(uint8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CHECK-LABEL: test_vbslq_s8
  return vbslq_s8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vbslq_s16(uint16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CHECK-LABEL: test_vbslq_s16
  return vbslq_s16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x4_t test_vbslq_s32(uint32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK-LABEL: test_vbslq_s32
  return vbslq_s32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int64x2_t test_vbslq_s64(uint64x2_t v1, int64x2_t v2, int64x2_t v3) {
  // CHECK-LABEL: test_vbslq_s64
  return vbslq_s64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x16_t test_vbslq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  // CHECK-LABEL: test_vbslq_u8
  return vbslq_u8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vbslq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CHECK-LABEL: test_vbslq_u16
  return vbslq_u16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x4_t test_vbslq_u32(uint32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK-LABEL: test_vbslq_u32
  return vbslq_s32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint64x2_t test_vbslq_u64(uint64x2_t v1, uint64x2_t v2, uint64x2_t v3) {
  // CHECK-LABEL: test_vbslq_u64
  return vbslq_u64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

float32x4_t test_vbslq_f32(uint32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK-LABEL: test_vbslq_f32
  return vbslq_f32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly8x16_t test_vbslq_p8(uint8x16_t v1, poly8x16_t v2, poly8x16_t v3) {
  // CHECK-LABEL: test_vbslq_p8
  return vbslq_p8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly16x8_t test_vbslq_p16(uint16x8_t v1, poly16x8_t v2, poly16x8_t v3) {
  // CHECK-LABEL: test_vbslq_p16
  return vbslq_p16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

float64x2_t test_vbslq_f64(uint64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK-LABEL: test_vbslq_f64
  return vbslq_f64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

float32x2_t test_vrecps_f32(float32x2_t v1, float32x2_t v2) {
   // CHECK-LABEL: test_vrecps_f32
   return vrecps_f32(v1, v2);
   // CHECK: frecps {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vrecpsq_f32(float32x4_t v1, float32x4_t v2) {
   // CHECK-LABEL: test_vrecpsq_f32
   return vrecpsq_f32(v1, v2);
   // CHECK: frecps {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vrecpsq_f64(float64x2_t v1, float64x2_t v2) {
   // CHECK-LABEL: test_vrecpsq_f64
  return vrecpsq_f64(v1, v2);
  // CHECK: frecps {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vrsqrts_f32(float32x2_t v1, float32x2_t v2) {
   // CHECK-LABEL: test_vrsqrts_f32
  return vrsqrts_f32(v1, v2);
  // CHECK: frsqrts {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vrsqrtsq_f32(float32x4_t v1, float32x4_t v2) {
   // CHECK-LABEL: test_vrsqrtsq_f32
  return vrsqrtsq_f32(v1, v2);
  // CHECK: frsqrts {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vrsqrtsq_f64(float64x2_t v1, float64x2_t v2) {
   // CHECK-LABEL: test_vrsqrtsq_f64
  return vrsqrtsq_f64(v1, v2);
  // CHECK: frsqrts {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint32x2_t test_vcage_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vcage_f32
  return vcage_f32(v1, v2);
  // CHECK: facge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vcage_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vcage_f64
  return vcage_f64(a, b);
  // CHECK: facge {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x4_t test_vcageq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vcageq_f32
  return vcageq_f32(v1, v2);
  // CHECK: facge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vcageq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vcageq_f64
  return vcageq_f64(v1, v2);
  // CHECK: facge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint32x2_t test_vcagt_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vcagt_f32
  return vcagt_f32(v1, v2);
  // CHECK: facgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vcagt_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vcagt_f64
  return vcagt_f64(a, b);
  // CHECK: facgt {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x4_t test_vcagtq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vcagtq_f32
  return vcagtq_f32(v1, v2);
  // CHECK: facgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vcagtq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vcagtq_f64
  return vcagtq_f64(v1, v2);
  // CHECK: facgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint32x2_t test_vcale_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vcale_f32
  return vcale_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facge {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint64x1_t test_vcale_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vcale_f64
  return vcale_f64(a, b);
  // CHECK: facge {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x4_t test_vcaleq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vcaleq_f32
  return vcaleq_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facge {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint64x2_t test_vcaleq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vcaleq_f64
  return vcaleq_f64(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facge {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint32x2_t test_vcalt_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vcalt_f32
  return vcalt_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facgt {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint64x1_t test_vcalt_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vcalt_f64
  return vcalt_f64(a, b);
  // CHECK: facgt {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x4_t test_vcaltq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vcaltq_f32
  return vcaltq_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facgt {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint64x2_t test_vcaltq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vcaltq_f64
  return vcaltq_f64(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facgt {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint8x8_t test_vtst_s8(int8x8_t v1, int8x8_t v2) {
   // CHECK-LABEL: test_vtst_s8
  return vtst_s8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vtst_s16(int16x4_t v1, int16x4_t v2) {
   // CHECK-LABEL: test_vtst_s16
  return vtst_s16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vtst_s32(int32x2_t v1, int32x2_t v2) {
   // CHECK-LABEL: test_vtst_s32
  return vtst_s32(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vtst_u8(uint8x8_t v1, uint8x8_t v2) {
   // CHECK-LABEL: test_vtst_u8
  return vtst_u8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vtst_u16(uint16x4_t v1, uint16x4_t v2) {
   // CHECK-LABEL: test_vtst_u16
  return vtst_u16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vtst_u32(uint32x2_t v1, uint32x2_t v2) {
   // CHECK-LABEL: test_vtst_u32
  return vtst_u32(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x16_t test_vtstq_s8(int8x16_t v1, int8x16_t v2) {
   // CHECK-LABEL: test_vtstq_s8
  return vtstq_s8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vtstq_s16(int16x8_t v1, int16x8_t v2) {
   // CHECK-LABEL: test_vtstq_s16
  return vtstq_s16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vtstq_s32(int32x4_t v1, int32x4_t v2) {
   // CHECK-LABEL: test_vtstq_s32
  return vtstq_s32(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vtstq_u8(uint8x16_t v1, uint8x16_t v2) {
   // CHECK-LABEL: test_vtstq_u8
  return vtstq_u8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vtstq_u16(uint16x8_t v1, uint16x8_t v2) {
   // CHECK-LABEL: test_vtstq_u16
  return vtstq_u16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vtstq_u32(uint32x4_t v1, uint32x4_t v2) {
   // CHECK-LABEL: test_vtstq_u32
  return vtstq_u32(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vtstq_s64(int64x2_t v1, int64x2_t v2) {
   // CHECK-LABEL: test_vtstq_s64
  return vtstq_s64(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vtstq_u64(uint64x2_t v1, uint64x2_t v2) {
   // CHECK-LABEL: test_vtstq_u64
  return vtstq_u64(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x8_t test_vtst_p8(poly8x8_t v1, poly8x8_t v2) {
   // CHECK-LABEL: test_vtst_p8
  return vtst_p8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vtst_p16(poly16x4_t v1, poly16x4_t v2) {
   // CHECK-LABEL: test_vtst_p16
  return vtst_p16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint8x16_t test_vtstq_p8(poly8x16_t v1, poly8x16_t v2) {
   // CHECK-LABEL: test_vtstq_p8
  return vtstq_p8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vtstq_p16(poly16x8_t v1, poly16x8_t v2) {
   // CHECK-LABEL: test_vtstq_p16
  return vtstq_p16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint64x1_t test_vtst_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vtst_s64
  return vtst_s64(a, b);
  // CHECK: cmtst {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x1_t test_vtst_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vtst_u64
  return vtst_u64(a, b);
  // CHECK: cmtst {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vceq_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK-LABEL: test_vceq_s8
  return vceq_s8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vceq_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK-LABEL: test_vceq_s16
  return vceq_s16(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vceq_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK-LABEL: test_vceq_s32
  return vceq_s32(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vceq_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vceq_s64
  return vceq_s64(a, b);
  // CHECK: cmeq {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x1_t test_vceq_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vceq_u64
  return vceq_u64(a, b);
  // CHECK: cmeq {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x2_t test_vceq_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vceq_f32
  return vceq_f32(v1, v2);
  // CHECK: fcmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vceq_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vceq_f64
  return vceq_f64(a, b);
  // CHECK: fcmeq {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vceq_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK-LABEL: test_vceq_u8
  return vceq_u8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vceq_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK-LABEL: test_vceq_u16
  return vceq_u16(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vceq_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK-LABEL: test_vceq_u32
  return vceq_u32(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vceq_p8(poly8x8_t v1, poly8x8_t v2) {
  // CHECK-LABEL: test_vceq_p8
  return vceq_p8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vceqq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK-LABEL: test_vceqq_s8
  return vceqq_s8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vceqq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK-LABEL: test_vceqq_s16
  return vceqq_s16(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vceqq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK-LABEL: test_vceqq_s32
  return vceqq_s32(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vceqq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vceqq_f32
  return vceqq_f32(v1, v2);
  // CHECK: fcmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vceqq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK-LABEL: test_vceqq_u8
  return vceqq_u8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vceqq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK-LABEL: test_vceqq_u16
  return vceqq_u16(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vceqq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK-LABEL: test_vceqq_u32
  return vceqq_u32(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vceqq_p8(poly8x16_t v1, poly8x16_t v2) {
  // CHECK-LABEL: test_vceqq_p8
  return vceqq_p8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}


uint64x2_t test_vceqq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK-LABEL: test_vceqq_s64
  return vceqq_s64(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vceqq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK-LABEL: test_vceqq_u64
  return vceqq_u64(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vceqq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vceqq_f64
  return vceqq_f64(v1, v2);
  // CHECK: fcmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}
uint8x8_t test_vcge_s8(int8x8_t v1, int8x8_t v2) {
// CHECK-LABEL: test_vcge_s8
  return vcge_s8(v1, v2);
// CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vcge_s16(int16x4_t v1, int16x4_t v2) {
// CHECK-LABEL: test_vcge_s16
  return vcge_s16(v1, v2);
// CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vcge_s32(int32x2_t v1, int32x2_t v2) {
// CHECK-LABEL: test_vcge_s32
  return vcge_s32(v1, v2);
// CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vcge_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vcge_s64
  return vcge_s64(a, b);
  // CHECK: cmge {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x1_t test_vcge_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vcge_u64
  return vcge_u64(a, b);
  // CHECK: cmhs {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x2_t test_vcge_f32(float32x2_t v1, float32x2_t v2) {
// CHECK-LABEL: test_vcge_f32
  return vcge_f32(v1, v2);
// CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vcge_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vcge_f64
  return vcge_f64(a, b);
  // CHECK: fcmge {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vcge_u8(uint8x8_t v1, uint8x8_t v2) {
// CHECK-LABEL: test_vcge_u8
  return vcge_u8(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vcge_u16(uint16x4_t v1, uint16x4_t v2) {
// CHECK-LABEL: test_vcge_u16
  return vcge_u16(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vcge_u32(uint32x2_t v1, uint32x2_t v2) {
// CHECK-LABEL: test_vcge_u32
  return vcge_u32(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x16_t test_vcgeq_s8(int8x16_t v1, int8x16_t v2) {
// CHECK-LABEL: test_vcgeq_s8
  return vcgeq_s8(v1, v2);
// CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vcgeq_s16(int16x8_t v1, int16x8_t v2) {
// CHECK-LABEL: test_vcgeq_s16
  return vcgeq_s16(v1, v2);
// CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vcgeq_s32(int32x4_t v1, int32x4_t v2) {
// CHECK-LABEL: test_vcgeq_s32
  return vcgeq_s32(v1, v2);
// CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vcgeq_f32(float32x4_t v1, float32x4_t v2) {
// CHECK-LABEL: test_vcgeq_f32
  return vcgeq_f32(v1, v2);
// CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vcgeq_u8(uint8x16_t v1, uint8x16_t v2) {
// CHECK-LABEL: test_vcgeq_u8
  return vcgeq_u8(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vcgeq_u16(uint16x8_t v1, uint16x8_t v2) {
// CHECK-LABEL: test_vcgeq_u16
  return vcgeq_u16(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vcgeq_u32(uint32x4_t v1, uint32x4_t v2) {
// CHECK-LABEL: test_vcgeq_u32
  return vcgeq_u32(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vcgeq_s64(int64x2_t v1, int64x2_t v2) {
// CHECK-LABEL: test_vcgeq_s64
  return vcgeq_s64(v1, v2);
// CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vcgeq_u64(uint64x2_t v1, uint64x2_t v2) {
// CHECK-LABEL: test_vcgeq_u64
  return vcgeq_u64(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vcgeq_f64(float64x2_t v1, float64x2_t v2) {
// CHECK-LABEL: test_vcgeq_f64
  return vcgeq_f64(v1, v2);
// CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

// Notes about vcle:
// LE condition predicate implemented as GE, so check reversed operands.
// Using registers other than v0, v1 are possible, but would be odd.
uint8x8_t test_vcle_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK-LABEL: test_vcle_s8
  return vcle_s8(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.8b, v1.8b, v0.8b
}

uint16x4_t test_vcle_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK-LABEL: test_vcle_s16
  return vcle_s16(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.4h, v1.4h, v0.4h
}

uint32x2_t test_vcle_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK-LABEL: test_vcle_s32
  return vcle_s32(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint64x1_t test_vcle_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vcle_s64
  return vcle_s64(a, b);
  // CHECK: cmge {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x1_t test_vcle_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vcle_u64
  return vcle_u64(a, b);
  // CHECK: cmhs {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x2_t test_vcle_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vcle_f32
  return vcle_f32(v1, v2);
  // CHECK: fcmge {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint64x1_t test_vcle_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vcle_f64
  return vcle_f64(a, b);
  // CHECK: fcmge {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vcle_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK-LABEL: test_vcle_u8
  return vcle_u8(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.8b, v1.8b, v0.8b
}

uint16x4_t test_vcle_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK-LABEL: test_vcle_u16
  return vcle_u16(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.4h, v1.4h, v0.4h
}

uint32x2_t test_vcle_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK-LABEL: test_vcle_u32
  return vcle_u32(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint8x16_t test_vcleq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK-LABEL: test_vcleq_s8
  return vcleq_s8(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.16b, v1.16b, v0.16b
}

uint16x8_t test_vcleq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK-LABEL: test_vcleq_s16
  return vcleq_s16(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.8h, v1.8h, v0.8h
}

uint32x4_t test_vcleq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK-LABEL: test_vcleq_s32
  return vcleq_s32(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint32x4_t test_vcleq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vcleq_f32
  return vcleq_f32(v1, v2);
  // CHECK: fcmge {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint8x16_t test_vcleq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK-LABEL: test_vcleq_u8
  return vcleq_u8(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.16b, v1.16b, v0.16b
}

uint16x8_t test_vcleq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK-LABEL: test_vcleq_u16
  return vcleq_u16(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.8h, v1.8h, v0.8h
}

uint32x4_t test_vcleq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK-LABEL: test_vcleq_u32
  return vcleq_u32(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint64x2_t test_vcleq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK-LABEL: test_vcleq_s64
  return vcleq_s64(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint64x2_t test_vcleq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK-LABEL: test_vcleq_u64
  return vcleq_u64(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint64x2_t test_vcleq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vcleq_f64
  return vcleq_f64(v1, v2);
  // CHECK: fcmge {{v[0-9]+}}.2d, v1.2d, v0.2d
}


uint8x8_t test_vcgt_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK-LABEL: test_vcgt_s8
  return vcgt_s8(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vcgt_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK-LABEL: test_vcgt_s16
  return vcgt_s16(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vcgt_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK-LABEL: test_vcgt_s32
  return vcgt_s32(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vcgt_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vcgt_s64
  return vcgt_s64(a, b);
  // CHECK: cmgt {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x1_t test_vcgt_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vcgt_u64
  return vcgt_u64(a, b);
  // CHECK: cmhi {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x2_t test_vcgt_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vcgt_f32
  return vcgt_f32(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vcgt_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vcgt_f64
  return vcgt_f64(a, b);
  // CHECK: fcmgt {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vcgt_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK-LABEL: test_vcgt_u8
  return vcgt_u8(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vcgt_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK-LABEL: test_vcgt_u16
  return vcgt_u16(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vcgt_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK-LABEL: test_vcgt_u32
  return vcgt_u32(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x16_t test_vcgtq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK-LABEL: test_vcgtq_s8
  return vcgtq_s8(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vcgtq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK-LABEL: test_vcgtq_s16
  return vcgtq_s16(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vcgtq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK-LABEL: test_vcgtq_s32
  return vcgtq_s32(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vcgtq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vcgtq_f32
  return vcgtq_f32(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vcgtq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK-LABEL: test_vcgtq_u8
  return vcgtq_u8(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vcgtq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK-LABEL: test_vcgtq_u16
  return vcgtq_u16(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vcgtq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK-LABEL: test_vcgtq_u32
  return vcgtq_u32(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vcgtq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK-LABEL: test_vcgtq_s64
  return vcgtq_s64(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vcgtq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK-LABEL: test_vcgtq_u64
  return vcgtq_u64(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vcgtq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vcgtq_f64
  return vcgtq_f64(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


// Notes about vclt:
// LT condition predicate implemented as GT, so check reversed operands.
// Using registers other than v0, v1 are possible, but would be odd.

uint8x8_t test_vclt_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK-LABEL: test_vclt_s8
  return vclt_s8(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.8b, v1.8b, v0.8b
}

uint16x4_t test_vclt_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK-LABEL: test_vclt_s16
  return vclt_s16(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.4h, v1.4h, v0.4h
}

uint32x2_t test_vclt_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK-LABEL: test_vclt_s32
  return vclt_s32(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint64x1_t test_vclt_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vclt_s64
  return vclt_s64(a, b);
  // CHECK: cmgt {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x1_t test_vclt_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vclt_u64
  return vclt_u64(a, b);
  // CHECK: cmhi {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint32x2_t test_vclt_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK-LABEL: test_vclt_f32
  return vclt_f32(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint64x1_t test_vclt_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vclt_f64
  return vclt_f64(a, b);
  // CHECK: fcmgt {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vclt_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK-LABEL: test_vclt_u8
  return vclt_u8(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.8b, v1.8b, v0.8b
}

uint16x4_t test_vclt_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK-LABEL: test_vclt_u16
  return vclt_u16(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.4h, v1.4h, v0.4h
}

uint32x2_t test_vclt_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK-LABEL: test_vclt_u32
  return vclt_u32(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint8x16_t test_vcltq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK-LABEL: test_vcltq_s8
  return vcltq_s8(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.16b, v1.16b, v0.16b
}

uint16x8_t test_vcltq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK-LABEL: test_vcltq_s16
  return vcltq_s16(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.8h, v1.8h, v0.8h
}

uint32x4_t test_vcltq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK-LABEL: test_vcltq_s32
  return vcltq_s32(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint32x4_t test_vcltq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK-LABEL: test_vcltq_f32
  return vcltq_f32(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint8x16_t test_vcltq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK-LABEL: test_vcltq_u8
  return vcltq_u8(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.16b, v1.16b, v0.16b
}

uint16x8_t test_vcltq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK-LABEL: test_vcltq_u16
  return vcltq_u16(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.8h, v1.8h, v0.8h
}

uint32x4_t test_vcltq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK-LABEL: test_vcltq_u32
  return vcltq_u32(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint64x2_t test_vcltq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK-LABEL: test_vcltq_s64
  return vcltq_s64(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint64x2_t test_vcltq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK-LABEL: test_vcltq_u64
  return vcltq_u64(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint64x2_t test_vcltq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK-LABEL: test_vcltq_f64
  return vcltq_f64(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
}


int8x8_t test_vhadd_s8(int8x8_t v1, int8x8_t v2) {
// CHECK-LABEL: test_vhadd_s8
  return vhadd_s8(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vhadd_s16(int16x4_t v1, int16x4_t v2) {
// CHECK-LABEL: test_vhadd_s16
  return vhadd_s16(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vhadd_s32(int32x2_t v1, int32x2_t v2) {
// CHECK-LABEL: test_vhadd_s32
  return vhadd_s32(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vhadd_u8(uint8x8_t v1, uint8x8_t v2) {
// CHECK-LABEL: test_vhadd_u8
  return vhadd_u8(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vhadd_u16(uint16x4_t v1, uint16x4_t v2) {
// CHECK-LABEL: test_vhadd_u16
  return vhadd_u16(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vhadd_u32(uint32x2_t v1, uint32x2_t v2) {
// CHECK-LABEL: test_vhadd_u32
  return vhadd_u32(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vhaddq_s8(int8x16_t v1, int8x16_t v2) {
// CHECK-LABEL: test_vhaddq_s8
  return vhaddq_s8(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vhaddq_s16(int16x8_t v1, int16x8_t v2) {
// CHECK-LABEL: test_vhaddq_s16
  return vhaddq_s16(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vhaddq_s32(int32x4_t v1, int32x4_t v2) {
// CHECK-LABEL: test_vhaddq_s32
  return vhaddq_s32(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
// CHECK-LABEL: test_vhaddq_u8
  return vhaddq_u8(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
// CHECK-LABEL: test_vhaddq_u16
  return vhaddq_u16(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
// CHECK-LABEL: test_vhaddq_u32
  return vhaddq_u32(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}


int8x8_t test_vhsub_s8(int8x8_t v1, int8x8_t v2) {
// CHECK-LABEL: test_vhsub_s8
  return vhsub_s8(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vhsub_s16(int16x4_t v1, int16x4_t v2) {
// CHECK-LABEL: test_vhsub_s16
  return vhsub_s16(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vhsub_s32(int32x2_t v1, int32x2_t v2) {
// CHECK-LABEL: test_vhsub_s32
  return vhsub_s32(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vhsub_u8(uint8x8_t v1, uint8x8_t v2) {
// CHECK-LABEL: test_vhsub_u8
  return vhsub_u8(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vhsub_u16(uint16x4_t v1, uint16x4_t v2) {
// CHECK-LABEL: test_vhsub_u16
  return vhsub_u16(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vhsub_u32(uint32x2_t v1, uint32x2_t v2) {
// CHECK-LABEL: test_vhsub_u32
  return vhsub_u32(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vhsubq_s8(int8x16_t v1, int8x16_t v2) {
// CHECK-LABEL: test_vhsubq_s8
  return vhsubq_s8(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vhsubq_s16(int16x8_t v1, int16x8_t v2) {
// CHECK-LABEL: test_vhsubq_s16
  return vhsubq_s16(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vhsubq_s32(int32x4_t v1, int32x4_t v2) {
// CHECK-LABEL: test_vhsubq_s32
  return vhsubq_s32(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vhsubq_u8(uint8x16_t v1, uint8x16_t v2) {
// CHECK-LABEL: test_vhsubq_u8
  return vhsubq_u8(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vhsubq_u16(uint16x8_t v1, uint16x8_t v2) {
// CHECK-LABEL: test_vhsubq_u16
  return vhsubq_u16(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vhsubq_u32(uint32x4_t v1, uint32x4_t v2) {
// CHECK-LABEL: test_vhsubq_u32
  return vhsubq_u32(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}


int8x8_t test_vrhadd_s8(int8x8_t v1, int8x8_t v2) {
// CHECK-LABEL: test_vrhadd_s8
  return vrhadd_s8(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vrhadd_s16(int16x4_t v1, int16x4_t v2) {
// CHECK-LABEL: test_vrhadd_s16
  return vrhadd_s16(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vrhadd_s32(int32x2_t v1, int32x2_t v2) {
// CHECK-LABEL: test_vrhadd_s32
  return vrhadd_s32(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vrhadd_u8(uint8x8_t v1, uint8x8_t v2) {
// CHECK-LABEL: test_vrhadd_u8
  return vrhadd_u8(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vrhadd_u16(uint16x4_t v1, uint16x4_t v2) {
// CHECK-LABEL: test_vrhadd_u16
  return vrhadd_u16(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vrhadd_u32(uint32x2_t v1, uint32x2_t v2) {
// CHECK-LABEL: test_vrhadd_u32
  return vrhadd_u32(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vrhaddq_s8(int8x16_t v1, int8x16_t v2) {
// CHECK-LABEL: test_vrhaddq_s8
  return vrhaddq_s8(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vrhaddq_s16(int16x8_t v1, int16x8_t v2) {
// CHECK-LABEL: test_vrhaddq_s16
  return vrhaddq_s16(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vrhaddq_s32(int32x4_t v1, int32x4_t v2) {
// CHECK-LABEL: test_vrhaddq_s32
  return vrhaddq_s32(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vrhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
// CHECK-LABEL: test_vrhaddq_u8
  return vrhaddq_u8(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vrhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
// CHECK-LABEL: test_vrhaddq_u16
  return vrhaddq_u16(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vrhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
// CHECK-LABEL: test_vrhaddq_u32
  return vrhaddq_u32(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
int8x8_t test_vqadd_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vqadd_s8
  return vqadd_s8(a, b);
  // CHECK: sqadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vqadd_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vqadd_s16
  return vqadd_s16(a, b);
  // CHECK: sqadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqadd_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vqadd_s32
  return vqadd_s32(a, b);
  // CHECK: sqadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vqadd_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vqadd_s64
  return vqadd_s64(a, b);
// CHECK:  sqadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vqadd_u8(uint8x8_t a, uint8x8_t b) {
// CHECK-LABEL: test_vqadd_u8
  return vqadd_u8(a, b);
  // CHECK: uqadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vqadd_u16(uint16x4_t a, uint16x4_t b) {
// CHECK-LABEL: test_vqadd_u16
  return vqadd_u16(a, b);
  // CHECK: uqadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vqadd_u32(uint32x2_t a, uint32x2_t b) {
// CHECK-LABEL: test_vqadd_u32
  return vqadd_u32(a, b);
  // CHECK: uqadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vqadd_u64(uint64x1_t a, uint64x1_t b) {
// CHECK:  test_vqadd_u64
  return vqadd_u64(a, b);
// CHECK:  uqadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vqaddq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vqaddq_s8
  return vqaddq_s8(a, b);
  // CHECK: sqadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vqaddq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vqaddq_s16
  return vqaddq_s16(a, b);
  // CHECK: sqadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqaddq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vqaddq_s32
  return vqaddq_s32(a, b);
  // CHECK: sqadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vqaddq_s64(int64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vqaddq_s64
  return vqaddq_s64(a, b);
// CHECK: sqadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vqaddq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK-LABEL: test_vqaddq_u8
  return vqaddq_u8(a, b);
  // CHECK: uqadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vqaddq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK-LABEL: test_vqaddq_u16
  return vqaddq_u16(a, b);
  // CHECK: uqadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vqaddq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK-LABEL: test_vqaddq_u32
  return vqaddq_u32(a, b);
  // CHECK: uqadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vqaddq_u64(uint64x2_t a, uint64x2_t b) {
// CHECK-LABEL: test_vqaddq_u64
  return vqaddq_u64(a, b);
// CHECK: uqadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vqsub_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vqsub_s8
  return vqsub_s8(a, b);
  // CHECK: sqsub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vqsub_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vqsub_s16
  return vqsub_s16(a, b);
  // CHECK: sqsub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqsub_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vqsub_s32
  return vqsub_s32(a, b);
  // CHECK: sqsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vqsub_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vqsub_s64
  return vqsub_s64(a, b);
// CHECK: sqsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vqsub_u8(uint8x8_t a, uint8x8_t b) {
// CHECK-LABEL: test_vqsub_u8
  return vqsub_u8(a, b);
  // CHECK: uqsub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vqsub_u16(uint16x4_t a, uint16x4_t b) {
// CHECK-LABEL: test_vqsub_u16
  return vqsub_u16(a, b);
  // CHECK: uqsub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vqsub_u32(uint32x2_t a, uint32x2_t b) {
// CHECK-LABEL: test_vqsub_u32
  return vqsub_u32(a, b);
  // CHECK: uqsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vqsub_u64(uint64x1_t a, uint64x1_t b) {
// CHECK-LABEL: test_vqsub_u64
  return vqsub_u64(a, b);
// CHECK:  uqsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vqsubq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vqsubq_s8
  return vqsubq_s8(a, b);
  // CHECK: sqsub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vqsubq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vqsubq_s16
  return vqsubq_s16(a, b);
  // CHECK: sqsub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqsubq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vqsubq_s32
  return vqsubq_s32(a, b);
  // CHECK: sqsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vqsubq_s64(int64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vqsubq_s64
  return vqsubq_s64(a, b);
// CHECK: sqsub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vqsubq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK-LABEL: test_vqsubq_u8
  return vqsubq_u8(a, b);
  // CHECK: uqsub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vqsubq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK-LABEL: test_vqsubq_u16
  return vqsubq_u16(a, b);
  // CHECK: uqsub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vqsubq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK-LABEL: test_vqsubq_u32
  return vqsubq_u32(a, b);
  // CHECK: uqsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vqsubq_u64(uint64x2_t a, uint64x2_t b) {
// CHECK-LABEL: test_vqsubq_u64
  return vqsubq_u64(a, b);
  // CHECK: uqsub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vshl_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vshl_s8
  return vshl_s8(a, b);
// CHECK: sshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vshl_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vshl_s16
  return vshl_s16(a, b);
// CHECK: sshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vshl_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vshl_s32
  return vshl_s32(a, b);
// CHECK: sshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vshl_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vshl_s64
  return vshl_s64(a, b);
// CHECK: sshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vshl_u8(uint8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vshl_u8
  return vshl_u8(a, b);
// CHECK: ushl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vshl_u16(uint16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vshl_u16
  return vshl_u16(a, b);
// CHECK: ushl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vshl_u32(uint32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vshl_u32
  return vshl_u32(a, b);
// CHECK: ushl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vshl_u64
  return vshl_u64(a, b);
// CHECK: ushl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vshlq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vshlq_s8
  return vshlq_s8(a, b);
// CHECK: sshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vshlq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vshlq_s16
  return vshlq_s16(a, b);
// CHECK: sshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vshlq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vshlq_s32
  return vshlq_s32(a, b);
// CHECK: sshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vshlq_s64(int64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vshlq_s64
  return vshlq_s64(a, b);
// CHECK: sshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vshlq_u8(uint8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vshlq_u8
  return vshlq_u8(a, b);
// CHECK: ushl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vshlq_u16(uint16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vshlq_u16
  return vshlq_u16(a, b);
// CHECK: ushl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vshlq_u32(uint32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vshlq_u32
  return vshlq_u32(a, b);
// CHECK: ushl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vshlq_u64(uint64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vshlq_u64
  return vshlq_u64(a, b);
// CHECK: ushl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vqshl_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vqshl_s8
  return vqshl_s8(a, b);
// CHECK: sqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vqshl_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vqshl_s16
  return vqshl_s16(a, b);
// CHECK: sqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqshl_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vqshl_s32
  return vqshl_s32(a, b);
// CHECK: sqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vqshl_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vqshl_s64
  return vqshl_s64(a, b);
// CHECK: sqshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vqshl_u8(uint8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vqshl_u8
  return vqshl_u8(a, b);
// CHECK: uqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vqshl_u16(uint16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vqshl_u16
  return vqshl_u16(a, b);
// CHECK: uqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vqshl_u32(uint32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vqshl_u32
  return vqshl_u32(a, b);
// CHECK: uqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vqshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vqshl_u64
  return vqshl_u64(a, b);
// CHECK: uqshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vqshlq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vqshlq_s8
  return vqshlq_s8(a, b);
// CHECK: sqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vqshlq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vqshlq_s16
  return vqshlq_s16(a, b);
// CHECK: sqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqshlq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vqshlq_s32
  return vqshlq_s32(a, b);
// CHECK: sqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vqshlq_s64(int64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vqshlq_s64
  return vqshlq_s64(a, b);
// CHECK: sqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vqshlq_u8(uint8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vqshlq_u8
  return vqshlq_u8(a, b);
// CHECK: uqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vqshlq_u16(uint16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vqshlq_u16
  return vqshlq_u16(a, b);
// CHECK: uqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vqshlq_u32(uint32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vqshlq_u32
  return vqshlq_u32(a, b);
// CHECK: uqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vqshlq_u64(uint64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vqshlq_u64
  return vqshlq_u64(a, b);
// CHECK: uqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vrshl_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vrshl_s8
  return vrshl_s8(a, b);
// CHECK: srshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vrshl_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vrshl_s16
  return vrshl_s16(a, b);
// CHECK: srshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vrshl_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vrshl_s32
  return vrshl_s32(a, b);
// CHECK: srshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vrshl_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vrshl_s64
  return vrshl_s64(a, b);
// CHECK: srshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vrshl_u8(uint8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vrshl_u8
  return vrshl_u8(a, b);
// CHECK: urshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vrshl_u16(uint16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vrshl_u16
  return vrshl_u16(a, b);
// CHECK: urshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vrshl_u32(uint32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vrshl_u32
  return vrshl_u32(a, b);
// CHECK: urshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vrshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vrshl_u64
  return vrshl_u64(a, b);
// CHECK: urshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vrshlq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vrshlq_s8
  return vrshlq_s8(a, b);
// CHECK: srshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vrshlq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vrshlq_s16
  return vrshlq_s16(a, b);
// CHECK: srshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vrshlq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vrshlq_s32
  return vrshlq_s32(a, b);
// CHECK: srshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vrshlq_s64(int64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vrshlq_s64
  return vrshlq_s64(a, b);
// CHECK: srshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vrshlq_u8(uint8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vrshlq_u8
  return vrshlq_u8(a, b);
// CHECK: urshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vrshlq_u16(uint16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vrshlq_u16
  return vrshlq_u16(a, b);
// CHECK: urshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vrshlq_u32(uint32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vrshlq_u32
  return vrshlq_u32(a, b);
// CHECK: urshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vrshlq_u64(uint64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vrshlq_u64
  return vrshlq_u64(a, b);
// CHECK: urshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vqrshl_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vqrshl_s8
  return vqrshl_s8(a, b);
// CHECK: sqrshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vqrshl_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vqrshl_s16
  return vqrshl_s16(a, b);
// CHECK: sqrshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqrshl_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vqrshl_s32
  return vqrshl_s32(a, b);
// CHECK: sqrshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vqrshl_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vqrshl_s64
  return vqrshl_s64(a, b);
// CHECK: sqrshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vqrshl_u8(uint8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vqrshl_u8
  return vqrshl_u8(a, b);
// CHECK: uqrshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vqrshl_u16(uint16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vqrshl_u16
  return vqrshl_u16(a, b);
// CHECK: uqrshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vqrshl_u32(uint32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vqrshl_u32
  return vqrshl_u32(a, b);
// CHECK: uqrshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vqrshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vqrshl_u64
  return vqrshl_u64(a, b);
// CHECK: uqrshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vqrshlq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vqrshlq_s8
  return vqrshlq_s8(a, b);
// CHECK: sqrshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vqrshlq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vqrshlq_s16
  return vqrshlq_s16(a, b);
// CHECK: sqrshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqrshlq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vqrshlq_s32
  return vqrshlq_s32(a, b);
// CHECK: sqrshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vqrshlq_s64(int64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vqrshlq_s64
  return vqrshlq_s64(a, b);
// CHECK: sqrshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

// CHECK-LABEL: test_vqrshlq_u8
uint8x16_t test_vqrshlq_u8(uint8x16_t a, int8x16_t b) {
  return vqrshlq_u8(a, b);
// CHECK: uqrshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vqrshlq_u16(uint16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vqrshlq_u16
  return vqrshlq_u16(a, b);
// CHECK: uqrshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vqrshlq_u32(uint32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vqrshlq_u32
  return vqrshlq_u32(a, b);
// CHECK: uqrshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vqrshlq_u64(uint64x2_t a, int64x2_t b) {
// CHECK-LABEL: test_vqrshlq_u64
  return vqrshlq_u64(a, b);
// CHECK: uqrshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly64x1_t test_vsli_n_p64(poly64x1_t a, poly64x1_t b) {
// CHECK-LABEL: test_vsli_n_p64
  return vsli_n_p64(a, b, 0); 
// CHECK: sli {{d[0-9]+}}, {{d[0-9]+}}, #0
}

poly64x2_t test_vsliq_n_p64(poly64x2_t a, poly64x2_t b) {
// CHECK-LABEL: test_vsliq_n_p64
  return vsliq_n_p64(a, b, 0); 
// CHECK: sli {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
}

int8x8_t test_vmax_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vmax_s8
  return vmax_s8(a, b);
// CHECK: smax {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vmax_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vmax_s16
  return vmax_s16(a, b);
// CHECK: smax {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmax_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vmax_s32
  return vmax_s32(a, b);
// CHECK: smax {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vmax_u8(uint8x8_t a, uint8x8_t b) {
// CHECK-LABEL: test_vmax_u8
  return vmax_u8(a, b);
// CHECK: umax {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmax_u16(uint16x4_t a, uint16x4_t b) {
// CHECK-LABEL: test_vmax_u16
  return vmax_u16(a, b);
// CHECK: umax {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmax_u32(uint32x2_t a, uint32x2_t b) {
// CHECK-LABEL: test_vmax_u32
  return vmax_u32(a, b);
// CHECK: umax {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmax_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vmax_f32
  return vmax_f32(a, b);
// CHECK: fmax {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vmaxq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vmaxq_s8
  return vmaxq_s8(a, b);
// CHECK: smax {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vmaxq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vmaxq_s16
  return vmaxq_s16(a, b);
// CHECK: smax {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vmaxq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vmaxq_s32
  return vmaxq_s32(a, b);
// CHECK: smax {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vmaxq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK-LABEL: test_vmaxq_u8
  return vmaxq_u8(a, b);
// CHECK: umax {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vmaxq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK-LABEL: test_vmaxq_u16
  return vmaxq_u16(a, b);
// CHECK: umax {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vmaxq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK-LABEL: test_vmaxq_u32
  return vmaxq_u32(a, b);
// CHECK: umax {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vmaxq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vmaxq_f32
  return vmaxq_f32(a, b);
// CHECK: fmax {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmaxq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vmaxq_f64
  return vmaxq_f64(a, b);
// CHECK: fmax {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vmin_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vmin_s8
  return vmin_s8(a, b);
// CHECK: smin {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vmin_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vmin_s16
  return vmin_s16(a, b);
// CHECK: smin {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmin_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vmin_s32
  return vmin_s32(a, b);
// CHECK: smin {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vmin_u8(uint8x8_t a, uint8x8_t b) {
// CHECK-LABEL: test_vmin_u8
  return vmin_u8(a, b);
// CHECK: umin {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmin_u16(uint16x4_t a, uint16x4_t b) {
// CHECK-LABEL: test_vmin_u16
  return vmin_u16(a, b);
// CHECK: umin {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmin_u32(uint32x2_t a, uint32x2_t b) {
// CHECK-LABEL: test_vmin_u32
  return vmin_u32(a, b);
// CHECK: umin {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmin_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vmin_f32
  return vmin_f32(a, b);
// CHECK: fmin {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vminq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vminq_s8
  return vminq_s8(a, b);
// CHECK: smin {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vminq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vminq_s16
  return vminq_s16(a, b);
// CHECK: smin {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vminq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vminq_s32
  return vminq_s32(a, b);
// CHECK: smin {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vminq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK-LABEL: test_vminq_u8
  return vminq_u8(a, b);
// CHECK: umin {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vminq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK-LABEL: test_vminq_u16
  return vminq_u16(a, b);
// CHECK: umin {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vminq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK-LABEL: test_vminq_u32
  return vminq_u32(a, b);
// CHECK: umin {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vminq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vminq_f32
  return vminq_f32(a, b);
// CHECK: fmin {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vminq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vminq_f64
  return vminq_f64(a, b);
// CHECK: fmin {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vmaxnm_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vmaxnm_f32
  return vmaxnm_f32(a, b);
// CHECK: fmaxnm {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vmaxnmq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vmaxnmq_f32
  return vmaxnmq_f32(a, b);
// CHECK: fmaxnm {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmaxnmq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vmaxnmq_f64
  return vmaxnmq_f64(a, b);
// CHECK: fmaxnm {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vminnm_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vminnm_f32
  return vminnm_f32(a, b);
// CHECK: fminnm {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vminnmq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vminnmq_f32
  return vminnmq_f32(a, b);
// CHECK: fminnm {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vminnmq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vminnmq_f64
  return vminnmq_f64(a, b);
// CHECK: fminnm {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vpmax_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vpmax_s8
  return vpmax_s8(a, b);
// CHECK: smaxp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vpmax_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vpmax_s16
  return vpmax_s16(a, b);
// CHECK: smaxp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vpmax_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vpmax_s32
  return vpmax_s32(a, b);
// CHECK: smaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vpmax_u8(uint8x8_t a, uint8x8_t b) {
// CHECK-LABEL: test_vpmax_u8
  return vpmax_u8(a, b);
// CHECK: umaxp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vpmax_u16(uint16x4_t a, uint16x4_t b) {
// CHECK-LABEL: test_vpmax_u16
  return vpmax_u16(a, b);
// CHECK: umaxp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vpmax_u32(uint32x2_t a, uint32x2_t b) {
// CHECK-LABEL: test_vpmax_u32
  return vpmax_u32(a, b);
// CHECK: umaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vpmax_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vpmax_f32
  return vpmax_f32(a, b);
// CHECK: fmaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vpmaxq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vpmaxq_s8
  return vpmaxq_s8(a, b);
// CHECK: smaxp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vpmaxq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vpmaxq_s16
  return vpmaxq_s16(a, b);
// CHECK: smaxp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vpmaxq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vpmaxq_s32
  return vpmaxq_s32(a, b);
// CHECK: smaxp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vpmaxq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK-LABEL: test_vpmaxq_u8
  return vpmaxq_u8(a, b);
// CHECK: umaxp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vpmaxq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK-LABEL: test_vpmaxq_u16
  return vpmaxq_u16(a, b);
// CHECK: umaxp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vpmaxq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK-LABEL: test_vpmaxq_u32
  return vpmaxq_u32(a, b);
// CHECK: umaxp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vpmaxq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vpmaxq_f32
  return vpmaxq_f32(a, b);
// CHECK: fmaxp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpmaxq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vpmaxq_f64
  return vpmaxq_f64(a, b);
// CHECK: fmaxp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vpmin_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vpmin_s8
  return vpmin_s8(a, b);
// CHECK: sminp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vpmin_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vpmin_s16
  return vpmin_s16(a, b);
// CHECK: sminp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vpmin_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vpmin_s32
  return vpmin_s32(a, b);
// CHECK: sminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vpmin_u8(uint8x8_t a, uint8x8_t b) {
// CHECK-LABEL: test_vpmin_u8
  return vpmin_u8(a, b);
// CHECK: uminp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vpmin_u16(uint16x4_t a, uint16x4_t b) {
// CHECK-LABEL: test_vpmin_u16
  return vpmin_u16(a, b);
// CHECK: uminp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vpmin_u32(uint32x2_t a, uint32x2_t b) {
// CHECK-LABEL: test_vpmin_u32
  return vpmin_u32(a, b);
// CHECK: uminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vpmin_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vpmin_f32
  return vpmin_f32(a, b);
// CHECK: fminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vpminq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vpminq_s8
  return vpminq_s8(a, b);
// CHECK: sminp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vpminq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vpminq_s16
  return vpminq_s16(a, b);
// CHECK: sminp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vpminq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vpminq_s32
  return vpminq_s32(a, b);
// CHECK: sminp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vpminq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK-LABEL: test_vpminq_u8
  return vpminq_u8(a, b);
// CHECK: uminp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vpminq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK-LABEL: test_vpminq_u16
  return vpminq_u16(a, b);
// CHECK: uminp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vpminq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK-LABEL: test_vpminq_u32
  return vpminq_u32(a, b);
// CHECK: uminp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vpminq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vpminq_f32
  return vpminq_f32(a, b);
// CHECK: fminp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpminq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vpminq_f64
  return vpminq_f64(a, b);
// CHECK: fminp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vpmaxnm_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vpmaxnm_f32
  return vpmaxnm_f32(a, b);
// CHECK: fmaxnmp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vpmaxnmq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vpmaxnmq_f32
  return vpmaxnmq_f32(a, b);
// CHECK: fmaxnmp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpmaxnmq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vpmaxnmq_f64
  return vpmaxnmq_f64(a, b);
// CHECK: fmaxnmp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vpminnm_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vpminnm_f32
  return vpminnm_f32(a, b);
// CHECK: fminnmp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vpminnmq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vpminnmq_f32
  return vpminnmq_f32(a, b);
// CHECK: fminnmp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpminnmq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vpminnmq_f64
  return vpminnmq_f64(a, b);
// CHECK: fminnmp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vpadd_s8(int8x8_t a, int8x8_t b) {
// CHECK-LABEL: test_vpadd_s8
  return vpadd_s8(a, b);
// CHECK: addp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vpadd_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vpadd_s16
  return vpadd_s16(a, b);
// CHECK: addp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vpadd_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vpadd_s32
  return vpadd_s32(a, b);
// CHECK: addp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vpadd_u8(uint8x8_t a, uint8x8_t b) {
// CHECK-LABEL: test_vpadd_u8
  return vpadd_u8(a, b);
// CHECK: addp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vpadd_u16(uint16x4_t a, uint16x4_t b) {
// CHECK-LABEL: test_vpadd_u16
  return vpadd_u16(a, b);
// CHECK: addp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vpadd_u32(uint32x2_t a, uint32x2_t b) {
// CHECK-LABEL: test_vpadd_u32
  return vpadd_u32(a, b);
// CHECK: addp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vpadd_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vpadd_f32
  return vpadd_f32(a, b);
// CHECK: faddp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vpaddq_s8(int8x16_t a, int8x16_t b) {
// CHECK-LABEL: test_vpaddq_s8
  return vpaddq_s8(a, b);
// CHECK: addp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vpaddq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vpaddq_s16
  return vpaddq_s16(a, b);
// CHECK: addp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vpaddq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vpaddq_s32
  return vpaddq_s32(a, b);
// CHECK: addp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vpaddq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK-LABEL: test_vpaddq_u8
  return vpaddq_u8(a, b);
// CHECK: addp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vpaddq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK-LABEL: test_vpaddq_u16
  return vpaddq_u16(a, b);
// CHECK: addp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vpaddq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK-LABEL: test_vpaddq_u32
  return vpaddq_u32(a, b);
// CHECK: addp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vpaddq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vpaddq_f32
  return vpaddq_f32(a, b);
// CHECK: faddp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpaddq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vpaddq_f64
  return vpaddq_f64(a, b);
// CHECK: faddp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int16x4_t test_vqdmulh_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vqdmulh_s16
  return vqdmulh_s16(a, b);
// CHECK: sqdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqdmulh_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vqdmulh_s32
  return vqdmulh_s32(a, b);
// CHECK: sqdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vqdmulhq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vqdmulhq_s16
  return vqdmulhq_s16(a, b);
// CHECK: sqdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqdmulhq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vqdmulhq_s32
  return vqdmulhq_s32(a, b);
// CHECK: sqdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int16x4_t test_vqrdmulh_s16(int16x4_t a, int16x4_t b) {
// CHECK-LABEL: test_vqrdmulh_s16
  return vqrdmulh_s16(a, b);
// CHECK: sqrdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqrdmulh_s32(int32x2_t a, int32x2_t b) {
// CHECK-LABEL: test_vqrdmulh_s32
  return vqrdmulh_s32(a, b);
// CHECK: sqrdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vqrdmulhq_s16(int16x8_t a, int16x8_t b) {
// CHECK-LABEL: test_vqrdmulhq_s16
  return vqrdmulhq_s16(a, b);
// CHECK: sqrdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqrdmulhq_s32(int32x4_t a, int32x4_t b) {
// CHECK-LABEL: test_vqrdmulhq_s32
  return vqrdmulhq_s32(a, b);
// CHECK: sqrdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x2_t test_vmulx_f32(float32x2_t a, float32x2_t b) {
// CHECK-LABEL: test_vmulx_f32
  return vmulx_f32(a, b);
// CHECK: fmulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vmulxq_f32(float32x4_t a, float32x4_t b) {
// CHECK-LABEL: test_vmulxq_f32
  return vmulxq_f32(a, b);
// CHECK: fmulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmulxq_f64(float64x2_t a, float64x2_t b) {
// CHECK-LABEL: test_vmulxq_f64
  return vmulxq_f64(a, b);
// CHECK: fmulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vshl_n_s8(int8x8_t a) {
// CHECK-LABEL: test_vshl_n_s8
  return vshl_n_s8(a, 3);
// CHECK: shl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vshl_n_s16(int16x4_t a) {
// CHECK-LABEL: test_vshl_n_s16
  return vshl_n_s16(a, 3);
// CHECK: shl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vshl_n_s32(int32x2_t a) {
// CHECK-LABEL: test_vshl_n_s32
  return vshl_n_s32(a, 3);
// CHECK: shl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vshlq_n_s8(int8x16_t a) {
// CHECK-LABEL: test_vshlq_n_s8
  return vshlq_n_s8(a, 3);
// CHECK: shl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vshlq_n_s16(int16x8_t a) {
// CHECK-LABEL: test_vshlq_n_s16
  return vshlq_n_s16(a, 3);
// CHECK: shl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vshlq_n_s32(int32x4_t a) {
// CHECK-LABEL: test_vshlq_n_s32
  return vshlq_n_s32(a, 3);
// CHECK: shl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vshlq_n_s64(int64x2_t a) {
// CHECK-LABEL: test_vshlq_n_s64
  return vshlq_n_s64(a, 3);
// CHECK: shl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vshl_n_u8(int8x8_t a) {
// CHECK-LABEL: test_vshl_n_u8
  return vshl_n_u8(a, 3);
// CHECK: shl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vshl_n_u16(int16x4_t a) {
// CHECK-LABEL: test_vshl_n_u16
  return vshl_n_u16(a, 3);
// CHECK: shl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vshl_n_u32(int32x2_t a) {
// CHECK-LABEL: test_vshl_n_u32
  return vshl_n_u32(a, 3);
// CHECK: shl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vshlq_n_u8(int8x16_t a) {
// CHECK-LABEL: test_vshlq_n_u8
  return vshlq_n_u8(a, 3);
// CHECK: shl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vshlq_n_u16(int16x8_t a) {
// CHECK-LABEL: test_vshlq_n_u16
  return vshlq_n_u16(a, 3);
// CHECK: shl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vshlq_n_u32(int32x4_t a) {
// CHECK-LABEL: test_vshlq_n_u32
  return vshlq_n_u32(a, 3);
// CHECK: shl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vshlq_n_u64(int64x2_t a) {
// CHECK-LABEL: test_vshlq_n_u64
  return vshlq_n_u64(a, 3);
// CHECK: shl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vshr_n_s8(int8x8_t a) {
  // CHECK-LABEL: test_vshr_n_s8
  return vshr_n_s8(a, 3);
  // CHECK: sshr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vshr_n_s16(int16x4_t a) {
  // CHECK-LABEL: test_vshr_n_s16
  return vshr_n_s16(a, 3);
  // CHECK: sshr {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vshr_n_s32(int32x2_t a) {
  // CHECK-LABEL: test_vshr_n_s32
  return vshr_n_s32(a, 3);
  // CHECK: sshr {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vshrq_n_s8(int8x16_t a) {
  // CHECK-LABEL: test_vshrq_n_s8
  return vshrq_n_s8(a, 3);
  // CHECK: sshr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vshrq_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vshrq_n_s16
  return vshrq_n_s16(a, 3);
  // CHECK: sshr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vshrq_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vshrq_n_s32
  return vshrq_n_s32(a, 3);
  // CHECK: sshr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vshrq_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vshrq_n_s64
  return vshrq_n_s64(a, 3);
  // CHECK: sshr {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vshr_n_u8(int8x8_t a) {
  // CHECK-LABEL: test_vshr_n_u8
  return vshr_n_u8(a, 3);
  // CHECK: ushr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vshr_n_u16(int16x4_t a) {
  // CHECK-LABEL: test_vshr_n_u16
  return vshr_n_u16(a, 3);
  // CHECK: ushr {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vshr_n_u32(int32x2_t a) {
  // CHECK-LABEL: test_vshr_n_u32
  return vshr_n_u32(a, 3);
  // CHECK: ushr {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vshrq_n_u8(int8x16_t a) {
  // CHECK-LABEL: test_vshrq_n_u8
  return vshrq_n_u8(a, 3);
  // CHECK: ushr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vshrq_n_u16(int16x8_t a) {
  // CHECK-LABEL: test_vshrq_n_u16
  return vshrq_n_u16(a, 3);
  // CHECK: ushr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vshrq_n_u32(int32x4_t a) {
  // CHECK-LABEL: test_vshrq_n_u32
  return vshrq_n_u32(a, 3);
  // CHECK: ushr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vshrq_n_u64(int64x2_t a) {
  // CHECK-LABEL: test_vshrq_n_u64
  return vshrq_n_u64(a, 3);
  // CHECK: ushr {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vsra_n_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vsra_n_s8
  return vsra_n_s8(a, b, 3);
  // CHECK: ssra {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vsra_n_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vsra_n_s16
  return vsra_n_s16(a, b, 3);
  // CHECK: ssra {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vsra_n_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vsra_n_s32
  return vsra_n_s32(a, b, 3);
  // CHECK: ssra {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vsraq_n_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsraq_n_s8
  return vsraq_n_s8(a, b, 3);
  // CHECK: ssra {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vsraq_n_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsraq_n_s16
  return vsraq_n_s16(a, b, 3);
  // CHECK: ssra {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vsraq_n_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsraq_n_s32
  return vsraq_n_s32(a, b, 3);
  // CHECK: ssra {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vsraq_n_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsraq_n_s64
  return vsraq_n_s64(a, b, 3);
  // CHECK: ssra {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vsra_n_u8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vsra_n_u8
  return vsra_n_u8(a, b, 3);
  // CHECK: usra {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vsra_n_u16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vsra_n_u16
  return vsra_n_u16(a, b, 3);
  // CHECK: usra {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vsra_n_u32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vsra_n_u32
  return vsra_n_u32(a, b, 3);
  // CHECK: usra {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vsraq_n_u8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsraq_n_u8
  return vsraq_n_u8(a, b, 3);
  // CHECK: usra {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vsraq_n_u16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsraq_n_u16
  return vsraq_n_u16(a, b, 3);
  // CHECK: usra {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vsraq_n_u32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsraq_n_u32
  return vsraq_n_u32(a, b, 3);
  // CHECK: usra {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vsraq_n_u64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsraq_n_u64
  return vsraq_n_u64(a, b, 3);
  // CHECK: usra {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vrshr_n_s8(int8x8_t a) {
  // CHECK-LABEL: test_vrshr_n_s8
  return vrshr_n_s8(a, 3);
  // CHECK: srshr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vrshr_n_s16(int16x4_t a) {
  // CHECK-LABEL: test_vrshr_n_s16
  return vrshr_n_s16(a, 3);
  // CHECK: srshr {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vrshr_n_s32(int32x2_t a) {
  // CHECK-LABEL: test_vrshr_n_s32
  return vrshr_n_s32(a, 3);
  // CHECK: srshr {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vrshrq_n_s8(int8x16_t a) {
  // CHECK-LABEL: test_vrshrq_n_s8
  return vrshrq_n_s8(a, 3);
  // CHECK: srshr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vrshrq_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vrshrq_n_s16
  return vrshrq_n_s16(a, 3);
  // CHECK: srshr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vrshrq_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vrshrq_n_s32
  return vrshrq_n_s32(a, 3);
  // CHECK: srshr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vrshrq_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vrshrq_n_s64
  return vrshrq_n_s64(a, 3);
  // CHECK: srshr {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vrshr_n_u8(int8x8_t a) {
  // CHECK-LABEL: test_vrshr_n_u8
  return vrshr_n_u8(a, 3);
  // CHECK: urshr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vrshr_n_u16(int16x4_t a) {
  // CHECK-LABEL: test_vrshr_n_u16
  return vrshr_n_u16(a, 3);
  // CHECK: urshr {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vrshr_n_u32(int32x2_t a) {
  // CHECK-LABEL: test_vrshr_n_u32
  return vrshr_n_u32(a, 3);
  // CHECK: urshr {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vrshrq_n_u8(int8x16_t a) {
  // CHECK-LABEL: test_vrshrq_n_u8
  return vrshrq_n_u8(a, 3);
  // CHECK: urshr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vrshrq_n_u16(int16x8_t a) {
  // CHECK-LABEL: test_vrshrq_n_u16
  return vrshrq_n_u16(a, 3);
  // CHECK: urshr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vrshrq_n_u32(int32x4_t a) {
  // CHECK-LABEL: test_vrshrq_n_u32
  return vrshrq_n_u32(a, 3);
  // CHECK: urshr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vrshrq_n_u64(int64x2_t a) {
  // CHECK-LABEL: test_vrshrq_n_u64
  return vrshrq_n_u64(a, 3);
  // CHECK: urshr {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vrsra_n_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vrsra_n_s8
  return vrsra_n_s8(a, b, 3);
  // CHECK: srsra {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vrsra_n_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vrsra_n_s16
  return vrsra_n_s16(a, b, 3);
  // CHECK: srsra {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vrsra_n_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vrsra_n_s32
  return vrsra_n_s32(a, b, 3);
  // CHECK: srsra {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vrsraq_n_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vrsraq_n_s8
  return vrsraq_n_s8(a, b, 3);
  // CHECK: srsra {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vrsraq_n_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vrsraq_n_s16
  return vrsraq_n_s16(a, b, 3);
  // CHECK: srsra {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vrsraq_n_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vrsraq_n_s32
  return vrsraq_n_s32(a, b, 3);
  // CHECK: srsra {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vrsraq_n_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vrsraq_n_s64
  return vrsraq_n_s64(a, b, 3);
  // CHECK: srsra {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vrsra_n_u8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vrsra_n_u8
  return vrsra_n_u8(a, b, 3);
  // CHECK: ursra {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vrsra_n_u16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vrsra_n_u16
  return vrsra_n_u16(a, b, 3);
  // CHECK: ursra {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vrsra_n_u32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vrsra_n_u32
  return vrsra_n_u32(a, b, 3);
  // CHECK: ursra {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vrsraq_n_u8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vrsraq_n_u8
  return vrsraq_n_u8(a, b, 3);
  // CHECK: ursra {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vrsraq_n_u16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vrsraq_n_u16
  return vrsraq_n_u16(a, b, 3);
  // CHECK: ursra {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vrsraq_n_u32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vrsraq_n_u32
  return vrsraq_n_u32(a, b, 3);
  // CHECK: ursra {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vrsraq_n_u64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vrsraq_n_u64
  return vrsraq_n_u64(a, b, 3);
  // CHECK: ursra {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vsri_n_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vsri_n_s8
  return vsri_n_s8(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vsri_n_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vsri_n_s16
  return vsri_n_s16(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vsri_n_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vsri_n_s32
  return vsri_n_s32(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vsriq_n_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsriq_n_s8
  return vsriq_n_s8(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vsriq_n_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsriq_n_s16
  return vsriq_n_s16(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vsriq_n_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsriq_n_s32
  return vsriq_n_s32(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vsriq_n_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsriq_n_s64
  return vsriq_n_s64(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vsri_n_u8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vsri_n_u8
  return vsri_n_u8(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vsri_n_u16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vsri_n_u16
  return vsri_n_u16(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vsri_n_u32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vsri_n_u32
  return vsri_n_u32(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vsriq_n_u8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsriq_n_u8
  return vsriq_n_u8(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vsriq_n_u16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsriq_n_u16
  return vsriq_n_u16(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vsriq_n_u32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsriq_n_u32
  return vsriq_n_u32(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vsriq_n_u64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsriq_n_u64
  return vsriq_n_u64(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

poly8x8_t test_vsri_n_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vsri_n_p8
  return vsri_n_p8(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

poly16x4_t test_vsri_n_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vsri_n_p16
  return vsri_n_p16(a, b, 15);
  // CHECK: sri {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #15
}

poly8x16_t test_vsriq_n_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vsriq_n_p8
  return vsriq_n_p8(a, b, 3);
  // CHECK: sri {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

poly16x8_t test_vsriq_n_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vsriq_n_p16
  return vsriq_n_p16(a, b, 15);
  // CHECK: sri {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #15
}

int8x8_t test_vsli_n_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vsli_n_s8
  return vsli_n_s8(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vsli_n_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vsli_n_s16
  return vsli_n_s16(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vsli_n_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vsli_n_s32
  return vsli_n_s32(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vsliq_n_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsliq_n_s8
  return vsliq_n_s8(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vsliq_n_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsliq_n_s16
  return vsliq_n_s16(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vsliq_n_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsliq_n_s32
  return vsliq_n_s32(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vsliq_n_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsliq_n_s64
  return vsliq_n_s64(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

uint8x8_t test_vsli_n_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vsli_n_u8
  return vsli_n_u8(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

uint16x4_t test_vsli_n_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vsli_n_u16
  return vsli_n_u16(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

uint32x2_t test_vsli_n_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vsli_n_u32
  return vsli_n_u32(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

uint8x16_t test_vsliq_n_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vsliq_n_u8
  return vsliq_n_u8(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

uint16x8_t test_vsliq_n_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vsliq_n_u16
  return vsliq_n_u16(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

uint32x4_t test_vsliq_n_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vsliq_n_u32
  return vsliq_n_u32(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

uint64x2_t test_vsliq_n_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vsliq_n_u64
  return vsliq_n_u64(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

poly8x8_t test_vsli_n_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vsli_n_p8
  return vsli_n_p8(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

poly16x4_t test_vsli_n_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vsli_n_p16
  return vsli_n_p16(a, b, 15);
  // CHECK: sli {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #15
}

poly8x16_t test_vsliq_n_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vsliq_n_p8
  return vsliq_n_p8(a, b, 3);
  // CHECK: sli {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

poly16x8_t test_vsliq_n_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vsliq_n_p16
  return vsliq_n_p16(a, b, 15);
  // CHECK: sli {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #15
}

int8x8_t test_vqshlu_n_s8(int8x8_t a) {
  // CHECK-LABEL: test_vqshlu_n_s8
  return vqshlu_n_s8(a, 3);
  // CHECK: sqshlu {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
}

int16x4_t test_vqshlu_n_s16(int16x4_t a) {
  // CHECK-LABEL: test_vqshlu_n_s16
  return vqshlu_n_s16(a, 3);
  // CHECK: sqshlu {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
}

int32x2_t test_vqshlu_n_s32(int32x2_t a) {
  // CHECK-LABEL: test_vqshlu_n_s32
  return vqshlu_n_s32(a, 3);
  // CHECK: sqshlu {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
}

int8x16_t test_vqshluq_n_s8(int8x16_t a) {
  // CHECK-LABEL: test_vqshluq_n_s8
  return vqshluq_n_s8(a, 3);
  // CHECK: sqshlu {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
}

int16x8_t test_vqshluq_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vqshluq_n_s16
  return vqshluq_n_s16(a, 3);
  // CHECK: sqshlu {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
}

int32x4_t test_vqshluq_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vqshluq_n_s32
  return vqshluq_n_s32(a, 3);
  // CHECK: sqshlu {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
}

int64x2_t test_vqshluq_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vqshluq_n_s64
  return vqshluq_n_s64(a, 3);
  // CHECK: sqshlu {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
}

int8x8_t test_vshrn_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vshrn_n_s16
  return vshrn_n_s16(a, 3);
  // CHECK: shrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

int16x4_t test_vshrn_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vshrn_n_s32
  return vshrn_n_s32(a, 9);
  // CHECK: shrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

int32x2_t test_vshrn_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vshrn_n_s64
  return vshrn_n_s64(a, 19);
  // CHECK: shrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

uint8x8_t test_vshrn_n_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vshrn_n_u16
  return vshrn_n_u16(a, 3);
  // CHECK: shrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

uint16x4_t test_vshrn_n_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vshrn_n_u32
  return vshrn_n_u32(a, 9);
  // CHECK: shrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

uint32x2_t test_vshrn_n_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vshrn_n_u64
  return vshrn_n_u64(a, 19);
  // CHECK: shrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

int8x16_t test_vshrn_high_n_s16(int8x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vshrn_high_n_s16
  return vshrn_high_n_s16(a, b, 3);
  // CHECK: shrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

int16x8_t test_vshrn_high_n_s32(int16x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vshrn_high_n_s32
  return vshrn_high_n_s32(a, b, 9);
  // CHECK: shrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

int32x4_t test_vshrn_high_n_s64(int32x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vshrn_high_n_s64
  return vshrn_high_n_s64(a, b, 19);
  // CHECK: shrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

uint8x16_t test_vshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vshrn_high_n_u16
  return vshrn_high_n_u16(a, b, 3);
  // CHECK: shrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

uint16x8_t test_vshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vshrn_high_n_u32
  return vshrn_high_n_u32(a, b, 9);
  // CHECK: shrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

uint32x4_t test_vshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vshrn_high_n_u64
  return vshrn_high_n_u64(a, b, 19);
  // CHECK: shrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

int8x8_t test_vqshrun_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vqshrun_n_s16
  return vqshrun_n_s16(a, 3);
  // CHECK: sqshrun {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

int16x4_t test_vqshrun_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vqshrun_n_s32
  return vqshrun_n_s32(a, 9);
  // CHECK: sqshrun {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

int32x2_t test_vqshrun_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vqshrun_n_s64
  return vqshrun_n_s64(a, 19);
  // CHECK: sqshrun {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

int8x16_t test_vqshrun_high_n_s16(int8x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vqshrun_high_n_s16
  return vqshrun_high_n_s16(a, b, 3);
  // CHECK: sqshrun2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

int16x8_t test_vqshrun_high_n_s32(int16x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vqshrun_high_n_s32
  return vqshrun_high_n_s32(a, b, 9);
  // CHECK: sqshrun2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

int32x4_t test_vqshrun_high_n_s64(int32x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vqshrun_high_n_s64
  return vqshrun_high_n_s64(a, b, 19);
  // CHECK: sqshrun2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

int8x8_t test_vrshrn_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vrshrn_n_s16
  return vrshrn_n_s16(a, 3);
  // CHECK: rshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

int16x4_t test_vrshrn_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vrshrn_n_s32
  return vrshrn_n_s32(a, 9);
  // CHECK: rshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

int32x2_t test_vrshrn_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vrshrn_n_s64
  return vrshrn_n_s64(a, 19);
  // CHECK: rshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

uint8x8_t test_vrshrn_n_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vrshrn_n_u16
  return vrshrn_n_u16(a, 3);
  // CHECK: rshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

uint16x4_t test_vrshrn_n_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vrshrn_n_u32
  return vrshrn_n_u32(a, 9);
  // CHECK: rshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

uint32x2_t test_vrshrn_n_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vrshrn_n_u64
  return vrshrn_n_u64(a, 19);
  // CHECK: rshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

int8x16_t test_vrshrn_high_n_s16(int8x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vrshrn_high_n_s16
  return vrshrn_high_n_s16(a, b, 3);
  // CHECK: rshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

int16x8_t test_vrshrn_high_n_s32(int16x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vrshrn_high_n_s32
  return vrshrn_high_n_s32(a, b, 9);
  // CHECK: rshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

int32x4_t test_vrshrn_high_n_s64(int32x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vrshrn_high_n_s64
  return vrshrn_high_n_s64(a, b, 19);
  // CHECK: rshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

uint8x16_t test_vrshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vrshrn_high_n_u16
  return vrshrn_high_n_u16(a, b, 3);
  // CHECK: rshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

uint16x8_t test_vrshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vrshrn_high_n_u32
  return vrshrn_high_n_u32(a, b, 9);
  // CHECK: rshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

uint32x4_t test_vrshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vrshrn_high_n_u64
  return vrshrn_high_n_u64(a, b, 19);
  // CHECK: rshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

int8x8_t test_vqrshrun_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vqrshrun_n_s16
  return vqrshrun_n_s16(a, 3);
  // CHECK: sqrshrun {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

int16x4_t test_vqrshrun_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vqrshrun_n_s32
  return vqrshrun_n_s32(a, 9);
  // CHECK: sqrshrun {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

int32x2_t test_vqrshrun_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vqrshrun_n_s64
  return vqrshrun_n_s64(a, 19);
  // CHECK: sqrshrun {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

int8x16_t test_vqrshrun_high_n_s16(int8x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vqrshrun_high_n_s16
  return vqrshrun_high_n_s16(a, b, 3);
  // CHECK: sqrshrun2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

int16x8_t test_vqrshrun_high_n_s32(int16x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vqrshrun_high_n_s32
  return vqrshrun_high_n_s32(a, b, 9);
  // CHECK: sqrshrun2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

int32x4_t test_vqrshrun_high_n_s64(int32x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vqrshrun_high_n_s64
  return vqrshrun_high_n_s64(a, b, 19);
  // CHECK: sqrshrun2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

int8x8_t test_vqshrn_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vqshrn_n_s16
  return vqshrn_n_s16(a, 3);
  // CHECK: sqshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

int16x4_t test_vqshrn_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vqshrn_n_s32
  return vqshrn_n_s32(a, 9);
  // CHECK: sqshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

int32x2_t test_vqshrn_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vqshrn_n_s64
  return vqshrn_n_s64(a, 19);
  // CHECK: sqshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

uint8x8_t test_vqshrn_n_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vqshrn_n_u16
  return vqshrn_n_u16(a, 3);
  // CHECK: uqshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

uint16x4_t test_vqshrn_n_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vqshrn_n_u32
  return vqshrn_n_u32(a, 9);
  // CHECK: uqshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

uint32x2_t test_vqshrn_n_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vqshrn_n_u64
  return vqshrn_n_u64(a, 19);
  // CHECK: uqshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

int8x16_t test_vqshrn_high_n_s16(int8x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vqshrn_high_n_s16
  return vqshrn_high_n_s16(a, b, 3);
  // CHECK: sqshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

int16x8_t test_vqshrn_high_n_s32(int16x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vqshrn_high_n_s32
  return vqshrn_high_n_s32(a, b, 9);
  // CHECK: sqshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

int32x4_t test_vqshrn_high_n_s64(int32x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vqshrn_high_n_s64
  return vqshrn_high_n_s64(a, b, 19);
  // CHECK: sqshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

uint8x16_t test_vqshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vqshrn_high_n_u16
  return vqshrn_high_n_u16(a, b, 3);
  // CHECK: uqshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

uint16x8_t test_vqshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vqshrn_high_n_u32
  return vqshrn_high_n_u32(a, b, 9);
  // CHECK: uqshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

uint32x4_t test_vqshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vqshrn_high_n_u64
  return vqshrn_high_n_u64(a, b, 19);
  // CHECK: uqshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

int8x8_t test_vqrshrn_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vqrshrn_n_s16
  return vqrshrn_n_s16(a, 3);
  // CHECK: sqrshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

int16x4_t test_vqrshrn_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vqrshrn_n_s32
  return vqrshrn_n_s32(a, 9);
  // CHECK: sqrshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

int32x2_t test_vqrshrn_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vqrshrn_n_s64
  return vqrshrn_n_s64(a, 19);
  // CHECK: sqrshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

uint8x8_t test_vqrshrn_n_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vqrshrn_n_u16
  return vqrshrn_n_u16(a, 3);
  // CHECK: uqrshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
}

uint16x4_t test_vqrshrn_n_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vqrshrn_n_u32
  return vqrshrn_n_u32(a, 9);
  // CHECK: uqrshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
}

uint32x2_t test_vqrshrn_n_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vqrshrn_n_u64
  return vqrshrn_n_u64(a, 19);
  // CHECK: uqrshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
}

int8x16_t test_vqrshrn_high_n_s16(int8x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vqrshrn_high_n_s16
  return vqrshrn_high_n_s16(a, b, 3);
  // CHECK: sqrshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

int16x8_t test_vqrshrn_high_n_s32(int16x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vqrshrn_high_n_s32
  return vqrshrn_high_n_s32(a, b, 9);
  // CHECK: sqrshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

int32x4_t test_vqrshrn_high_n_s64(int32x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vqrshrn_high_n_s64
  return vqrshrn_high_n_s64(a, b, 19);
  // CHECK: sqrshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

uint8x16_t test_vqrshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vqrshrn_high_n_u16
  return vqrshrn_high_n_u16(a, b, 3);
  // CHECK: uqrshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
}

uint16x8_t test_vqrshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vqrshrn_high_n_u32
  return vqrshrn_high_n_u32(a, b, 9);
  // CHECK: uqrshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
}

uint32x4_t test_vqrshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vqrshrn_high_n_u64
  return vqrshrn_high_n_u64(a, b, 19);
  // CHECK: uqrshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
}

int16x8_t test_vshll_n_s8(int8x8_t a) {
// CHECK-LABEL: test_vshll_n_s8
  return vshll_n_s8(a, 3);
// CHECK: sshll {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, #3
}

int32x4_t test_vshll_n_s16(int16x4_t a) {
// CHECK-LABEL: test_vshll_n_s16
  return vshll_n_s16(a, 9);
// CHECK: sshll {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, #9
}

int64x2_t test_vshll_n_s32(int32x2_t a) {
// CHECK-LABEL: test_vshll_n_s32
  return vshll_n_s32(a, 19);
// CHECK: sshll {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, #19
}

uint16x8_t test_vshll_n_u8(uint8x8_t a) {
// CHECK-LABEL: test_vshll_n_u8
  return vshll_n_u8(a, 3);
// CHECK: ushll {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, #3
}

uint32x4_t test_vshll_n_u16(uint16x4_t a) {
// CHECK-LABEL: test_vshll_n_u16
  return vshll_n_u16(a, 9);
// CHECK: ushll {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, #9
}

uint64x2_t test_vshll_n_u32(uint32x2_t a) {
// CHECK-LABEL: test_vshll_n_u32
  return vshll_n_u32(a, 19);
// CHECK: ushll {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, #19
}

int16x8_t test_vshll_high_n_s8(int8x16_t a) {
// CHECK-LABEL: test_vshll_high_n_s8
  return vshll_high_n_s8(a, 3);
// CHECK: sshll2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, #3
}

int32x4_t test_vshll_high_n_s16(int16x8_t a) {
// CHECK-LABEL: test_vshll_high_n_s16
  return vshll_high_n_s16(a, 9);
// CHECK: sshll2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, #9
}

int64x2_t test_vshll_high_n_s32(int32x4_t a) {
// CHECK-LABEL: test_vshll_high_n_s32
  return vshll_high_n_s32(a, 19);
// CHECK: sshll2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, #19
}

uint16x8_t test_vshll_high_n_u8(uint8x16_t a) {
// CHECK-LABEL: test_vshll_high_n_u8
  return vshll_high_n_u8(a, 3);
// CHECK: ushll2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, #3
}

uint32x4_t test_vshll_high_n_u16(uint16x8_t a) {
// CHECK-LABEL: test_vshll_high_n_u16
  return vshll_high_n_u16(a, 9);
// CHECK: ushll2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, #9
}

uint64x2_t test_vshll_high_n_u32(uint32x4_t a) {
// CHECK-LABEL: test_vshll_high_n_u32
  return vshll_high_n_u32(a, 19);
// CHECK: ushll2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, #19
}

int16x8_t test_vmovl_s8(int8x8_t a) {
// CHECK-LABEL: test_vmovl_s8
  return vmovl_s8(a);
// CHECK: sshll {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, #0
}

int32x4_t test_vmovl_s16(int16x4_t a) {
// CHECK-LABEL: test_vmovl_s16
  return vmovl_s16(a);
// CHECK: sshll {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, #0
}

int64x2_t test_vmovl_s32(int32x2_t a) {
// CHECK-LABEL: test_vmovl_s32
  return vmovl_s32(a);
// CHECK: sshll {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, #0
}

uint16x8_t test_vmovl_u8(uint8x8_t a) {
// CHECK-LABEL: test_vmovl_u8
  return vmovl_u8(a);
// CHECK: ushll {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, #0
}

uint32x4_t test_vmovl_u16(uint16x4_t a) {
// CHECK-LABEL: test_vmovl_u16
  return vmovl_u16(a);
// CHECK: ushll {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, #0
}

uint64x2_t test_vmovl_u32(uint32x2_t a) {
// CHECK-LABEL: test_vmovl_u32
  return vmovl_u32(a);
// CHECK: ushll {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, #0
}

int16x8_t test_vmovl_high_s8(int8x16_t a) {
// CHECK-LABEL: test_vmovl_high_s8
  return vmovl_high_s8(a);
// CHECK: sshll2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, #0
}

int32x4_t test_vmovl_high_s16(int16x8_t a) {
// CHECK-LABEL: test_vmovl_high_s16
  return vmovl_high_s16(a);
// CHECK: sshll2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, #0
}

int64x2_t test_vmovl_high_s32(int32x4_t a) {
// CHECK-LABEL: test_vmovl_high_s32
  return vmovl_high_s32(a);
// CHECK: sshll2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, #0
}

uint16x8_t test_vmovl_high_u8(uint8x16_t a) {
// CHECK-LABEL: test_vmovl_high_u8
  return vmovl_high_u8(a);
// CHECK: ushll2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, #0
}

uint32x4_t test_vmovl_high_u16(uint16x8_t a) {
// CHECK-LABEL: test_vmovl_high_u16
  return vmovl_high_u16(a);
// CHECK: ushll2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, #0
}

uint64x2_t test_vmovl_high_u32(uint32x4_t a) {
// CHECK-LABEL: test_vmovl_high_u32
  return vmovl_high_u32(a);
// CHECK: ushll2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, #0
}

float32x2_t test_vcvt_n_f32_s32(int32x2_t a) {
  // CHECK-LABEL: test_vcvt_n_f32_s32
  return vcvt_n_f32_s32(a, 31);
  // CHECK: scvtf {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #31
}

float32x4_t test_vcvtq_n_f32_s32(int32x4_t a) {
  // CHECK-LABEL: test_vcvtq_n_f32_s32
  return vcvtq_n_f32_s32(a, 31);
  // CHECK: scvtf {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #31
}

float64x2_t test_vcvtq_n_f64_s64(int64x2_t a) {
  // CHECK-LABEL: test_vcvtq_n_f64_s64
  return vcvtq_n_f64_s64(a, 50);
  // CHECK: scvtf {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #50
}

float32x2_t test_vcvt_n_f32_u32(uint32x2_t a) {
  // CHECK-LABEL: test_vcvt_n_f32_u32
  return vcvt_n_f32_u32(a, 31);
  // CHECK: ucvtf {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #31
}

float32x4_t test_vcvtq_n_f32_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vcvtq_n_f32_u32
  return vcvtq_n_f32_u32(a, 31);
  // CHECK: ucvtf {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #31
}

float64x2_t test_vcvtq_n_f64_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vcvtq_n_f64_u64
  return vcvtq_n_f64_u64(a, 50);
  // CHECK: ucvtf {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #50
}

int32x2_t test_vcvt_n_s32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvt_n_s32_f32
  return vcvt_n_s32_f32(a, 31);
  // CHECK: fcvtzs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #31
}

int32x4_t test_vcvtq_n_s32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtq_n_s32_f32
  return vcvtq_n_s32_f32(a, 31);
  // CHECK: fcvtzs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #31
}

int64x2_t test_vcvtq_n_s64_f64(float64x2_t a) {
  // CHECK-LABEL: test_vcvtq_n_s64_f64
  return vcvtq_n_s64_f64(a, 50);
  // CHECK: fcvtzs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #50
}

uint32x2_t test_vcvt_n_u32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvt_n_u32_f32
  return vcvt_n_u32_f32(a, 31);
  // CHECK: fcvtzu {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #31
}

uint32x4_t test_vcvtq_n_u32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtq_n_u32_f32
  return vcvtq_n_u32_f32(a, 31);
  // CHECK: fcvtzu {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #31
}

uint64x2_t test_vcvtq_n_u64_f64(float64x2_t a) {
  // CHECK-LABEL: test_vcvtq_n_u64_f64
  return vcvtq_n_u64_f64(a, 50);
  // CHECK: fcvtzu {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #50
}

int16x8_t test_vaddl_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vaddl_s8
  return vaddl_s8(a, b);
  // CHECK: saddl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x4_t test_vaddl_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vaddl_s16
  return vaddl_s16(a, b);
  // CHECK: saddl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int64x2_t test_vaddl_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vaddl_s32
  return vaddl_s32(a, b);
  // CHECK: saddl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint16x8_t test_vaddl_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vaddl_u8
  return vaddl_u8(a, b);
  // CHECK: uaddl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x4_t test_vaddl_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vaddl_u16
  return vaddl_u16(a, b);
  // CHECK: uaddl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint64x2_t test_vaddl_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vaddl_u32
  return vaddl_u32(a, b);
  // CHECK: uaddl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vaddl_high_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vaddl_high_s8
  return vaddl_high_s8(a, b);
  // CHECK: saddl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x4_t test_vaddl_high_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vaddl_high_s16
  return vaddl_high_s16(a, b);
  // CHECK: saddl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int64x2_t test_vaddl_high_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vaddl_high_s32
  return vaddl_high_s32(a, b);
  // CHECK: saddl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint16x8_t test_vaddl_high_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vaddl_high_u8
  return vaddl_high_u8(a, b);
  // CHECK: uaddl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint32x4_t test_vaddl_high_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vaddl_high_u16
  return vaddl_high_u16(a, b);
  // CHECK: uaddl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint64x2_t test_vaddl_high_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vaddl_high_u32
  return vaddl_high_u32(a, b);
  // CHECK: uaddl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int16x8_t test_vaddw_s8(int16x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vaddw_s8
  return vaddw_s8(a, b);
  // CHECK: saddw {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8b
}

int32x4_t test_vaddw_s16(int32x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vaddw_s16
  return vaddw_s16(a, b);
  // CHECK: saddw {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4h
}

int64x2_t test_vaddw_s32(int64x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vaddw_s32
  return vaddw_s32(a, b);
  // CHECK: saddw {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2s
}

uint16x8_t test_vaddw_u8(uint16x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vaddw_u8
  return vaddw_u8(a, b);
  // CHECK: uaddw {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8b
}

uint32x4_t test_vaddw_u16(uint32x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vaddw_u16
  return vaddw_u16(a, b);
  // CHECK: uaddw {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4h
}

uint64x2_t test_vaddw_u32(uint64x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vaddw_u32
  return vaddw_u32(a, b);
  // CHECK: uaddw {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2s
}

int16x8_t test_vaddw_high_s8(int16x8_t a, int8x16_t b) {
  // CHECK-LABEL: test_vaddw_high_s8
  return vaddw_high_s8(a, b);
  // CHECK: saddw2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.16b
}

int32x4_t test_vaddw_high_s16(int32x4_t a, int16x8_t b) {
  // CHECK-LABEL: test_vaddw_high_s16
  return vaddw_high_s16(a, b);
  // CHECK: saddw2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.8h
}

int64x2_t test_vaddw_high_s32(int64x2_t a, int32x4_t b) {
  // CHECK-LABEL: test_vaddw_high_s32
  return vaddw_high_s32(a, b);
  // CHECK: saddw2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.4s
}

uint16x8_t test_vaddw_high_u8(uint16x8_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vaddw_high_u8
  return vaddw_high_u8(a, b);
  // CHECK: uaddw2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.16b
}

uint32x4_t test_vaddw_high_u16(uint32x4_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vaddw_high_u16
  return vaddw_high_u16(a, b);
  // CHECK: uaddw2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.8h
}

uint64x2_t test_vaddw_high_u32(uint64x2_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vaddw_high_u32
  return vaddw_high_u32(a, b);
  // CHECK: uaddw2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.4s
}

int16x8_t test_vsubl_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vsubl_s8
  return vsubl_s8(a, b);
  // CHECK: ssubl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x4_t test_vsubl_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vsubl_s16
  return vsubl_s16(a, b);
  // CHECK: ssubl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int64x2_t test_vsubl_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vsubl_s32
  return vsubl_s32(a, b);
  // CHECK: ssubl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint16x8_t test_vsubl_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vsubl_u8
  return vsubl_u8(a, b);
  // CHECK: usubl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x4_t test_vsubl_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vsubl_u16
  return vsubl_u16(a, b);
  // CHECK: usubl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint64x2_t test_vsubl_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vsubl_u32
  return vsubl_u32(a, b);
  // CHECK: usubl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vsubl_high_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsubl_high_s8
  return vsubl_high_s8(a, b);
  // CHECK: ssubl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x4_t test_vsubl_high_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsubl_high_s16
  return vsubl_high_s16(a, b);
  // CHECK: ssubl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int64x2_t test_vsubl_high_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsubl_high_s32
  return vsubl_high_s32(a, b);
  // CHECK: ssubl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint16x8_t test_vsubl_high_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vsubl_high_u8
  return vsubl_high_u8(a, b);
  // CHECK: usubl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint32x4_t test_vsubl_high_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vsubl_high_u16
  return vsubl_high_u16(a, b);
  // CHECK: usubl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint64x2_t test_vsubl_high_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vsubl_high_u32
  return vsubl_high_u32(a, b);
  // CHECK: usubl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int16x8_t test_vsubw_s8(int16x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vsubw_s8
  return vsubw_s8(a, b);
  // CHECK: ssubw {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8b
}

int32x4_t test_vsubw_s16(int32x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vsubw_s16
  return vsubw_s16(a, b);
  // CHECK: ssubw {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4h
}

int64x2_t test_vsubw_s32(int64x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vsubw_s32
  return vsubw_s32(a, b);
  // CHECK: ssubw {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2s
}

uint16x8_t test_vsubw_u8(uint16x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vsubw_u8
  return vsubw_u8(a, b);
  // CHECK: usubw {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8b
}

uint32x4_t test_vsubw_u16(uint32x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vsubw_u16
  return vsubw_u16(a, b);
  // CHECK: usubw {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4h
}

uint64x2_t test_vsubw_u32(uint64x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vsubw_u32
  return vsubw_u32(a, b);
  // CHECK: usubw {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2s
}

int16x8_t test_vsubw_high_s8(int16x8_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsubw_high_s8
  return vsubw_high_s8(a, b);
  // CHECK: ssubw2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.16b
}

int32x4_t test_vsubw_high_s16(int32x4_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsubw_high_s16
  return vsubw_high_s16(a, b);
  // CHECK: ssubw2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.8h
}

int64x2_t test_vsubw_high_s32(int64x2_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsubw_high_s32
  return vsubw_high_s32(a, b);
  // CHECK: ssubw2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.4s
}

uint16x8_t test_vsubw_high_u8(uint16x8_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vsubw_high_u8
  return vsubw_high_u8(a, b);
  // CHECK: usubw2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.16b
}

uint32x4_t test_vsubw_high_u16(uint32x4_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vsubw_high_u16
  return vsubw_high_u16(a, b);
  // CHECK: usubw2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.8h
}

uint64x2_t test_vsubw_high_u32(uint64x2_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vsubw_high_u32
  return vsubw_high_u32(a, b);
  // CHECK: usubw2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.4s
}

int8x8_t test_vaddhn_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vaddhn_s16
  return vaddhn_s16(a, b);
  // CHECK: addhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int16x4_t test_vaddhn_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vaddhn_s32
  return vaddhn_s32(a, b);
  // CHECK: addhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x2_t test_vaddhn_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vaddhn_s64
  return vaddhn_s64(a, b);
  // CHECK: addhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x8_t test_vaddhn_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vaddhn_u16
  return vaddhn_u16(a, b);
  // CHECK: addhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint16x4_t test_vaddhn_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vaddhn_u32
  return vaddhn_u32(a, b);
  // CHECK: addhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x2_t test_vaddhn_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vaddhn_u64
  return vaddhn_u64(a, b);
  // CHECK: addhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x16_t test_vaddhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vaddhn_high_s16
  return vaddhn_high_s16(r, a, b);
  // CHECK: addhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int16x8_t test_vaddhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vaddhn_high_s32
  return vaddhn_high_s32(r, a, b);
  // CHECK: addhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x4_t test_vaddhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vaddhn_high_s64
  return vaddhn_high_s64(r, a, b);
  // CHECK: addhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vaddhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vaddhn_high_u16
  return vaddhn_high_u16(r, a, b);
  // CHECK: addhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint16x8_t test_vaddhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vaddhn_high_u32
  return vaddhn_high_u32(r, a, b);
  // CHECK: addhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vaddhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vaddhn_high_u64
  return vaddhn_high_u64(r, a, b);
  // CHECK: addhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vraddhn_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vraddhn_s16
  return vraddhn_s16(a, b);
  // CHECK: raddhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int16x4_t test_vraddhn_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vraddhn_s32
  return vraddhn_s32(a, b);
  // CHECK: raddhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x2_t test_vraddhn_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vraddhn_s64
  return vraddhn_s64(a, b);
  // CHECK: raddhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x8_t test_vraddhn_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vraddhn_u16
  return vraddhn_u16(a, b);
  // CHECK: raddhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint16x4_t test_vraddhn_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vraddhn_u32
  return vraddhn_u32(a, b);
  // CHECK: raddhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x2_t test_vraddhn_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vraddhn_u64
  return vraddhn_u64(a, b);
  // CHECK: raddhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x16_t test_vraddhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vraddhn_high_s16
  return vraddhn_high_s16(r, a, b);
  // CHECK: raddhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int16x8_t test_vraddhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vraddhn_high_s32
  return vraddhn_high_s32(r, a, b);
  // CHECK: raddhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x4_t test_vraddhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vraddhn_high_s64
  return vraddhn_high_s64(r, a, b);
  // CHECK: raddhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vraddhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vraddhn_high_u16
  return vraddhn_high_u16(r, a, b);
  // CHECK: raddhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint16x8_t test_vraddhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vraddhn_high_u32
  return vraddhn_high_u32(r, a, b);
  // CHECK: raddhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vraddhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vraddhn_high_u64
  return vraddhn_high_u64(r, a, b);
  // CHECK: raddhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vsubhn_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsubhn_s16
  return vsubhn_s16(a, b);
  // CHECK: subhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int16x4_t test_vsubhn_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsubhn_s32
  return vsubhn_s32(a, b);
  // CHECK: subhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x2_t test_vsubhn_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsubhn_s64
  return vsubhn_s64(a, b);
  // CHECK: subhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x8_t test_vsubhn_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vsubhn_u16
  return vsubhn_u16(a, b);
  // CHECK: subhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint16x4_t test_vsubhn_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vsubhn_u32
  return vsubhn_u32(a, b);
  // CHECK: subhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x2_t test_vsubhn_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vsubhn_u64
  return vsubhn_u64(a, b);
  // CHECK: subhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x16_t test_vsubhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsubhn_high_s16
  return vsubhn_high_s16(r, a, b);
  // CHECK: subhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int16x8_t test_vsubhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsubhn_high_s32
  return vsubhn_high_s32(r, a, b);
  // CHECK: subhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x4_t test_vsubhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsubhn_high_s64
  return vsubhn_high_s64(r, a, b);
  // CHECK: subhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vsubhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vsubhn_high_u16
  return vsubhn_high_u16(r, a, b);
  // CHECK: subhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint16x8_t test_vsubhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vsubhn_high_u32
  return vsubhn_high_u32(r, a, b);
  // CHECK: subhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vsubhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vsubhn_high_u64
  return vsubhn_high_u64(r, a, b);
  // CHECK: subhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vrsubhn_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vrsubhn_s16
  return vrsubhn_s16(a, b);
  // CHECK: rsubhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int16x4_t test_vrsubhn_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vrsubhn_s32
  return vrsubhn_s32(a, b);
  // CHECK: rsubhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x2_t test_vrsubhn_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vrsubhn_s64
  return vrsubhn_s64(a, b);
  // CHECK: rsubhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x8_t test_vrsubhn_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vrsubhn_u16
  return vrsubhn_u16(a, b);
  // CHECK: rsubhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint16x4_t test_vrsubhn_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vrsubhn_u32
  return vrsubhn_u32(a, b);
  // CHECK: rsubhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x2_t test_vrsubhn_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vrsubhn_u64
  return vrsubhn_u64(a, b);
  // CHECK: rsubhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x16_t test_vrsubhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vrsubhn_high_s16
  return vrsubhn_high_s16(r, a, b);
  // CHECK: rsubhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int16x8_t test_vrsubhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vrsubhn_high_s32
  return vrsubhn_high_s32(r, a, b);
  // CHECK: rsubhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x4_t test_vrsubhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vrsubhn_high_s64
  return vrsubhn_high_s64(r, a, b);
  // CHECK: rsubhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vrsubhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vrsubhn_high_u16
  return vrsubhn_high_u16(r, a, b);
  // CHECK: rsubhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint16x8_t test_vrsubhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vrsubhn_high_u32
  return vrsubhn_high_u32(r, a, b);
  // CHECK: rsubhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vrsubhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vrsubhn_high_u64
  return vrsubhn_high_u64(r, a, b);
  // CHECK: rsubhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int16x8_t test_vabdl_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vabdl_s8
  return vabdl_s8(a, b);
  // CHECK: sabdl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
int32x4_t test_vabdl_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vabdl_s16
  return vabdl_s16(a, b);
  // CHECK: sabdl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int64x2_t test_vabdl_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vabdl_s32
  return vabdl_s32(a, b);
  // CHECK: sabdl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}
uint16x8_t test_vabdl_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vabdl_u8
  return vabdl_u8(a, b);
  // CHECK: uabdl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
uint32x4_t test_vabdl_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vabdl_u16
  return vabdl_u16(a, b);
  // CHECK: uabdl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
uint64x2_t test_vabdl_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vabdl_u32
  return vabdl_u32(a, b);
  // CHECK: uabdl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vabal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  // CHECK-LABEL: test_vabal_s8
  return vabal_s8(a, b, c);
  // CHECK: sabal {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
int32x4_t test_vabal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  // CHECK-LABEL: test_vabal_s16
  return vabal_s16(a, b, c);
  // CHECK: sabal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int64x2_t test_vabal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  // CHECK-LABEL: test_vabal_s32
  return vabal_s32(a, b, c);
  // CHECK: sabal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}
uint16x8_t test_vabal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vabal_u8
  return vabal_u8(a, b, c);
  // CHECK: uabal {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
uint32x4_t test_vabal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  // CHECK-LABEL: test_vabal_u16
  return vabal_u16(a, b, c);
  // CHECK: uabal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
uint64x2_t test_vabal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  // CHECK-LABEL: test_vabal_u32
  return vabal_u32(a, b, c);
  // CHECK: uabal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vabdl_high_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vabdl_high_s8
  return vabdl_high_s8(a, b);
  // CHECK: sabdl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
int32x4_t test_vabdl_high_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vabdl_high_s16
  return vabdl_high_s16(a, b);
  // CHECK: sabdl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int64x2_t test_vabdl_high_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vabdl_high_s32
  return vabdl_high_s32(a, b);
  // CHECK: sabdl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint16x8_t test_vabdl_high_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vabdl_high_u8
  return vabdl_high_u8(a, b);
  // CHECK: uabdl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
uint32x4_t test_vabdl_high_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vabdl_high_u16
  return vabdl_high_u16(a, b);
  // CHECK: uabdl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
uint64x2_t test_vabdl_high_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vabdl_high_u32
  return vabdl_high_u32(a, b);
  // CHECK: uabdl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int16x8_t test_vabal_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
  // CHECK-LABEL: test_vabal_high_s8
  return vabal_high_s8(a, b, c);
  // CHECK: sabal2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
int32x4_t test_vabal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  // CHECK-LABEL: test_vabal_high_s16
  return vabal_high_s16(a, b, c);
  // CHECK: sabal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int64x2_t test_vabal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  // CHECK-LABEL: test_vabal_high_s32
  return vabal_high_s32(a, b, c);
  // CHECK: sabal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint16x8_t test_vabal_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vabal_high_u8
  return vabal_high_u8(a, b, c);
  // CHECK: uabal2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
uint32x4_t test_vabal_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
  // CHECK-LABEL: test_vabal_high_u16
  return vabal_high_u16(a, b, c);
  // CHECK: uabal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
uint64x2_t test_vabal_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: test_vabal_high_u32
  return vabal_high_u32(a, b, c);
  // CHECK: uabal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int16x8_t test_vmull_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vmull_s8
  return vmull_s8(a, b);
  // CHECK: smull {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
int32x4_t test_vmull_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vmull_s16
  return vmull_s16(a, b);
  // CHECK: smull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int64x2_t test_vmull_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vmull_s32
  return vmull_s32(a, b);
  // CHECK: smull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}
uint16x8_t test_vmull_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vmull_u8
  return vmull_u8(a, b);
  // CHECK: umull {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
uint32x4_t test_vmull_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vmull_u16
  return vmull_u16(a, b);
  // CHECK: umull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
uint64x2_t test_vmull_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vmull_u32
  return vmull_u32(a, b);
  // CHECK: umull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vmull_high_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vmull_high_s8
  return vmull_high_s8(a, b);
  // CHECK: smull2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
int32x4_t test_vmull_high_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vmull_high_s16
  return vmull_high_s16(a, b);
  // CHECK: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int64x2_t test_vmull_high_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vmull_high_s32
  return vmull_high_s32(a, b);
  // CHECK: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint16x8_t test_vmull_high_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vmull_high_u8
  return vmull_high_u8(a, b);
  // CHECK: umull2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
uint32x4_t test_vmull_high_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vmull_high_u16
  return vmull_high_u16(a, b);
  // CHECK: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
uint64x2_t test_vmull_high_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vmull_high_u32
  return vmull_high_u32(a, b);
  // CHECK: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int16x8_t test_vmlal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  // CHECK-LABEL: test_vmlal_s8
  return vmlal_s8(a, b, c);
  // CHECK: smlal {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
int32x4_t test_vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  // CHECK-LABEL: test_vmlal_s16
  return vmlal_s16(a, b, c);
  // CHECK: smlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int64x2_t test_vmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  // CHECK-LABEL: test_vmlal_s32
  return vmlal_s32(a, b, c);
  // CHECK: smlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}
uint16x8_t test_vmlal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vmlal_u8
  return vmlal_u8(a, b, c);
  // CHECK: umlal {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
uint32x4_t test_vmlal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  // CHECK-LABEL: test_vmlal_u16
  return vmlal_u16(a, b, c);
  // CHECK: umlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
uint64x2_t test_vmlal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  // CHECK-LABEL: test_vmlal_u32
  return vmlal_u32(a, b, c);
  // CHECK: umlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vmlal_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
  // CHECK-LABEL: test_vmlal_high_s8
  return vmlal_high_s8(a, b, c);
  // CHECK: smlal2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
int32x4_t test_vmlal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  // CHECK-LABEL: test_vmlal_high_s16
  return vmlal_high_s16(a, b, c);
  // CHECK: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int64x2_t test_vmlal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  // CHECK-LABEL: test_vmlal_high_s32
  return vmlal_high_s32(a, b, c);
  // CHECK: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint16x8_t test_vmlal_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vmlal_high_u8
  return vmlal_high_u8(a, b, c);
  // CHECK: umlal2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
uint32x4_t test_vmlal_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
  // CHECK-LABEL: test_vmlal_high_u16
  return vmlal_high_u16(a, b, c);
  // CHECK: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
uint64x2_t test_vmlal_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: test_vmlal_high_u32
  return vmlal_high_u32(a, b, c);
  // CHECK: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int16x8_t test_vmlsl_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  // CHECK-LABEL: test_vmlsl_s8
  return vmlsl_s8(a, b, c);
  // CHECK: smlsl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
int32x4_t test_vmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  // CHECK-LABEL: test_vmlsl_s16
  return vmlsl_s16(a, b, c);
  // CHECK: smlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int64x2_t test_vmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  // CHECK-LABEL: test_vmlsl_s32
  return vmlsl_s32(a, b, c);
  // CHECK: smlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}
uint16x8_t test_vmlsl_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vmlsl_u8
  return vmlsl_u8(a, b, c);
  // CHECK: umlsl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
uint32x4_t test_vmlsl_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  // CHECK-LABEL: test_vmlsl_u16
  return vmlsl_u16(a, b, c);
  // CHECK: umlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
uint64x2_t test_vmlsl_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  // CHECK-LABEL: test_vmlsl_u32
  return vmlsl_u32(a, b, c);
  // CHECK: umlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vmlsl_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
  // CHECK-LABEL: test_vmlsl_high_s8
  return vmlsl_high_s8(a, b, c);
  // CHECK: smlsl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
int32x4_t test_vmlsl_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  // CHECK-LABEL: test_vmlsl_high_s16
  return vmlsl_high_s16(a, b, c);
  // CHECK: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int64x2_t test_vmlsl_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  // CHECK-LABEL: test_vmlsl_high_s32
  return vmlsl_high_s32(a, b, c);
  // CHECK: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint16x8_t test_vmlsl_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vmlsl_high_u8
  return vmlsl_high_u8(a, b, c);
  // CHECK: umlsl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
uint32x4_t test_vmlsl_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
  // CHECK-LABEL: test_vmlsl_high_u16
  return vmlsl_high_u16(a, b, c);
  // CHECK: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
uint64x2_t test_vmlsl_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: test_vmlsl_high_u32
  return vmlsl_high_u32(a, b, c);
  // CHECK: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x4_t test_vqdmull_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vqdmull_s16
  return vqdmull_s16(a, b);
  // CHECK: sqdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int64x2_t test_vqdmull_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vqdmull_s32
  return vqdmull_s32(a, b);
  // CHECK: sqdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int32x4_t test_vqdmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  // CHECK-LABEL: test_vqdmlal_s16
  return vqdmlal_s16(a, b, c);
  // CHECK: sqdmlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int64x2_t test_vqdmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  // CHECK-LABEL: test_vqdmlal_s32
  return vqdmlal_s32(a, b, c);
  // CHECK: sqdmlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int32x4_t test_vqdmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  // CHECK-LABEL: test_vqdmlsl_s16
  return vqdmlsl_s16(a, b, c);
  // CHECK: sqdmlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int64x2_t test_vqdmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  // CHECK-LABEL: test_vqdmlsl_s32
  return vqdmlsl_s32(a, b, c);
  // CHECK: sqdmlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int32x4_t test_vqdmull_high_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vqdmull_high_s16
  return vqdmull_high_s16(a, b);
  // CHECK: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int64x2_t test_vqdmull_high_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vqdmull_high_s32
  return vqdmull_high_s32(a, b);
  // CHECK: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x4_t test_vqdmlal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  // CHECK-LABEL: test_vqdmlal_high_s16
  return vqdmlal_high_s16(a, b, c);
  // CHECK: sqdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int64x2_t test_vqdmlal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  // CHECK-LABEL: test_vqdmlal_high_s32
  return vqdmlal_high_s32(a, b, c);
  // CHECK: sqdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int32x4_t test_vqdmlsl_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  // CHECK-LABEL: test_vqdmlsl_high_s16
  return vqdmlsl_high_s16(a, b, c);
  // CHECK: sqdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int64x2_t test_vqdmlsl_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  // CHECK-LABEL: test_vqdmlsl_high_s32
  return vqdmlsl_high_s32(a, b, c);
  // CHECK: sqdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

poly16x8_t test_vmull_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vmull_p8
  return vmull_p8(a, b);
  // CHECK: pmull {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly16x8_t test_vmull_high_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vmull_high_p8
  return vmull_high_p8(a, b);
  // CHECK: pmull2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int64_t test_vaddd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vaddd_s64
  return vaddd_s64(a, b);
// CHECK: add {{[xd][0-9]+}}, {{[xd][0-9]+}}, {{[xd][0-9]+}}
}

uint64_t test_vaddd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vaddd_u64
  return vaddd_u64(a, b);
// CHECK: add {{[xd][0-9]+}}, {{[xd][0-9]+}}, {{[xd][0-9]+}}
}

int64_t test_vsubd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vsubd_s64
  return vsubd_s64(a, b);
// CHECK: sub {{[xd][0-9]+}}, {{[xd][0-9]+}}, {{[xd][0-9]+}}
}

uint64_t test_vsubd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vsubd_u64
  return vsubd_u64(a, b);
// CHECK: sub {{[xd][0-9]+}}, {{[xd][0-9]+}}, {{[xd][0-9]+}}
}

int8_t test_vqaddb_s8(int8_t a, int8_t b) {
// CHECK-LABEL: test_vqaddb_s8
  return vqaddb_s8(a, b);
// CHECK: sqadd {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
}

int16_t test_vqaddh_s16(int16_t a, int16_t b) {
// CHECK-LABEL: test_vqaddh_s16
  return vqaddh_s16(a, b);
// CHECK: sqadd {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

int32_t test_vqadds_s32(int32_t a, int32_t b) {
// CHECK-LABEL: test_vqadds_s32
  return vqadds_s32(a, b);
// CHECK: sqadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

int64_t test_vqaddd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vqaddd_s64
  return vqaddd_s64(a, b);
// CHECK: sqadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8_t test_vqaddb_u8(uint8_t a, uint8_t b) {
// CHECK-LABEL: test_vqaddb_u8
  return vqaddb_u8(a, b);
// CHECK: uqadd {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
}

uint16_t test_vqaddh_u16(uint16_t a, uint16_t b) {
// CHECK-LABEL: test_vqaddh_u16
  return vqaddh_u16(a, b);
// CHECK: uqadd {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

uint32_t test_vqadds_u32(uint32_t a, uint32_t b) {
// CHECK-LABEL: test_vqadds_u32
  return vqadds_u32(a, b);
// CHECK: uqadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

uint64_t test_vqaddd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vqaddd_u64
  return vqaddd_u64(a, b);
// CHECK: uqadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8_t test_vqsubb_s8(int8_t a, int8_t b) {
// CHECK-LABEL: test_vqsubb_s8
  return vqsubb_s8(a, b);
// CHECK: sqsub {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
}

int16_t test_vqsubh_s16(int16_t a, int16_t b) {
// CHECK-LABEL: test_vqsubh_s16
  return vqsubh_s16(a, b);
// CHECK: sqsub {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

int32_t test_vqsubs_s32(int32_t a, int32_t b) {
  // CHECK-LABEL: test_vqsubs_s32
  return vqsubs_s32(a, b);
// CHECK: sqsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

int64_t test_vqsubd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vqsubd_s64
  return vqsubd_s64(a, b);
// CHECK: sqsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8_t test_vqsubb_u8(uint8_t a, uint8_t b) {
// CHECK-LABEL: test_vqsubb_u8
  return vqsubb_u8(a, b);
// CHECK: uqsub {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
}

uint16_t test_vqsubh_u16(uint16_t a, uint16_t b) {
// CHECK-LABEL: test_vqsubh_u16
  return vqsubh_u16(a, b);
// CHECK: uqsub {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

uint32_t test_vqsubs_u32(uint32_t a, uint32_t b) {
// CHECK-LABEL: test_vqsubs_u32
  return vqsubs_u32(a, b);
// CHECK: uqsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

uint64_t test_vqsubd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vqsubd_u64
  return vqsubd_u64(a, b);
// CHECK: uqsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int64_t test_vshld_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vshld_s64
  return vshld_s64(a, b);
// CHECK: sshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64_t test_vshld_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vshld_u64
  return vshld_u64(a, b);
// CHECK: ushl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK-LABEL: test_vqshlb_s8
int8_t test_vqshlb_s8(int8_t a, int8_t b) {
  return vqshlb_s8(a, b);
// CHECK: sqshl {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
}

// CHECK-LABEL: test_vqshlh_s16
int16_t test_vqshlh_s16(int16_t a, int16_t b) {
  return vqshlh_s16(a, b);
// CHECK: sqshl {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

// CHECK-LABEL: test_vqshls_s32
int32_t test_vqshls_s32(int32_t a, int32_t b) {
  return vqshls_s32(a, b);
// CHECK: sqshl {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK-LABEL: test_vqshld_s64
int64_t test_vqshld_s64(int64_t a, int64_t b) {
  return vqshld_s64(a, b);
// CHECK: sqshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK-LABEL: test_vqshlb_u8
uint8_t test_vqshlb_u8(uint8_t a, uint8_t b) {
  return vqshlb_u8(a, b);
// CHECK: uqshl {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
}

// CHECK-LABEL: test_vqshlh_u16
uint16_t test_vqshlh_u16(uint16_t a, uint16_t b) {
  return vqshlh_u16(a, b);
// CHECK: uqshl {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

// CHECK-LABEL: test_vqshls_u32
uint32_t test_vqshls_u32(uint32_t a, uint32_t b) {
  return vqshls_u32(a, b);
// CHECK: uqshl {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK-LABEL: test_vqshld_u64
uint64_t test_vqshld_u64(uint64_t a, uint64_t b) {
  return vqshld_u64(a, b);
// CHECK: uqshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK-LABEL: test_vrshld_s64
int64_t test_vrshld_s64(int64_t a, int64_t b) {
  return vrshld_s64(a, b);
// CHECK: srshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}


// CHECK-LABEL: test_vrshld_u64
uint64_t test_vrshld_u64(uint64_t a, uint64_t b) {
  return vrshld_u64(a, b);
// CHECK: urshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK-LABEL: test_vqrshlb_s8
int8_t test_vqrshlb_s8(int8_t a, int8_t b) {
  return vqrshlb_s8(a, b);
// CHECK: sqrshl {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
}

// CHECK-LABEL: test_vqrshlh_s16
int16_t test_vqrshlh_s16(int16_t a, int16_t b) {
  return vqrshlh_s16(a, b);
// CHECK: sqrshl {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

// CHECK-LABEL: test_vqrshls_s32
int32_t test_vqrshls_s32(int32_t a, int32_t b) {
  return vqrshls_s32(a, b);
// CHECK: sqrshl {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK-LABEL: test_vqrshld_s64
int64_t test_vqrshld_s64(int64_t a, int64_t b) {
  return vqrshld_s64(a, b);
// CHECK: sqrshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK-LABEL: test_vqrshlb_u8
uint8_t test_vqrshlb_u8(uint8_t a, uint8_t b) {
  return vqrshlb_u8(a, b);
// CHECK: uqrshl {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
}

// CHECK-LABEL: test_vqrshlh_u16
uint16_t test_vqrshlh_u16(uint16_t a, uint16_t b) {
  return vqrshlh_u16(a, b);
// CHECK: uqrshl {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

// CHECK-LABEL: test_vqrshls_u32
uint32_t test_vqrshls_u32(uint32_t a, uint32_t b) {
  return vqrshls_u32(a, b);
// CHECK: uqrshl {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK-LABEL: test_vqrshld_u64
uint64_t test_vqrshld_u64(uint64_t a, uint64_t b) {
  return vqrshld_u64(a, b);
// CHECK: uqrshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK-LABEL: test_vpaddd_s64
int64_t test_vpaddd_s64(int64x2_t a) {
  return vpaddd_s64(a);
// CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

// CHECK-LABEL: test_vpadds_f32
float32_t test_vpadds_f32(float32x2_t a) {
  return vpadds_f32(a);
// CHECK: faddp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

// CHECK-LABEL: test_vpaddd_f64
float64_t test_vpaddd_f64(float64x2_t a) {
  return vpaddd_f64(a);
// CHECK: faddp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

// CHECK-LABEL: test_vpmaxnms_f32
float32_t test_vpmaxnms_f32(float32x2_t a) {
  return vpmaxnms_f32(a);
// CHECK: fmaxnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

// CHECK-LABEL: test_vpmaxnmqd_f64
float64_t test_vpmaxnmqd_f64(float64x2_t a) {
  return vpmaxnmqd_f64(a);
// CHECK: fmaxnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

// CHECK-LABEL: test_vpmaxs_f32
float32_t test_vpmaxs_f32(float32x2_t a) {
  return vpmaxs_f32(a);
// CHECK: fmaxp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

// CHECK-LABEL: test_vpmaxqd_f64
float64_t test_vpmaxqd_f64(float64x2_t a) {
  return vpmaxqd_f64(a);
// CHECK: fmaxp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

// CHECK-LABEL: test_vpminnms_f32
float32_t test_vpminnms_f32(float32x2_t a) {
  return vpminnms_f32(a);
// CHECK: fminnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

// CHECK-LABEL: test_vpminnmqd_f64
float64_t test_vpminnmqd_f64(float64x2_t a) {
  return vpminnmqd_f64(a);
// CHECK: fminnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

// CHECK-LABEL: test_vpmins_f32
float32_t test_vpmins_f32(float32x2_t a) {
  return vpmins_f32(a);
// CHECK: fminp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

// CHECK-LABEL: test_vpminqd_f64
float64_t test_vpminqd_f64(float64x2_t a) {
  return vpminqd_f64(a);
// CHECK: fminp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

int16_t test_vqdmulhh_s16(int16_t a, int16_t b) {
// CHECK-LABEL: test_vqdmulhh_s16
  return vqdmulhh_s16(a, b);
// CHECK: sqdmulh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

int32_t test_vqdmulhs_s32(int32_t a, int32_t b) {
// CHECK-LABEL: test_vqdmulhs_s32
  return vqdmulhs_s32(a, b);
// CHECK: sqdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

int16_t test_vqrdmulhh_s16(int16_t a, int16_t b) {
// CHECK-LABEL: test_vqrdmulhh_s16
  return vqrdmulhh_s16(a, b);
// CHECK: sqrdmulh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
}

int32_t test_vqrdmulhs_s32(int32_t a, int32_t b) {
// CHECK-LABEL: test_vqrdmulhs_s32
  return vqrdmulhs_s32(a, b);
// CHECK: sqrdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

float32_t test_vmulxs_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vmulxs_f32
  return vmulxs_f32(a, b);
// CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

float64_t test_vmulxd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vmulxd_f64
  return vmulxd_f64(a, b);
// CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

float64x1_t test_vmulx_f64(float64x1_t a, float64x1_t b) {
// CHECK-LABEL: test_vmulx_f64
  return vmulx_f64(a, b);
// CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

float32_t test_vrecpss_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vrecpss_f32
  return vrecpss_f32(a, b);
// CHECK: frecps {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

float64_t test_vrecpsd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vrecpsd_f64
  return vrecpsd_f64(a, b);
// CHECK: frecps {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

float32_t test_vrsqrtss_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vrsqrtss_f32
  return vrsqrtss_f32(a, b);
// CHECK: frsqrts {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

float64_t test_vrsqrtsd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vrsqrtsd_f64
  return vrsqrtsd_f64(a, b);
// CHECK: frsqrts {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

float32_t test_vcvts_f32_s32(int32_t a) {
// CHECK-LABEL: test_vcvts_f32_s32
// CHECK: scvtf {{s[0-9]+}}, {{[ws][0-9]+}}
  return vcvts_f32_s32(a);
}

float64_t test_vcvtd_f64_s64(int64_t a) {
// CHECK-LABEL: test_vcvtd_f64_s64
// CHECK: scvtf {{d[0-9]+}}, {{[dx][0-9]+}}
  return vcvtd_f64_s64(a);
}

float32_t test_vcvts_f32_u32(uint32_t a) {
// CHECK-LABEL: test_vcvts_f32_u32
// CHECK: ucvtf {{s[0-9]+}}, {{[ws][0-9]+}}
  return vcvts_f32_u32(a);
}

float64_t test_vcvtd_f64_u64(uint64_t a) {
// CHECK-LABEL: test_vcvtd_f64_u64
// CHECK: ucvtf {{d[0-9]+}}, {{[xd][0-9]+}}
  return vcvtd_f64_u64(a);
}

float32_t test_vrecpes_f32(float32_t a) {
// CHECK-LABEL: test_vrecpes_f32
// CHECK: frecpe {{s[0-9]+}}, {{s[0-9]+}}
  return vrecpes_f32(a);
}
 
float64_t test_vrecped_f64(float64_t a) {
// CHECK-LABEL: test_vrecped_f64
// CHECK: frecpe {{d[0-9]+}}, {{d[0-9]+}}
  return vrecped_f64(a);
}
 
float32_t test_vrecpxs_f32(float32_t a) {
// CHECK-LABEL: test_vrecpxs_f32
// CHECK: frecpx {{s[0-9]+}}, {{s[0-9]+}}
  return vrecpxs_f32(a);
 }
 
float64_t test_vrecpxd_f64(float64_t a) {
// CHECK-LABEL: test_vrecpxd_f64
// CHECK: frecpx {{d[0-9]+}}, {{d[0-9]+}}
  return vrecpxd_f64(a);
}

uint32x2_t test_vrsqrte_u32(uint32x2_t a) {
// CHECK-LABEL: test_vrsqrte_u32
// CHECK: ursqrte {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  return vrsqrte_u32(a);
}

uint32x4_t test_vrsqrteq_u32(uint32x4_t a) {
// CHECK-LABEL: test_vrsqrteq_u32
// CHECK: ursqrte {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  return vrsqrteq_u32(a);
}

float32_t test_vrsqrtes_f32(float32_t a) {
// CHECK: vrsqrtes_f32
// CHECK: frsqrte {{s[0-9]+}}, {{s[0-9]+}}
  return vrsqrtes_f32(a);
}

float64_t test_vrsqrted_f64(float64_t a) {
// CHECK: vrsqrted_f64
// CHECK: frsqrte {{d[0-9]+}}, {{d[0-9]+}}
  return vrsqrted_f64(a);
}

uint8x16_t test_vld1q_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld1q_u8
  return vld1q_u8(a);
  // CHECK: {{ld1 { v[0-9]+.16b }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

uint16x8_t test_vld1q_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld1q_u16
  return vld1q_u16(a);
  // CHECK: {{ld1 { v[0-9]+.8h }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

uint32x4_t test_vld1q_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld1q_u32
  return vld1q_u32(a);
  // CHECK: {{ld1 { v[0-9]+.4s }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

uint64x2_t test_vld1q_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld1q_u64
  return vld1q_u64(a);
  // CHECK: {{ld1 { v[0-9]+.2d }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

int8x16_t test_vld1q_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld1q_s8
  return vld1q_s8(a);
  // CHECK: {{ld1 { v[0-9]+.16b }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

int16x8_t test_vld1q_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld1q_s16
  return vld1q_s16(a);
  // CHECK: {{ld1 { v[0-9]+.8h }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

int32x4_t test_vld1q_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld1q_s32
  return vld1q_s32(a);
  // CHECK: {{ld1 { v[0-9]+.4s }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

int64x2_t test_vld1q_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld1q_s64
  return vld1q_s64(a);
  // CHECK: {{ld1 { v[0-9]+.2d }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

float16x8_t test_vld1q_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld1q_f16
  return vld1q_f16(a);
  // CHECK: {{ld1 { v[0-9]+.8h }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

float32x4_t test_vld1q_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld1q_f32
  return vld1q_f32(a);
  // CHECK: {{ld1 { v[0-9]+.4s }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

float64x2_t test_vld1q_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld1q_f64
  return vld1q_f64(a);
  // CHECK: {{ld1 { v[0-9]+.2d }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

poly8x16_t test_vld1q_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld1q_p8
  return vld1q_p8(a);
  // CHECK: {{ld1 { v[0-9]+.16b }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

poly16x8_t test_vld1q_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld1q_p16
  return vld1q_p16(a);
  // CHECK: {{ld1 { v[0-9]+.8h }|ldr q[0-9]+}}, [{{x[0-9]+|sp}}]
}

uint8x8_t test_vld1_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld1_u8
  return vld1_u8(a);
  // CHECK: {{ld1 { v[0-9]+.8b }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

uint16x4_t test_vld1_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld1_u16
  return vld1_u16(a);
  // CHECK: {{ld1 { v[0-9]+.4h }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

uint32x2_t test_vld1_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld1_u32
  return vld1_u32(a);
  // CHECK: {{ld1 { v[0-9]+.2s }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

uint64x1_t test_vld1_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld1_u64
  return vld1_u64(a);
  // CHECK: {{ld1 { v[0-9]+.1d }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

int8x8_t test_vld1_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld1_s8
  return vld1_s8(a);
  // CHECK: {{ld1 { v[0-9]+.8b }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

int16x4_t test_vld1_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld1_s16
  return vld1_s16(a);
  // CHECK: {{ld1 { v[0-9]+.4h }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

int32x2_t test_vld1_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld1_s32
  return vld1_s32(a);
  // CHECK: {{ld1 { v[0-9]+.2s }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

int64x1_t test_vld1_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld1_s64
  return vld1_s64(a);
  // CHECK: {{ld1 { v[0-9]+.1d }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

float16x4_t test_vld1_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld1_f16
  return vld1_f16(a);
  // CHECK: {{ld1 { v[0-9]+.4h }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

float32x2_t test_vld1_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld1_f32
  return vld1_f32(a);
  // CHECK: {{ld1 { v[0-9]+.2s }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

float64x1_t test_vld1_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld1_f64
  return vld1_f64(a);
  // CHECK: {{ld1 { v[0-9]+.1d }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

poly8x8_t test_vld1_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld1_p8
  return vld1_p8(a);
  // CHECK: {{ld1 { v[0-9]+.8b }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

poly16x4_t test_vld1_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld1_p16
  return vld1_p16(a);
  // CHECK: {{ld1 { v[0-9]+.4h }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}

uint8x16x2_t test_vld2q_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld2q_u8
  return vld2q_u8(a);
  // CHECK: ld2 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x8x2_t test_vld2q_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld2q_u16
  return vld2q_u16(a);
  // CHECK: ld2 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x4x2_t test_vld2q_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld2q_u32
  return vld2q_u32(a);
  // CHECK: ld2 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x2x2_t test_vld2q_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld2q_u64
  return vld2q_u64(a);
  // CHECK: ld2 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

int8x16x2_t test_vld2q_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld2q_s8
  return vld2q_s8(a);
  // CHECK: ld2 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

int16x8x2_t test_vld2q_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld2q_s16
  return vld2q_s16(a);
  // CHECK: ld2 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

int32x4x2_t test_vld2q_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld2q_s32
  return vld2q_s32(a);
  // CHECK: ld2 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

int64x2x2_t test_vld2q_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld2q_s64
  return vld2q_s64(a);
  // CHECK: ld2 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

float16x8x2_t test_vld2q_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld2q_f16
  return vld2q_f16(a);
  // CHECK: ld2 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

float32x4x2_t test_vld2q_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld2q_f32
  return vld2q_f32(a);
  // CHECK: ld2 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

float64x2x2_t test_vld2q_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld2q_f64
  return vld2q_f64(a);
  // CHECK: ld2 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x16x2_t test_vld2q_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld2q_p8
  return vld2q_p8(a);
  // CHECK: ld2 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x8x2_t test_vld2q_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld2q_p16
  return vld2q_p16(a);
  // CHECK: ld2 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint8x8x2_t test_vld2_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld2_u8
  return vld2_u8(a);
  // CHECK: ld2 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x4x2_t test_vld2_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld2_u16
  return vld2_u16(a);
  // CHECK: ld2 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x2x2_t test_vld2_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld2_u32
  return vld2_u32(a);
  // CHECK: ld2 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x1x2_t test_vld2_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld2_u64
  return vld2_u64(a);
  // CHECK: {{ld1|ld2}} {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

int8x8x2_t test_vld2_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld2_s8
  return vld2_s8(a);
  // CHECK: ld2 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

int16x4x2_t test_vld2_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld2_s16
  return vld2_s16(a);
  // CHECK: ld2 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

int32x2x2_t test_vld2_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld2_s32
  return vld2_s32(a);
  // CHECK: ld2 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

int64x1x2_t test_vld2_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld2_s64
  return vld2_s64(a);
  // CHECK: {{ld1|ld2}} {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

float16x4x2_t test_vld2_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld2_f16
  return vld2_f16(a);
  // CHECK: ld2 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

float32x2x2_t test_vld2_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld2_f32
  return vld2_f32(a);
  // CHECK: ld2 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

float64x1x2_t test_vld2_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld2_f64
  return vld2_f64(a);
  // CHECK: {{ld1|ld2}} {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x8x2_t test_vld2_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld2_p8
  return vld2_p8(a);
  // CHECK: ld2 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x4x2_t test_vld2_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld2_p16
  return vld2_p16(a);
  // CHECK: ld2 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint8x16x3_t test_vld3q_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld3q_u8
  return vld3q_u8(a);
  // CHECK: ld3 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x8x3_t test_vld3q_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld3q_u16
  return vld3q_u16(a);
  // CHECK: ld3 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x4x3_t test_vld3q_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld3q_u32
  return vld3q_u32(a);
  // CHECK: ld3 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x2x3_t test_vld3q_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld3q_u64
  return vld3q_u64(a);
  // CHECK: ld3 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

int8x16x3_t test_vld3q_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld3q_s8
  return vld3q_s8(a);
  // CHECK: ld3 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

int16x8x3_t test_vld3q_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld3q_s16
  return vld3q_s16(a);
  // CHECK: ld3 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

int32x4x3_t test_vld3q_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld3q_s32
  return vld3q_s32(a);
  // CHECK: ld3 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

int64x2x3_t test_vld3q_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld3q_s64
  return vld3q_s64(a);
  // CHECK: ld3 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

float16x8x3_t test_vld3q_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld3q_f16
  return vld3q_f16(a);
  // CHECK: ld3 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

float32x4x3_t test_vld3q_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld3q_f32
  return vld3q_f32(a);
  // CHECK: ld3 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

float64x2x3_t test_vld3q_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld3q_f64
  return vld3q_f64(a);
  // CHECK: ld3 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x16x3_t test_vld3q_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld3q_p8
  return vld3q_p8(a);
  // CHECK: ld3 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x8x3_t test_vld3q_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld3q_p16
  return vld3q_p16(a);
  // CHECK: ld3 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint8x8x3_t test_vld3_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld3_u8
  return vld3_u8(a);
  // CHECK: ld3 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x4x3_t test_vld3_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld3_u16
  return vld3_u16(a);
  // CHECK: ld3 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x2x3_t test_vld3_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld3_u32
  return vld3_u32(a);
  // CHECK: ld3 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x1x3_t test_vld3_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld3_u64
  return vld3_u64(a);
  // CHECK: {{ld1|ld3}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

int8x8x3_t test_vld3_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld3_s8
  return vld3_s8(a);
  // CHECK: ld3 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

int16x4x3_t test_vld3_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld3_s16
  return vld3_s16(a);
  // CHECK: ld3 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

int32x2x3_t test_vld3_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld3_s32
  return vld3_s32(a);
  // CHECK: ld3 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

int64x1x3_t test_vld3_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld3_s64
  return vld3_s64(a);
  // CHECK: {{ld1|ld3}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

float16x4x3_t test_vld3_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld3_f16
  return vld3_f16(a);
  // CHECK: ld3 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

float32x2x3_t test_vld3_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld3_f32
  return vld3_f32(a);
  // CHECK: ld3 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

float64x1x3_t test_vld3_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld3_f64
  return vld3_f64(a);
  // CHECK: {{ld1|ld3}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x8x3_t test_vld3_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld3_p8
  return vld3_p8(a);
  // CHECK: ld3 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x4x3_t test_vld3_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld3_p16
  return vld3_p16(a);
  // CHECK: ld3 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint8x16x4_t test_vld4q_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld4q_u8
  return vld4q_u8(a);
  // CHECK: ld4 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x8x4_t test_vld4q_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld4q_u16
  return vld4q_u16(a);
  // CHECK: ld4 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x4x4_t test_vld4q_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld4q_u32
  return vld4q_u32(a);
  // CHECK: ld4 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x2x4_t test_vld4q_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld4q_u64
  return vld4q_u64(a);
  // CHECK: ld4 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

int8x16x4_t test_vld4q_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld4q_s8
  return vld4q_s8(a);
  // CHECK: ld4 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

int16x8x4_t test_vld4q_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld4q_s16
  return vld4q_s16(a);
  // CHECK: ld4 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

int32x4x4_t test_vld4q_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld4q_s32
  return vld4q_s32(a);
  // CHECK: ld4 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

int64x2x4_t test_vld4q_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld4q_s64
  return vld4q_s64(a);
  // CHECK: ld4 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

float16x8x4_t test_vld4q_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld4q_f16
  return vld4q_f16(a);
  // CHECK: ld4 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

float32x4x4_t test_vld4q_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld4q_f32
  return vld4q_f32(a);
  // CHECK: ld4 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

float64x2x4_t test_vld4q_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld4q_f64
  return vld4q_f64(a);
  // CHECK: ld4 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x16x4_t test_vld4q_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld4q_p8
  return vld4q_p8(a);
  // CHECK: ld4 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x8x4_t test_vld4q_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld4q_p16
  return vld4q_p16(a);
  // CHECK: ld4 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint8x8x4_t test_vld4_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld4_u8
  return vld4_u8(a);
  // CHECK: ld4 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x4x4_t test_vld4_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld4_u16
  return vld4_u16(a);
  // CHECK: ld4 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x2x4_t test_vld4_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld4_u32
  return vld4_u32(a);
  // CHECK: ld4 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x1x4_t test_vld4_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld4_u64
  return vld4_u64(a);
  // CHECK: {{ld1|ld4}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

int8x8x4_t test_vld4_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld4_s8
  return vld4_s8(a);
  // CHECK: ld4 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

int16x4x4_t test_vld4_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld4_s16
  return vld4_s16(a);
  // CHECK: ld4 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

int32x2x4_t test_vld4_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld4_s32
  return vld4_s32(a);
  // CHECK: ld4 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

int64x1x4_t test_vld4_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld4_s64
  return vld4_s64(a);
  // CHECK: {{ld1|ld4}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

float16x4x4_t test_vld4_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld4_f16
  return vld4_f16(a);
  // CHECK: ld4 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

float32x2x4_t test_vld4_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld4_f32
  return vld4_f32(a);
  // CHECK: ld4 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

float64x1x4_t test_vld4_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld4_f64
  return vld4_f64(a);
  // CHECK: {{ld1|ld4}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x8x4_t test_vld4_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld4_p8
  return vld4_p8(a);
  // CHECK: ld4 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x4x4_t test_vld4_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld4_p16
  return vld4_p16(a);
  // CHECK: ld4 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u8(uint8_t *a, uint8x16_t b) {
  // CHECK-LABEL: test_vst1q_u8
  vst1q_u8(a, b);
  // CHECK: {{st1 { v[0-9]+.16b }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u16(uint16_t *a, uint16x8_t b) {
  // CHECK-LABEL: test_vst1q_u16
  vst1q_u16(a, b);
  // CHECK: {{st1 { v[0-9]+.8h }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u32(uint32_t *a, uint32x4_t b) {
  // CHECK-LABEL: test_vst1q_u32
  vst1q_u32(a, b);
  // CHECK: {{st1 { v[0-9]+.4s }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u64(uint64_t *a, uint64x2_t b) {
  // CHECK-LABEL: test_vst1q_u64
  vst1q_u64(a, b);
  // CHECK: {{st1 { v[0-9]+.2d }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s8(int8_t *a, int8x16_t b) {
  // CHECK-LABEL: test_vst1q_s8
  vst1q_s8(a, b);
  // CHECK: {{st1 { v[0-9]+.16b }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s16(int16_t *a, int16x8_t b) {
  // CHECK-LABEL: test_vst1q_s16
  vst1q_s16(a, b);
  // CHECK: {{st1 { v[0-9]+.8h }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s32(int32_t *a, int32x4_t b) {
  // CHECK-LABEL: test_vst1q_s32
  vst1q_s32(a, b);
  // CHECK: {{st1 { v[0-9]+.4s }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s64(int64_t *a, int64x2_t b) {
  // CHECK-LABEL: test_vst1q_s64
  vst1q_s64(a, b);
  // CHECK: {{st1 { v[0-9]+.2d }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f16(float16_t *a, float16x8_t b) {
  // CHECK-LABEL: test_vst1q_f16
  vst1q_f16(a, b);
  // CHECK: {{st1 { v[0-9]+.8h }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f32(float32_t *a, float32x4_t b) {
  // CHECK-LABEL: test_vst1q_f32
  vst1q_f32(a, b);
  // CHECK: {{st1 { v[0-9]+.4s }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f64(float64_t *a, float64x2_t b) {
  // CHECK-LABEL: test_vst1q_f64
  vst1q_f64(a, b);
  // CHECK: {{st1 { v[0-9]+.2d }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p8(poly8_t *a, poly8x16_t b) {
  // CHECK-LABEL: test_vst1q_p8
  vst1q_p8(a, b);
  // CHECK: {{st1 { v[0-9]+.16b }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p16(poly16_t *a, poly16x8_t b) {
  // CHECK-LABEL: test_vst1q_p16
  vst1q_p16(a, b);
  // CHECK: {{st1 { v[0-9]+.8h }|str q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u8(uint8_t *a, uint8x8_t b) {
  // CHECK-LABEL: test_vst1_u8
  vst1_u8(a, b);
  // CHECK: {{st1 { v[0-9]+.8b }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u16(uint16_t *a, uint16x4_t b) {
  // CHECK-LABEL: test_vst1_u16
  vst1_u16(a, b);
  // CHECK: {{st1 { v[0-9]+.4h }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u32(uint32_t *a, uint32x2_t b) {
  // CHECK-LABEL: test_vst1_u32
  vst1_u32(a, b);
  // CHECK: {{st1 { v[0-9]+.2s }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u64(uint64_t *a, uint64x1_t b) {
  // CHECK-LABEL: test_vst1_u64
  vst1_u64(a, b);
  // CHECK: {{st1 { v[0-9]+.1d }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s8(int8_t *a, int8x8_t b) {
  // CHECK-LABEL: test_vst1_s8
  vst1_s8(a, b);
  // CHECK: {{st1 { v[0-9]+.8b }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s16(int16_t *a, int16x4_t b) {
  // CHECK-LABEL: test_vst1_s16
  vst1_s16(a, b);
  // CHECK: {{st1 { v[0-9]+.4h }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s32(int32_t *a, int32x2_t b) {
  // CHECK-LABEL: test_vst1_s32
  vst1_s32(a, b);
  // CHECK: {{st1 { v[0-9]+.2s }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s64(int64_t *a, int64x1_t b) {
  // CHECK-LABEL: test_vst1_s64
  vst1_s64(a, b);
  // CHECK: {{st1 { v[0-9]+.1d }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f16(float16_t *a, float16x4_t b) {
  // CHECK-LABEL: test_vst1_f16
  vst1_f16(a, b);
  // CHECK: {{st1 { v[0-9]+.4h }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f32(float32_t *a, float32x2_t b) {
  // CHECK-LABEL: test_vst1_f32
  vst1_f32(a, b);
  // CHECK: {{st1 { v[0-9]+.2s }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f64(float64_t *a, float64x1_t b) {
  // CHECK-LABEL: test_vst1_f64
  vst1_f64(a, b);
  // CHECK: {{st1 { v[0-9]+.1d }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p8(poly8_t *a, poly8x8_t b) {
  // CHECK-LABEL: test_vst1_p8
  vst1_p8(a, b);
  // CHECK: {{st1 { v[0-9]+.8b }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p16(poly16_t *a, poly16x4_t b) {
  // CHECK-LABEL: test_vst1_p16
  vst1_p16(a, b);
  // CHECK: {{st1 { v[0-9]+.4h }|str d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_u8(uint8_t *a, uint8x16x2_t b) {
  // CHECK-LABEL: test_vst2q_u8
  vst2q_u8(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_u16(uint16_t *a, uint16x8x2_t b) {
  // CHECK-LABEL: test_vst2q_u16
  vst2q_u16(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_u32(uint32_t *a, uint32x4x2_t b) {
  // CHECK-LABEL: test_vst2q_u32
  vst2q_u32(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_u64(uint64_t *a, uint64x2x2_t b) {
  // CHECK-LABEL: test_vst2q_u64
  vst2q_u64(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_s8(int8_t *a, int8x16x2_t b) {
  // CHECK-LABEL: test_vst2q_s8
  vst2q_s8(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_s16(int16_t *a, int16x8x2_t b) {
  // CHECK-LABEL: test_vst2q_s16
  vst2q_s16(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_s32(int32_t *a, int32x4x2_t b) {
  // CHECK-LABEL: test_vst2q_s32
  vst2q_s32(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_s64(int64_t *a, int64x2x2_t b) {
  // CHECK-LABEL: test_vst2q_s64
  vst2q_s64(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_f16(float16_t *a, float16x8x2_t b) {
  // CHECK-LABEL: test_vst2q_f16
  vst2q_f16(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_f32(float32_t *a, float32x4x2_t b) {
  // CHECK-LABEL: test_vst2q_f32
  vst2q_f32(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_f64(float64_t *a, float64x2x2_t b) {
  // CHECK-LABEL: test_vst2q_f64
  vst2q_f64(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_p8(poly8_t *a, poly8x16x2_t b) {
  // CHECK-LABEL: test_vst2q_p8
  vst2q_p8(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_p16(poly16_t *a, poly16x8x2_t b) {
  // CHECK-LABEL: test_vst2q_p16
  vst2q_p16(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_u8(uint8_t *a, uint8x8x2_t b) {
  // CHECK-LABEL: test_vst2_u8
  vst2_u8(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_u16(uint16_t *a, uint16x4x2_t b) {
  // CHECK-LABEL: test_vst2_u16
  vst2_u16(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_u32(uint32_t *a, uint32x2x2_t b) {
  // CHECK-LABEL: test_vst2_u32
  vst2_u32(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_u64(uint64_t *a, uint64x1x2_t b) {
  // CHECK-LABEL: test_vst2_u64
  vst2_u64(a, b);
  // CHECK: {{st1|st2}} {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_s8(int8_t *a, int8x8x2_t b) {
  // CHECK-LABEL: test_vst2_s8
  vst2_s8(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_s16(int16_t *a, int16x4x2_t b) {
  // CHECK-LABEL: test_vst2_s16
  vst2_s16(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_s32(int32_t *a, int32x2x2_t b) {
  // CHECK-LABEL: test_vst2_s32
  vst2_s32(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_s64(int64_t *a, int64x1x2_t b) {
  // CHECK-LABEL: test_vst2_s64
  vst2_s64(a, b);
  // CHECK: {{st1|st2}} {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_f16(float16_t *a, float16x4x2_t b) {
  // CHECK-LABEL: test_vst2_f16
  vst2_f16(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_f32(float32_t *a, float32x2x2_t b) {
  // CHECK-LABEL: test_vst2_f32
  vst2_f32(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_f64(float64_t *a, float64x1x2_t b) {
  // CHECK-LABEL: test_vst2_f64
  vst2_f64(a, b);
  // CHECK: {{st1|st2}} {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_p8(poly8_t *a, poly8x8x2_t b) {
  // CHECK-LABEL: test_vst2_p8
  vst2_p8(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_p16(poly16_t *a, poly16x4x2_t b) {
  // CHECK-LABEL: test_vst2_p16
  vst2_p16(a, b);
  // CHECK: st2 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_u8(uint8_t *a, uint8x16x3_t b) {
  // CHECK-LABEL: test_vst3q_u8
  vst3q_u8(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_u16(uint16_t *a, uint16x8x3_t b) {
  // CHECK-LABEL: test_vst3q_u16
  vst3q_u16(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_u32(uint32_t *a, uint32x4x3_t b) {
  // CHECK-LABEL: test_vst3q_u32
  vst3q_u32(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_u64(uint64_t *a, uint64x2x3_t b) {
  // CHECK-LABEL: test_vst3q_u64
  vst3q_u64(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_s8(int8_t *a, int8x16x3_t b) {
  // CHECK-LABEL: test_vst3q_s8
  vst3q_s8(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_s16(int16_t *a, int16x8x3_t b) {
  // CHECK-LABEL: test_vst3q_s16
  vst3q_s16(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_s32(int32_t *a, int32x4x3_t b) {
  // CHECK-LABEL: test_vst3q_s32
  vst3q_s32(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_s64(int64_t *a, int64x2x3_t b) {
  // CHECK-LABEL: test_vst3q_s64
  vst3q_s64(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_f16(float16_t *a, float16x8x3_t b) {
  // CHECK-LABEL: test_vst3q_f16
  vst3q_f16(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_f32(float32_t *a, float32x4x3_t b) {
  // CHECK-LABEL: test_vst3q_f32
  vst3q_f32(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_f64(float64_t *a, float64x2x3_t b) {
  // CHECK-LABEL: test_vst3q_f64
  vst3q_f64(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_p8(poly8_t *a, poly8x16x3_t b) {
  // CHECK-LABEL: test_vst3q_p8
  vst3q_p8(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_p16(poly16_t *a, poly16x8x3_t b) {
  // CHECK-LABEL: test_vst3q_p16
  vst3q_p16(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_u8(uint8_t *a, uint8x8x3_t b) {
  // CHECK-LABEL: test_vst3_u8
  vst3_u8(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_u16(uint16_t *a, uint16x4x3_t b) {
  // CHECK-LABEL: test_vst3_u16
  vst3_u16(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_u32(uint32_t *a, uint32x2x3_t b) {
  // CHECK-LABEL: test_vst3_u32
  vst3_u32(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_u64(uint64_t *a, uint64x1x3_t b) {
  // CHECK-LABEL: test_vst3_u64
  vst3_u64(a, b);
  // CHECK: {{st1|st3}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_s8(int8_t *a, int8x8x3_t b) {
  // CHECK-LABEL: test_vst3_s8
  vst3_s8(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_s16(int16_t *a, int16x4x3_t b) {
  // CHECK-LABEL: test_vst3_s16
  vst3_s16(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_s32(int32_t *a, int32x2x3_t b) {
  // CHECK-LABEL: test_vst3_s32
  vst3_s32(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_s64(int64_t *a, int64x1x3_t b) {
  // CHECK-LABEL: test_vst3_s64
  vst3_s64(a, b);
  // CHECK: {{st1|st3}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_f16(float16_t *a, float16x4x3_t b) {
  // CHECK-LABEL: test_vst3_f16
  vst3_f16(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_f32(float32_t *a, float32x2x3_t b) {
  // CHECK-LABEL: test_vst3_f32
  vst3_f32(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_f64(float64_t *a, float64x1x3_t b) {
  // CHECK-LABEL: test_vst3_f64
  vst3_f64(a, b);
  // CHECK: {{st1|st3}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_p8(poly8_t *a, poly8x8x3_t b) {
  // CHECK-LABEL: test_vst3_p8
  vst3_p8(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_p16(poly16_t *a, poly16x4x3_t b) {
  // CHECK-LABEL: test_vst3_p16
  vst3_p16(a, b);
  // CHECK: st3 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_u8(uint8_t *a, uint8x16x4_t b) {
  // CHECK-LABEL: test_vst4q_u8
  vst4q_u8(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_u16(uint16_t *a, uint16x8x4_t b) {
  // CHECK-LABEL: test_vst4q_u16
  vst4q_u16(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_u32(uint32_t *a, uint32x4x4_t b) {
  // CHECK-LABEL: test_vst4q_u32
  vst4q_u32(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_u64(uint64_t *a, uint64x2x4_t b) {
  // CHECK-LABEL: test_vst4q_u64
  vst4q_u64(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_s8(int8_t *a, int8x16x4_t b) {
  // CHECK-LABEL: test_vst4q_s8
  vst4q_s8(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_s16(int16_t *a, int16x8x4_t b) {
  // CHECK-LABEL: test_vst4q_s16
  vst4q_s16(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_s32(int32_t *a, int32x4x4_t b) {
  // CHECK-LABEL: test_vst4q_s32
  vst4q_s32(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_s64(int64_t *a, int64x2x4_t b) {
  // CHECK-LABEL: test_vst4q_s64
  vst4q_s64(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_f16(float16_t *a, float16x8x4_t b) {
  // CHECK-LABEL: test_vst4q_f16
  vst4q_f16(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_f32(float32_t *a, float32x4x4_t b) {
  // CHECK-LABEL: test_vst4q_f32
  vst4q_f32(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_f64(float64_t *a, float64x2x4_t b) {
  // CHECK-LABEL: test_vst4q_f64
  vst4q_f64(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_p8(poly8_t *a, poly8x16x4_t b) {
  // CHECK-LABEL: test_vst4q_p8
  vst4q_p8(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_p16(poly16_t *a, poly16x8x4_t b) {
  // CHECK-LABEL: test_vst4q_p16
  vst4q_p16(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_u8(uint8_t *a, uint8x8x4_t b) {
  // CHECK-LABEL: test_vst4_u8
  vst4_u8(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_u16(uint16_t *a, uint16x4x4_t b) {
  // CHECK-LABEL: test_vst4_u16
  vst4_u16(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_u32(uint32_t *a, uint32x2x4_t b) {
  // CHECK-LABEL: test_vst4_u32
  vst4_u32(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_u64(uint64_t *a, uint64x1x4_t b) {
  // CHECK-LABEL: test_vst4_u64
  vst4_u64(a, b);
  // CHECK: {{st1|st4}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_s8(int8_t *a, int8x8x4_t b) {
  // CHECK-LABEL: test_vst4_s8
  vst4_s8(a, b);
// CHECK: st4 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_s16(int16_t *a, int16x4x4_t b) {
  // CHECK-LABEL: test_vst4_s16
  vst4_s16(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_s32(int32_t *a, int32x2x4_t b) {
  // CHECK-LABEL: test_vst4_s32
  vst4_s32(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_s64(int64_t *a, int64x1x4_t b) {
  // CHECK-LABEL: test_vst4_s64
  vst4_s64(a, b);
  // CHECK: {{st1|st4}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_f16(float16_t *a, float16x4x4_t b) {
  // CHECK-LABEL: test_vst4_f16
  vst4_f16(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_f32(float32_t *a, float32x2x4_t b) {
  // CHECK-LABEL: test_vst4_f32
  vst4_f32(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_f64(float64_t *a, float64x1x4_t b) {
  // CHECK-LABEL: test_vst4_f64
  vst4_f64(a, b);
  // CHECK: {{st1|st4}} {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_p8(poly8_t *a, poly8x8x4_t b) {
  // CHECK-LABEL: test_vst4_p8
  vst4_p8(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_p16(poly16_t *a, poly16x4x4_t b) {
  // CHECK-LABEL: test_vst4_p16
  vst4_p16(a, b);
  // CHECK: st4 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint8x16x2_t test_vld1q_u8_x2(uint8_t const *a) {
  // CHECK-LABEL: test_vld1q_u8_x2
  return vld1q_u8_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x8x2_t test_vld1q_u16_x2(uint16_t const *a) {
  // CHECK-LABEL: test_vld1q_u16_x2
  return vld1q_u16_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x4x2_t test_vld1q_u32_x2(uint32_t const *a) {
  // CHECK-LABEL: test_vld1q_u32_x2
  return vld1q_u32_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x2x2_t test_vld1q_u64_x2(uint64_t const *a) {
  // CHECK-LABEL: test_vld1q_u64_x2
  return vld1q_u64_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

int8x16x2_t test_vld1q_s8_x2(int8_t const *a) {
  // CHECK-LABEL: test_vld1q_s8_x2
  return vld1q_s8_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

int16x8x2_t test_vld1q_s16_x2(int16_t const *a) {
  // CHECK-LABEL: test_vld1q_s16_x2
  return vld1q_s16_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

int32x4x2_t test_vld1q_s32_x2(int32_t const *a) {
  // CHECK-LABEL: test_vld1q_s32_x2
  return vld1q_s32_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

int64x2x2_t test_vld1q_s64_x2(int64_t const *a) {
  // CHECK-LABEL: test_vld1q_s64_x2
  return vld1q_s64_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

float16x8x2_t test_vld1q_f16_x2(float16_t const *a) {
  // CHECK-LABEL: test_vld1q_f16_x2
  return vld1q_f16_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

float32x4x2_t test_vld1q_f32_x2(float32_t const *a) {
  // CHECK-LABEL: test_vld1q_f32_x2
  return vld1q_f32_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

float64x2x2_t test_vld1q_f64_x2(float64_t const *a) {
  // CHECK-LABEL: test_vld1q_f64_x2
  return vld1q_f64_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x16x2_t test_vld1q_p8_x2(poly8_t const *a) {
  // CHECK-LABEL: test_vld1q_p8_x2
  return vld1q_p8_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x8x2_t test_vld1q_p16_x2(poly16_t const *a) {
  // CHECK-LABEL: test_vld1q_p16_x2
  return vld1q_p16_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

poly64x2x2_t test_vld1q_p64_x2(poly64_t const *a) {
  // CHECK-LABEL: test_vld1q_p64_x2
  return vld1q_p64_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

uint8x8x2_t test_vld1_u8_x2(uint8_t const *a) {
  // CHECK-LABEL: test_vld1_u8_x2
  return vld1_u8_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x4x2_t test_vld1_u16_x2(uint16_t const *a) {
  // CHECK-LABEL: test_vld1_u16_x2
  return vld1_u16_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x2x2_t test_vld1_u32_x2(uint32_t const *a) {
  // CHECK-LABEL: test_vld1_u32_x2
  return vld1_u32_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x1x2_t test_vld1_u64_x2(uint64_t const *a) {
  // CHECK-LABEL: test_vld1_u64_x2
  return vld1_u64_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

int8x8x2_t test_vld1_s8_x2(int8_t const *a) {
  // CHECK-LABEL: test_vld1_s8_x2
  return vld1_s8_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

int16x4x2_t test_vld1_s16_x2(int16_t const *a) {
  // CHECK-LABEL: test_vld1_s16_x2
  return vld1_s16_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

int32x2x2_t test_vld1_s32_x2(int32_t const *a) {
  // CHECK-LABEL: test_vld1_s32_x2
  return vld1_s32_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

int64x1x2_t test_vld1_s64_x2(int64_t const *a) {
  // CHECK-LABEL: test_vld1_s64_x2
  return vld1_s64_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

float16x4x2_t test_vld1_f16_x2(float16_t const *a) {
  // CHECK-LABEL: test_vld1_f16_x2
  return vld1_f16_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

float32x2x2_t test_vld1_f32_x2(float32_t const *a) {
  // CHECK-LABEL: test_vld1_f32_x2
  return vld1_f32_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

float64x1x2_t test_vld1_f64_x2(float64_t const *a) {
  // CHECK-LABEL: test_vld1_f64_x2
  return vld1_f64_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x8x2_t test_vld1_p8_x2(poly8_t const *a) {
  // CHECK-LABEL: test_vld1_p8_x2
  return vld1_p8_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x4x2_t test_vld1_p16_x2(poly16_t const *a) {
  // CHECK-LABEL: test_vld1_p16_x2
  return vld1_p16_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

poly64x1x2_t test_vld1_p64_x2(poly64_t const *a) {
  // CHECK-LABEL: test_vld1_p64_x2
  return vld1_p64_x2(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

uint8x16x3_t test_vld1q_u8_x3(uint8_t const *a) {
  // CHECK-LABEL: test_vld1q_u8_x3
  return vld1q_u8_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x8x3_t test_vld1q_u16_x3(uint16_t const *a) {
  // CHECK-LABEL: test_vld1q_u16_x3
  return vld1q_u16_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x4x3_t test_vld1q_u32_x3(uint32_t const *a) {
  // CHECK-LABEL: test_vld1q_u32_x3
  return vld1q_u32_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x2x3_t test_vld1q_u64_x3(uint64_t const *a) {
  // CHECK-LABEL: test_vld1q_u64_x3
  return vld1q_u64_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

int8x16x3_t test_vld1q_s8_x3(int8_t const *a) {
  // CHECK-LABEL: test_vld1q_s8_x3
  return vld1q_s8_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

int16x8x3_t test_vld1q_s16_x3(int16_t const *a) {
  // CHECK-LABEL: test_vld1q_s16_x3
  return vld1q_s16_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

int32x4x3_t test_vld1q_s32_x3(int32_t const *a) {
  // CHECK-LABEL: test_vld1q_s32_x3
  return vld1q_s32_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

int64x2x3_t test_vld1q_s64_x3(int64_t const *a) {
  // CHECK-LABEL: test_vld1q_s64_x3
  return vld1q_s64_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

float16x8x3_t test_vld1q_f16_x3(float16_t const *a) {
  // CHECK-LABEL: test_vld1q_f16_x3
  return vld1q_f16_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

float32x4x3_t test_vld1q_f32_x3(float32_t const *a) {
  // CHECK-LABEL: test_vld1q_f32_x3
  return vld1q_f32_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

float64x2x3_t test_vld1q_f64_x3(float64_t const *a) {
  // CHECK-LABEL: test_vld1q_f64_x3
  return vld1q_f64_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x16x3_t test_vld1q_p8_x3(poly8_t const *a) {
  // CHECK-LABEL: test_vld1q_p8_x3
  return vld1q_p8_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x8x3_t test_vld1q_p16_x3(poly16_t const *a) {
  // CHECK-LABEL: test_vld1q_p16_x3
  return vld1q_p16_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

poly64x2x3_t test_vld1q_p64_x3(poly64_t const *a) {
  // CHECK-LABEL: test_vld1q_p64_x3
  return vld1q_p64_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

uint8x8x3_t test_vld1_u8_x3(uint8_t const *a) {
  // CHECK-LABEL: test_vld1_u8_x3
  return vld1_u8_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x4x3_t test_vld1_u16_x3(uint16_t const *a) {
  // CHECK-LABEL: test_vld1_u16_x3
  return vld1_u16_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x2x3_t test_vld1_u32_x3(uint32_t const *a) {
  // CHECK-LABEL: test_vld1_u32_x3
  return vld1_u32_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x1x3_t test_vld1_u64_x3(uint64_t const *a) {
  // CHECK-LABEL: test_vld1_u64_x3
  return vld1_u64_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

int8x8x3_t test_vld1_s8_x3(int8_t const *a) {
  // CHECK-LABEL: test_vld1_s8_x3
  return vld1_s8_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

int16x4x3_t test_vld1_s16_x3(int16_t const *a) {
  // CHECK-LABEL: test_vld1_s16_x3
  return vld1_s16_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

int32x2x3_t test_vld1_s32_x3(int32_t const *a) {
  // CHECK-LABEL: test_vld1_s32_x3
  return vld1_s32_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

int64x1x3_t test_vld1_s64_x3(int64_t const *a) {
  // CHECK-LABEL: test_vld1_s64_x3
  return vld1_s64_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

float16x4x3_t test_vld1_f16_x3(float16_t const *a) {
  // CHECK-LABEL: test_vld1_f16_x3
  return vld1_f16_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

float32x2x3_t test_vld1_f32_x3(float32_t const *a) {
  // CHECK-LABEL: test_vld1_f32_x3
  return vld1_f32_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

float64x1x3_t test_vld1_f64_x3(float64_t const *a) {
  // CHECK-LABEL: test_vld1_f64_x3
  return vld1_f64_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x8x3_t test_vld1_p8_x3(poly8_t const *a) {
  // CHECK-LABEL: test_vld1_p8_x3
  return vld1_p8_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x4x3_t test_vld1_p16_x3(poly16_t const *a) {
  // CHECK-LABEL: test_vld1_p16_x3
  return vld1_p16_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

poly64x1x3_t test_vld1_p64_x3(poly64_t const *a) {
  // CHECK-LABEL: test_vld1_p64_x3
  return vld1_p64_x3(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

uint8x16x4_t test_vld1q_u8_x4(uint8_t const *a) {
  // CHECK-LABEL: test_vld1q_u8_x4
  return vld1q_u8_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x8x4_t test_vld1q_u16_x4(uint16_t const *a) {
  // CHECK-LABEL: test_vld1q_u16_x4
  return vld1q_u16_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x4x4_t test_vld1q_u32_x4(uint32_t const *a) {
  // CHECK-LABEL: test_vld1q_u32_x4
  return vld1q_u32_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x2x4_t test_vld1q_u64_x4(uint64_t const *a) {
  // CHECK-LABEL: test_vld1q_u64_x4
  return vld1q_u64_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

int8x16x4_t test_vld1q_s8_x4(int8_t const *a) {
  // CHECK-LABEL: test_vld1q_s8_x4
  return vld1q_s8_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

int16x8x4_t test_vld1q_s16_x4(int16_t const *a) {
  // CHECK-LABEL: test_vld1q_s16_x4
  return vld1q_s16_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

int32x4x4_t test_vld1q_s32_x4(int32_t const *a) {
  // CHECK-LABEL: test_vld1q_s32_x4
  return vld1q_s32_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

int64x2x4_t test_vld1q_s64_x4(int64_t const *a) {
  // CHECK-LABEL: test_vld1q_s64_x4
  return vld1q_s64_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

float16x8x4_t test_vld1q_f16_x4(float16_t const *a) {
  // CHECK-LABEL: test_vld1q_f16_x4
  return vld1q_f16_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

float32x4x4_t test_vld1q_f32_x4(float32_t const *a) {
  // CHECK-LABEL: test_vld1q_f32_x4
  return vld1q_f32_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

float64x2x4_t test_vld1q_f64_x4(float64_t const *a) {
  // CHECK-LABEL: test_vld1q_f64_x4
  return vld1q_f64_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x16x4_t test_vld1q_p8_x4(poly8_t const *a) {
  // CHECK-LABEL: test_vld1q_p8_x4
  return vld1q_p8_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x8x4_t test_vld1q_p16_x4(poly16_t const *a) {
  // CHECK-LABEL: test_vld1q_p16_x4
  return vld1q_p16_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

poly64x2x4_t test_vld1q_p64_x4(poly64_t const *a) {
  // CHECK-LABEL: test_vld1q_p64_x4
  return vld1q_p64_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

uint8x8x4_t test_vld1_u8_x4(uint8_t const *a) {
  // CHECK-LABEL: test_vld1_u8_x4
  return vld1_u8_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

uint16x4x4_t test_vld1_u16_x4(uint16_t const *a) {
  // CHECK-LABEL: test_vld1_u16_x4
  return vld1_u16_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

uint32x2x4_t test_vld1_u32_x4(uint32_t const *a) {
  // CHECK-LABEL: test_vld1_u32_x4
  return vld1_u32_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

uint64x1x4_t test_vld1_u64_x4(uint64_t const *a) {
  // CHECK-LABEL: test_vld1_u64_x4
  return vld1_u64_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

int8x8x4_t test_vld1_s8_x4(int8_t const *a) {
  // CHECK-LABEL: test_vld1_s8_x4
  return vld1_s8_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

int16x4x4_t test_vld1_s16_x4(int16_t const *a) {
  // CHECK-LABEL: test_vld1_s16_x4
  return vld1_s16_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

int32x2x4_t test_vld1_s32_x4(int32_t const *a) {
  // CHECK-LABEL: test_vld1_s32_x4
  return vld1_s32_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

int64x1x4_t test_vld1_s64_x4(int64_t const *a) {
  // CHECK-LABEL: test_vld1_s64_x4
  return vld1_s64_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

float16x4x4_t test_vld1_f16_x4(float16_t const *a) {
  // CHECK-LABEL: test_vld1_f16_x4
  return vld1_f16_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

float32x2x4_t test_vld1_f32_x4(float32_t const *a) {
  // CHECK-LABEL: test_vld1_f32_x4
  return vld1_f32_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

float64x1x4_t test_vld1_f64_x4(float64_t const *a) {
  // CHECK-LABEL: test_vld1_f64_x4
  return vld1_f64_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

poly8x8x4_t test_vld1_p8_x4(poly8_t const *a) {
  // CHECK-LABEL: test_vld1_p8_x4
  return vld1_p8_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

poly16x4x4_t test_vld1_p16_x4(poly16_t const *a) {
  // CHECK-LABEL: test_vld1_p16_x4
  return vld1_p16_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

poly64x1x4_t test_vld1_p64_x4(poly64_t const *a) {
  // CHECK-LABEL: test_vld1_p64_x4
  return vld1_p64_x4(a);
  // CHECK: ld1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u8_x2(uint8_t *a, uint8x16x2_t b) {
  // CHECK-LABEL: test_vst1q_u8_x2
  vst1q_u8_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u16_x2(uint16_t *a, uint16x8x2_t b) {
  // CHECK-LABEL: test_vst1q_u16_x2
  vst1q_u16_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u32_x2(uint32_t *a, uint32x4x2_t b) {
  // CHECK-LABEL: test_vst1q_u32_x2
  vst1q_u32_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u64_x2(uint64_t *a, uint64x2x2_t b) {
  // CHECK-LABEL: test_vst1q_u64_x2
  vst1q_u64_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s8_x2(int8_t *a, int8x16x2_t b) {
  // CHECK-LABEL: test_vst1q_s8_x2
  vst1q_s8_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s16_x2(int16_t *a, int16x8x2_t b) {
  // CHECK-LABEL: test_vst1q_s16_x2
  vst1q_s16_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s32_x2(int32_t *a, int32x4x2_t b) {
  // CHECK-LABEL: test_vst1q_s32_x2
  vst1q_s32_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s64_x2(int64_t *a, int64x2x2_t b) {
  // CHECK-LABEL: test_vst1q_s64_x2
  vst1q_s64_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f16_x2(float16_t *a, float16x8x2_t b) {
  // CHECK-LABEL: test_vst1q_f16_x2
  vst1q_f16_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f32_x2(float32_t *a, float32x4x2_t b) {
  // CHECK-LABEL: test_vst1q_f32_x2
  vst1q_f32_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f64_x2(float64_t *a, float64x2x2_t b) {
  // CHECK-LABEL: test_vst1q_f64_x2
  vst1q_f64_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p8_x2(poly8_t *a, poly8x16x2_t b) {
  // CHECK-LABEL: test_vst1q_p8_x2
  vst1q_p8_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p16_x2(poly16_t *a, poly16x8x2_t b) {
  // CHECK-LABEL: test_vst1q_p16_x2
  vst1q_p16_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p64_x2(poly64_t *a, poly64x2x2_t b) {
  // CHECK-LABEL: test_vst1q_p64_x2
  vst1q_p64_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u8_x2(uint8_t *a, uint8x8x2_t b) {
  // CHECK-LABEL: test_vst1_u8_x2
  vst1_u8_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u16_x2(uint16_t *a, uint16x4x2_t b) {
  // CHECK-LABEL: test_vst1_u16_x2
  vst1_u16_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u32_x2(uint32_t *a, uint32x2x2_t b) {
  // CHECK-LABEL: test_vst1_u32_x2
  vst1_u32_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u64_x2(uint64_t *a, uint64x1x2_t b) {
  // CHECK-LABEL: test_vst1_u64_x2
  vst1_u64_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s8_x2(int8_t *a, int8x8x2_t b) {
  // CHECK-LABEL: test_vst1_s8_x2
  vst1_s8_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s16_x2(int16_t *a, int16x4x2_t b) {
  // CHECK-LABEL: test_vst1_s16_x2
  vst1_s16_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s32_x2(int32_t *a, int32x2x2_t b) {
  // CHECK-LABEL: test_vst1_s32_x2
  vst1_s32_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s64_x2(int64_t *a, int64x1x2_t b) {
  // CHECK-LABEL: test_vst1_s64_x2
  vst1_s64_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f16_x2(float16_t *a, float16x4x2_t b) {
  // CHECK-LABEL: test_vst1_f16_x2
  vst1_f16_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f32_x2(float32_t *a, float32x2x2_t b) {
  // CHECK-LABEL: test_vst1_f32_x2
  vst1_f32_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f64_x2(float64_t *a, float64x1x2_t b) {
  // CHECK-LABEL: test_vst1_f64_x2
  vst1_f64_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p8_x2(poly8_t *a, poly8x8x2_t b) {
  // CHECK-LABEL: test_vst1_p8_x2
  vst1_p8_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p16_x2(poly16_t *a, poly16x4x2_t b) {
  // CHECK-LABEL: test_vst1_p16_x2
  vst1_p16_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p64_x2(poly64_t *a, poly64x1x2_t b) {
  // CHECK-LABEL: test_vst1_p64_x2
  vst1_p64_x2(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u8_x3(uint8_t *a, uint8x16x3_t b) {
  // CHECK-LABEL: test_vst1q_u8_x3
  vst1q_u8_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u16_x3(uint16_t *a, uint16x8x3_t b) {
  // CHECK-LABEL: test_vst1q_u16_x3
  vst1q_u16_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u32_x3(uint32_t *a, uint32x4x3_t b) {
  // CHECK-LABEL: test_vst1q_u32_x3
  vst1q_u32_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u64_x3(uint64_t *a, uint64x2x3_t b) {
  // CHECK-LABEL: test_vst1q_u64_x3
  vst1q_u64_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s8_x3(int8_t *a, int8x16x3_t b) {
  // CHECK-LABEL: test_vst1q_s8_x3
  vst1q_s8_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s16_x3(int16_t *a, int16x8x3_t b) {
  // CHECK-LABEL: test_vst1q_s16_x3
  vst1q_s16_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s32_x3(int32_t *a, int32x4x3_t b) {
  // CHECK-LABEL: test_vst1q_s32_x3
  vst1q_s32_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s64_x3(int64_t *a, int64x2x3_t b) {
  // CHECK-LABEL: test_vst1q_s64_x3
  vst1q_s64_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f16_x3(float16_t *a, float16x8x3_t b) {
  // CHECK-LABEL: test_vst1q_f16_x3
  vst1q_f16_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f32_x3(float32_t *a, float32x4x3_t b) {
  // CHECK-LABEL: test_vst1q_f32_x3
  vst1q_f32_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f64_x3(float64_t *a, float64x2x3_t b) {
  // CHECK-LABEL: test_vst1q_f64_x3
  vst1q_f64_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p8_x3(poly8_t *a, poly8x16x3_t b) {
  // CHECK-LABEL: test_vst1q_p8_x3
  vst1q_p8_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p16_x3(poly16_t *a, poly16x8x3_t b) {
  // CHECK-LABEL: test_vst1q_p16_x3
  vst1q_p16_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p64_x3(poly64_t *a, poly64x2x3_t b) {
  // CHECK-LABEL: test_vst1q_p64_x3
  vst1q_p64_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u8_x3(uint8_t *a, uint8x8x3_t b) {
  // CHECK-LABEL: test_vst1_u8_x3
  vst1_u8_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u16_x3(uint16_t *a, uint16x4x3_t b) {
  // CHECK-LABEL: test_vst1_u16_x3
  vst1_u16_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u32_x3(uint32_t *a, uint32x2x3_t b) {
  // CHECK-LABEL: test_vst1_u32_x3
  vst1_u32_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u64_x3(uint64_t *a, uint64x1x3_t b) {
  // CHECK-LABEL: test_vst1_u64_x3
  vst1_u64_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s8_x3(int8_t *a, int8x8x3_t b) {
  // CHECK-LABEL: test_vst1_s8_x3
  vst1_s8_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s16_x3(int16_t *a, int16x4x3_t b) {
  // CHECK-LABEL: test_vst1_s16_x3
  vst1_s16_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s32_x3(int32_t *a, int32x2x3_t b) {
  // CHECK-LABEL: test_vst1_s32_x3
  vst1_s32_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s64_x3(int64_t *a, int64x1x3_t b) {
  // CHECK-LABEL: test_vst1_s64_x3
  vst1_s64_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f16_x3(float16_t *a, float16x4x3_t b) {
  // CHECK-LABEL: test_vst1_f16_x3
  vst1_f16_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f32_x3(float32_t *a, float32x2x3_t b) {
  // CHECK-LABEL: test_vst1_f32_x3
  vst1_f32_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f64_x3(float64_t *a, float64x1x3_t b) {
  // CHECK-LABEL: test_vst1_f64_x3
  vst1_f64_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p8_x3(poly8_t *a, poly8x8x3_t b) {
  // CHECK-LABEL: test_vst1_p8_x3
  vst1_p8_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p16_x3(poly16_t *a, poly16x4x3_t b) {
  // CHECK-LABEL: test_vst1_p16_x3
  vst1_p16_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p64_x3(poly64_t *a, poly64x1x3_t b) {
  // CHECK-LABEL: test_vst1_p64_x3
  vst1_p64_x3(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u8_x4(uint8_t *a, uint8x16x4_t b) {
  // CHECK-LABEL: test_vst1q_u8_x4
  vst1q_u8_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u16_x4(uint16_t *a, uint16x8x4_t b) {
  // CHECK-LABEL: test_vst1q_u16_x4
  vst1q_u16_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u32_x4(uint32_t *a, uint32x4x4_t b) {
  // CHECK-LABEL: test_vst1q_u32_x4
  vst1q_u32_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_u64_x4(uint64_t *a, uint64x2x4_t b) {
  // CHECK-LABEL: test_vst1q_u64_x4
  vst1q_u64_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s8_x4(int8_t *a, int8x16x4_t b) {
  // CHECK-LABEL: test_vst1q_s8_x4
  vst1q_s8_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s16_x4(int16_t *a, int16x8x4_t b) {
  // CHECK-LABEL: test_vst1q_s16_x4
  vst1q_s16_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s32_x4(int32_t *a, int32x4x4_t b) {
  // CHECK-LABEL: test_vst1q_s32_x4
  vst1q_s32_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_s64_x4(int64_t *a, int64x2x4_t b) {
  // CHECK-LABEL: test_vst1q_s64_x4
  vst1q_s64_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f16_x4(float16_t *a, float16x8x4_t b) {
  // CHECK-LABEL: test_vst1q_f16_x4
  vst1q_f16_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f32_x4(float32_t *a, float32x4x4_t b) {
  // CHECK-LABEL: test_vst1q_f32_x4
  vst1q_f32_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s, v[0-9]+.4s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_f64_x4(float64_t *a, float64x2x4_t b) {
  // CHECK-LABEL: test_vst1q_f64_x4
  vst1q_f64_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p8_x4(poly8_t *a, poly8x16x4_t b) {
  // CHECK-LABEL: test_vst1q_p8_x4
  vst1q_p8_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b, v[0-9]+.16b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p16_x4(poly16_t *a, poly16x8x4_t b) {
  // CHECK-LABEL: test_vst1q_p16_x4
  vst1q_p16_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h, v[0-9]+.8h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p64_x4(poly64_t *a, poly64x2x4_t b) {
  // CHECK-LABEL: test_vst1q_p64_x4
  vst1q_p64_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u8_x4(uint8_t *a, uint8x8x4_t b) {
  // CHECK-LABEL: test_vst1_u8_x4
  vst1_u8_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u16_x4(uint16_t *a, uint16x4x4_t b) {
  // CHECK-LABEL: test_vst1_u16_x4
  vst1_u16_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u32_x4(uint32_t *a, uint32x2x4_t b) {
  // CHECK-LABEL: test_vst1_u32_x4
  vst1_u32_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_u64_x4(uint64_t *a, uint64x1x4_t b) {
  // CHECK-LABEL: test_vst1_u64_x4
  vst1_u64_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s8_x4(int8_t *a, int8x8x4_t b) {
  // CHECK-LABEL: test_vst1_s8_x4
  vst1_s8_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s16_x4(int16_t *a, int16x4x4_t b) {
  // CHECK-LABEL: test_vst1_s16_x4
  vst1_s16_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s32_x4(int32_t *a, int32x2x4_t b) {
  // CHECK-LABEL: test_vst1_s32_x4
  vst1_s32_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_s64_x4(int64_t *a, int64x1x4_t b) {
  // CHECK-LABEL: test_vst1_s64_x4
  vst1_s64_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f16_x4(float16_t *a, float16x4x4_t b) {
  // CHECK-LABEL: test_vst1_f16_x4
  vst1_f16_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f32_x4(float32_t *a, float32x2x4_t b) {
  // CHECK-LABEL: test_vst1_f32_x4
  vst1_f32_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s, v[0-9]+.2s ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_f64_x4(float64_t *a, float64x1x4_t b) {
  // CHECK-LABEL: test_vst1_f64_x4
  vst1_f64_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p8_x4(poly8_t *a, poly8x8x4_t b) {
  // CHECK-LABEL: test_vst1_p8_x4
  vst1_p8_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b, v[0-9]+.8b ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p16_x4(poly16_t *a, poly16x4x4_t b) {
  // CHECK-LABEL: test_vst1_p16_x4
  vst1_p16_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h, v[0-9]+.4h ?}}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p64_x4(poly64_t *a, poly64x1x4_t b) {
  // CHECK-LABEL: test_vst1_p64_x4
  vst1_p64_x4(a, b);
  // CHECK: st1 {{{ ?v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d ?}}}, [{{x[0-9]+|sp}}]
}

int64_t test_vceqd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vceqd_s64
// CHECK: {{cmeq d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (int64_t)vceqd_s64(a, b);
}

uint64_t test_vceqd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vceqd_u64
// CHECK: {{cmeq d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (int64_t)vceqd_u64(a, b);
}

int64_t test_vceqzd_s64(int64_t a) {
// CHECK-LABEL: test_vceqzd_s64
// CHECK: {{cmeq d[0-9]+, d[0-9]+, #0x0|cmp x0, #0}}
  return (int64_t)vceqzd_s64(a);
}

int64_t test_vceqzd_u64(int64_t a) {
// CHECK-LABEL: test_vceqzd_u64
// CHECK: {{cmeq d[0-9]+, d[0-9]+, #0x0|cmp x0, #0}}
  return (int64_t)vceqzd_u64(a);
}

int64_t test_vcged_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vcged_s64
// CHECK: {{cmge d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (int64_t)vcged_s64(a, b);
}

uint64_t test_vcged_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vcged_u64
// CHECK: {{cmhs d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
    return (uint64_t)vcged_u64(a, b);
}

int64_t test_vcgezd_s64(int64_t a) {
// CHECK-LABEL: test_vcgezd_s64
// CHECK: {{cmge d[0-9]+, d[0-9]+, #0x0|eor x0, x[0-9]+, x0, asr #63}}
  return (int64_t)vcgezd_s64(a);
}

int64_t test_vcgtd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vcgtd_s64
// CHECK: {{cmgt d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (int64_t)vcgtd_s64(a, b);
}

uint64_t test_vcgtd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vcgtd_u64
// CHECK: {{cmhi d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (uint64_t)vcgtd_u64(a, b);
}

int64_t test_vcgtzd_s64(int64_t a) {
// CHECK-LABEL: test_vcgtzd_s64
// CHECK: {{cmgt d[0-9]+, d[0-9]+, #0x0|cmp x0, #0}}
  return (int64_t)vcgtzd_s64(a);
}

int64_t test_vcled_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vcled_s64
// CHECK: {{cmge d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (int64_t)vcled_s64(a, b);
}

uint64_t test_vcled_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vcled_u64
// CHECK: {{cmhs d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (uint64_t)vcled_u64(a, b);
}

int64_t test_vclezd_s64(int64_t a) {
// CHECK-LABEL: test_vclezd_s64
// CHECK: {{cmle d[0-9]+, d[0-9]+, #0x0|cmp x0, #1}}
  return (int64_t)vclezd_s64(a);
}

int64_t test_vcltd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vcltd_s64
// CHECK: {{cmgt d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (int64_t)vcltd_s64(a, b);
}

uint64_t test_vcltd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vcltd_u64
// CHECK: {{cmhi d[0-9]+, d[0-9]+, d[0-9]+|cmp x0, x1}}
  return (uint64_t)vcltd_u64(a, b);
}

int64_t test_vcltzd_s64(int64_t a) {
// CHECK-LABEL: test_vcltzd_s64
// CHECK: {{cmlt d[0-9]+, d[0-9]+, #0x0|asr x0, x0, #63}}
  return (int64_t)vcltzd_s64(a);
}

int64_t test_vtstd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vtstd_s64
// CHECK: {{cmtst d[0-9]+, d[0-9]+, d[0-9]+|tst x1, x0}}
  return (int64_t)vtstd_s64(a, b);
}

uint64_t test_vtstd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vtstd_u64
// CHECK: {{cmtst d[0-9]+, d[0-9]+, d[0-9]+|tst x1, x0}}
  return (uint64_t)vtstd_u64(a, b);
}

int64_t test_vabsd_s64(int64_t a) {
// CHECK-LABEL: test_vabsd_s64
// CHECK: abs {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vabsd_s64(a);
}

int8_t test_vqabsb_s8(int8_t a) {
// CHECK-LABEL: test_vqabsb_s8
// CHECK: sqabs {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
  return (int8_t)vqabsb_s8(a);
}

int16_t test_vqabsh_s16(int16_t a) {
// CHECK-LABEL: test_vqabsh_s16
// CHECK: sqabs {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
  return (int16_t)vqabsh_s16(a);
}

int32_t test_vqabss_s32(int32_t a) {
// CHECK-LABEL: test_vqabss_s32
// CHECK: sqabs {{s[0-9]+}}, {{s[0-9]+}}
  return (int32_t)vqabss_s32(a);
}

int64_t test_vqabsd_s64(int64_t a) {
// CHECK-LABEL: test_vqabsd_s64
// CHECK: sqabs {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vqabsd_s64(a);
}

int64_t test_vnegd_s64(int64_t a) {
// CHECK-LABEL: test_vnegd_s64
// CHECK: neg {{[xd][0-9]+}}, {{[xd][0-9]+}}
  return (int64_t)vnegd_s64(a);
}

int8_t test_vqnegb_s8(int8_t a) {
// CHECK-LABEL: test_vqnegb_s8
// CHECK: sqneg {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
  return (int8_t)vqnegb_s8(a);
}

int16_t test_vqnegh_s16(int16_t a) {
// CHECK-LABEL: test_vqnegh_s16
// CHECK: sqneg {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
  return (int16_t)vqnegh_s16(a);
}

int32_t test_vqnegs_s32(int32_t a) {
// CHECK-LABEL: test_vqnegs_s32
// CHECK: sqneg {{s[0-9]+}}, {{s[0-9]+}}
  return (int32_t)vqnegs_s32(a);
}

int64_t test_vqnegd_s64(int64_t a) {
// CHECK-LABEL: test_vqnegd_s64
// CHECK: sqneg {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vqnegd_s64(a);
}

int8_t test_vuqaddb_s8(int8_t a, int8_t b) {
// CHECK-LABEL: test_vuqaddb_s8
// CHECK: suqadd {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
  return (int8_t)vuqaddb_s8(a, b);
}

int16_t test_vuqaddh_s16(int16_t a, int16_t b) {
// CHECK-LABEL: test_vuqaddh_s16
// CHECK: suqadd {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
  return (int16_t)vuqaddh_s16(a, b);
}

int32_t test_vuqadds_s32(int32_t a, int32_t b) {
// CHECK-LABEL: test_vuqadds_s32
// CHECK: suqadd {{s[0-9]+}}, {{s[0-9]+}}
  return (int32_t)vuqadds_s32(a, b);
}

int64_t test_vuqaddd_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vuqaddd_s64
// CHECK: suqadd {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vuqaddd_s64(a, b);
}

uint8_t test_vsqaddb_u8(uint8_t a, uint8_t b) {
// CHECK-LABEL: test_vsqaddb_u8
// CHECK: usqadd {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}
  return (uint8_t)vsqaddb_u8(a, b);
}

uint16_t test_vsqaddh_u16(uint16_t a, uint16_t b) {
// CHECK-LABEL: test_vsqaddh_u16
// CHECK: usqadd {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
  return (uint16_t)vsqaddh_u16(a, b);
}

uint32_t test_vsqadds_u32(uint32_t a, uint32_t b) {
// CHECK-LABEL: test_vsqadds_u32
// CHECK: usqadd {{s[0-9]+}}, {{s[0-9]+}}
  return (uint32_t)vsqadds_u32(a, b);
}

uint64_t test_vsqaddd_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vsqaddd_u64
// CHECK: usqadd {{d[0-9]+}}, {{d[0-9]+}}
  return (uint64_t)vsqaddd_u64(a, b);
}

int32_t test_vqdmlalh_s16(int32_t a, int16_t b, int16_t c) {

// CHECK-ARM64-LABEL: test_vqdmlalh_s16
// CHECK-ARM64: sqdmull v[[PROD:[0-9]+]].4s, {{v[0-9]+.4h}}, {{v[0-9]+.4h}}
// CHECK-ARM64: sqadd {{s[0-9]+}}, {{s[0-9]+}}, s[[PROD]]
  return (int32_t)vqdmlalh_s16(a, b, c);
}

int64_t test_vqdmlals_s32(int64_t a, int32_t b, int32_t c) {
// CHECK-LABEL: test_vqdmlals_s32
// CHECK: sqdmlal {{d[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  return (int64_t)vqdmlals_s32(a, b, c);
}

int32_t test_vqdmlslh_s16(int32_t a, int16_t b, int16_t c) {

// CHECK-ARM64-LABEL: test_vqdmlslh_s16
// CHECK-ARM64: sqdmull v[[PROD:[0-9]+]].4s, {{v[0-9]+.4h}}, {{v[0-9]+.4h}}
// CHECK-ARM64: sqsub {{s[0-9]+}}, {{s[0-9]+}}, s[[PROD]]
  return (int32_t)vqdmlslh_s16(a, b, c);
}

int64_t test_vqdmlsls_s32(int64_t a, int32_t b, int32_t c) {
// CHECK-LABEL: test_vqdmlsls_s32
// CHECK: sqdmlsl {{d[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  return (int64_t)vqdmlsls_s32(a, b, c);
}

int32_t test_vqdmullh_s16(int16_t a, int16_t b) {
// CHECK-LABEL: test_vqdmullh_s16
// CHECK: sqdmull {{s[0-9]+|v[0-9]+.4s}}, {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}
  return (int32_t)vqdmullh_s16(a, b);
}

int64_t test_vqdmulls_s32(int32_t a, int32_t b) {
// CHECK-LABEL: test_vqdmulls_s32
// CHECK: sqdmull {{d[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  return (int64_t)vqdmulls_s32(a, b);
}

int8_t test_vqmovunh_s16(int16_t a) {
// CHECK-LABEL: test_vqmovunh_s16
// CHECK: sqxtun {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}
  return (int8_t)vqmovunh_s16(a);
}

int16_t test_vqmovuns_s32(int32_t a) {
// CHECK-LABEL: test_vqmovuns_s32
// CHECK: sqxtun {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}
  return (int16_t)vqmovuns_s32(a);
}

int32_t test_vqmovund_s64(int64_t a) {
// CHECK-LABEL: test_vqmovund_s64
// CHECK: sqxtun {{s[0-9]+}}, {{d[0-9]+}}
  return (int32_t)vqmovund_s64(a);
}

int8_t test_vqmovnh_s16(int16_t a) {
// CHECK-LABEL: test_vqmovnh_s16
// CHECK: sqxtn {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}
  return (int8_t)vqmovnh_s16(a);
}

int16_t test_vqmovns_s32(int32_t a) {
// CHECK-LABEL: test_vqmovns_s32
// CHECK: sqxtn {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}
  return (int16_t)vqmovns_s32(a);
}

int32_t test_vqmovnd_s64(int64_t a) {
// CHECK-LABEL: test_vqmovnd_s64
// CHECK: sqxtn {{s[0-9]+}}, {{d[0-9]+}}
  return (int32_t)vqmovnd_s64(a);
}

int8_t test_vqmovnh_u16(int16_t a) {
// CHECK-LABEL: test_vqmovnh_u16
// CHECK: uqxtn {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}
  return (int8_t)vqmovnh_u16(a);
}

int16_t test_vqmovns_u32(int32_t a) {
// CHECK-LABEL: test_vqmovns_u32
// CHECK: uqxtn {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}
  return (int16_t)vqmovns_u32(a);
}

int32_t test_vqmovnd_u64(int64_t a) {
// CHECK-LABEL: test_vqmovnd_u64
// CHECK: uqxtn {{s[0-9]+}}, {{d[0-9]+}}
  return (int32_t)vqmovnd_u64(a);
}

uint32_t test_vceqs_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vceqs_f32
// CHECK: {{fcmeq s0, s0, s1|fcmp s0, s1}}
  return (uint32_t)vceqs_f32(a, b);
}

uint64_t test_vceqd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vceqd_f64
// CHECK: {{fcmeq d0, d0, d1|fcmp d0, d1}}
  return (uint64_t)vceqd_f64(a, b);
}

uint32_t test_vceqzs_f32(float32_t a) {
// CHECK-LABEL: test_vceqzs_f32
// CHECK: {{fcmeq s0, s0, #0.0|fcmp s0, #0.0}}
  return (uint32_t)vceqzs_f32(a);
}

uint64_t test_vceqzd_f64(float64_t a) {
// CHECK-LABEL: test_vceqzd_f64
// CHECK: {{fcmeq d0, d0, #0.0|fcmp d0, #0.0}}
  return (uint64_t)vceqzd_f64(a);
}

uint32_t test_vcges_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vcges_f32
// CHECK: {{fcmge s0, s0, s1|fcmp s0, s1}}
  return (uint32_t)vcges_f32(a, b);
}

uint64_t test_vcged_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vcged_f64
// CHECK: {{fcmge d0, d0, d1|fcmp d0, d1}}
  return (uint64_t)vcged_f64(a, b);
}

uint32_t test_vcgezs_f32(float32_t a) {
// CHECK-LABEL: test_vcgezs_f32
// CHECK: {{fcmge s0, s0, #0.0|fcmp s0, #0.0}}
  return (uint32_t)vcgezs_f32(a);
}

uint64_t test_vcgezd_f64(float64_t a) {
// CHECK-LABEL: test_vcgezd_f64
// CHECK: {{fcmge d0, d0, #0.0|fcmp d0, #0.0}}
  return (uint64_t)vcgezd_f64(a);
}

uint32_t test_vcgts_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vcgts_f32
// CHECK: {{fcmgt s0, s0, s1|fcmp s0, s1}}
  return (uint32_t)vcgts_f32(a, b);
}

uint64_t test_vcgtd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vcgtd_f64
// CHECK: {{fcmgt d0, d0, d1|fcmp d0, d1}}
  return (uint64_t)vcgtd_f64(a, b);
}

uint32_t test_vcgtzs_f32(float32_t a) {
// CHECK-LABEL: test_vcgtzs_f32
// CHECK: {{fcmgt s0, s0, #0.0|fcmp s0, #0.0}}
  return (uint32_t)vcgtzs_f32(a);
}

uint64_t test_vcgtzd_f64(float64_t a) {
// CHECK-LABEL: test_vcgtzd_f64
// CHECK: {{fcmgt d0, d0, #0.0|fcmp d0, #0.0}}
  return (uint64_t)vcgtzd_f64(a);
}

uint32_t test_vcles_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vcles_f32
// CHECK: {{fcmge s0, s1, s0|fcmp s0, s1}}
  return (uint32_t)vcles_f32(a, b);
}

uint64_t test_vcled_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vcled_f64
// CHECK: {{fcmge d0, d1, d0|fcmp d0, d1}}
  return (uint64_t)vcled_f64(a, b);
}

uint32_t test_vclezs_f32(float32_t a) {
// CHECK-LABEL: test_vclezs_f32
// CHECK: {{fcmle s0, s0, #0.0|fcmp s0, #0.0}}
  return (uint32_t)vclezs_f32(a);
}

uint64_t test_vclezd_f64(float64_t a) {
// CHECK-LABEL: test_vclezd_f64
// CHECK: {{fcmle d0, d0, #0.0|fcmp d0, #0.0}}
  return (uint64_t)vclezd_f64(a);
}

uint32_t test_vclts_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vclts_f32
// CHECK: {{fcmgt s0, s1, s0|fcmp s0, s1}}
  return (uint32_t)vclts_f32(a, b);
}

uint64_t test_vcltd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vcltd_f64
// CHECK: {{fcmgt d0, d1, d0|fcmp d0, d1}}
  return (uint64_t)vcltd_f64(a, b);
}

uint32_t test_vcltzs_f32(float32_t a) {
// CHECK-LABEL: test_vcltzs_f32
// CHECK: {{fcmlt s0, s0, #0.0|fcmp s0, #0.0}}
  return (uint32_t)vcltzs_f32(a);
}

uint64_t test_vcltzd_f64(float64_t a) {
// CHECK-LABEL: test_vcltzd_f64
// CHECK: {{fcmlt d0, d0, #0.0|fcmp d0, #0.0}}
  return (uint64_t)vcltzd_f64(a);
}

uint32_t test_vcages_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vcages_f32
// CHECK: facge s0, s0, s1
  return (uint32_t)vcages_f32(a, b);
}

uint64_t test_vcaged_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vcaged_f64
// CHECK: facge d0, d0, d1
  return (uint64_t)vcaged_f64(a, b);
}

uint32_t test_vcagts_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vcagts_f32
// CHECK: facgt s0, s0, s1
  return (uint32_t)vcagts_f32(a, b);
}

uint64_t test_vcagtd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vcagtd_f64
// CHECK: facgt d0, d0, d1
  return (uint64_t)vcagtd_f64(a, b);
}

uint32_t test_vcales_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vcales_f32
// CHECK: facge s0, s1, s0
  return (uint32_t)vcales_f32(a, b);
}

uint64_t test_vcaled_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vcaled_f64
// CHECK: facge d0, d1, d0
  return (uint64_t)vcaled_f64(a, b);
}

uint32_t test_vcalts_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vcalts_f32
// CHECK: facgt s0, s1, s0
  return (uint32_t)vcalts_f32(a, b);
}

uint64_t test_vcaltd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vcaltd_f64
// CHECK: facgt d0, d1, d0
  return (uint64_t)vcaltd_f64(a, b);
}

int64_t test_vshrd_n_s64(int64_t a) {
// CHECK-LABEL: test_vshrd_n_s64
// CHECK: {{sshr d[0-9]+, d[0-9]+, #1|asr x0, x0, #1}}
  return (int64_t)vshrd_n_s64(a, 1);
}

int64x1_t test_vshr_n_s64(int64x1_t a) {
// CHECK-LABEL: test_vshr_n_s64
// CHECK: sshr {{d[0-9]+}}, {{d[0-9]+}}, #1
  return vshr_n_s64(a, 1);
}

uint64_t test_vshrd_n_u64(uint64_t a) {

// CHECK-ARM64-LABEL: test_vshrd_n_u64
// CHECK-ARM64: mov x0, xzr
  return (uint64_t)vshrd_n_u64(a, 64);
}

uint64_t test_vshrd_n_u64_2() {

// CHECK-ARM64-LABEL: test_vshrd_n_u64_2
// CHECK-ARM64: mov x0, xzr
  uint64_t a = UINT64_C(0xf000000000000000);
  return vshrd_n_u64(a, 64);
}

uint64x1_t test_vshr_n_u64(uint64x1_t a) {
// CHECK-LABEL: test_vshr_n_u64
// CHECK: ushr {{d[0-9]+}}, {{d[0-9]+}}, #1
  return vshr_n_u64(a, 1);
}

int64_t test_vrshrd_n_s64(int64_t a) {
// CHECK-LABEL: test_vrshrd_n_s64
// CHECK: srshr {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (int64_t)vrshrd_n_s64(a, 63);
}

int64x1_t test_vrshr_n_s64(int64x1_t a) {
// CHECK-LABEL: test_vrshr_n_s64
// CHECK: srshr d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vrshr_n_s64(a, 1);
}

uint64_t test_vrshrd_n_u64(uint64_t a) {
// CHECK-LABEL: test_vrshrd_n_u64
// CHECK: urshr {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (uint64_t)vrshrd_n_u64(a, 63);
}

uint64x1_t test_vrshr_n_u64(uint64x1_t a) {
// CHECK-LABEL: test_vrshr_n_u64
// CHECK: urshr d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vrshr_n_u64(a, 1);
}

int64_t test_vsrad_n_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vsrad_n_s64
// CHECK: {{ssra d[0-9]+, d[0-9]+, #63|add x0, x0, x1, asr #63}}
  return (int64_t)vsrad_n_s64(a, b, 63);
}

int64x1_t test_vsra_n_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vsra_n_s64
// CHECK: ssra d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vsra_n_s64(a, b, 1);
}

uint64_t test_vsrad_n_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vsrad_n_u64
// CHECK: {{usra d[0-9]+, d[0-9]+, #63|add x0, x0, x1, lsr #63}}
  return (uint64_t)vsrad_n_u64(a, b, 63);
}

uint64_t test_vsrad_n_u64_2(uint64_t a, uint64_t b) {

// CHECK-ARM64-LABEL: test_vsrad_n_u64_2
// CHECK-ARM64-NOT: add
  return (uint64_t)vsrad_n_u64(a, b, 64);
}

uint64x1_t test_vsra_n_u64(uint64x1_t a, uint64x1_t b) {
// CHECK-LABEL: test_vsra_n_u64
// CHECK: usra d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vsra_n_u64(a, b, 1);
}

int64_t test_vrsrad_n_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vrsrad_n_s64
// CHECK: {{srsra d[0-9]+, d[0-9]+, #63}}
  return (int64_t)vrsrad_n_s64(a, b, 63);
}

int64x1_t test_vrsra_n_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vrsra_n_s64
// CHECK: srsra d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vrsra_n_s64(a, b, 1);
}

uint64_t test_vrsrad_n_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vrsrad_n_u64
// CHECK: ursra {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (uint64_t)vrsrad_n_u64(a, b, 63);
}

uint64x1_t test_vrsra_n_u64(uint64x1_t a, uint64x1_t b) {
// CHECK-LABEL: test_vrsra_n_u64
// CHECK: ursra d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vrsra_n_u64(a, b, 1);
}

int64_t test_vshld_n_s64(int64_t a) {
// CHECK-LABEL: test_vshld_n_s64
// CHECK: {{shl d[0-9]+, d[0-9]+, #1|lsl x0, x0, #1}}
  return (int64_t)vshld_n_s64(a, 1);
}
int64x1_t test_vshl_n_s64(int64x1_t a) {
// CHECK-LABEL: test_vshl_n_s64
// CHECK: shl d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vshl_n_s64(a, 1);
}

uint64_t test_vshld_n_u64(uint64_t a) {
// CHECK-LABEL: test_vshld_n_u64
// CHECK: {{shl d[0-9]+, d[0-9]+, #63|lsl x0, x0, #63}}
  return (uint64_t)vshld_n_u64(a, 63);
}

uint64x1_t test_vshl_n_u64(uint64x1_t a) {
// CHECK-LABEL: test_vshl_n_u64
// CHECK: shl d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vshl_n_u64(a, 1);
}

int8_t test_vqshlb_n_s8(int8_t a) {
// CHECK-LABEL: test_vqshlb_n_s8
// CHECK: sqshl {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, #7
  return (int8_t)vqshlb_n_s8(a, 7);
}

int16_t test_vqshlh_n_s16(int16_t a) {
// CHECK-LABEL: test_vqshlh_n_s16
// CHECK: sqshl {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, #15
  return (int16_t)vqshlh_n_s16(a, 15);
}

int32_t test_vqshls_n_s32(int32_t a) {
// CHECK-LABEL: test_vqshls_n_s32
// CHECK: sqshl {{s[0-9]+}}, {{s[0-9]+}}, #31
  return (int32_t)vqshls_n_s32(a, 31);
}

int64_t test_vqshld_n_s64(int64_t a) {
// CHECK-LABEL: test_vqshld_n_s64
// CHECK: sqshl {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (int64_t)vqshld_n_s64(a, 63);
}

int8x8_t test_vqshl_n_s8(int8x8_t a) {
  // CHECK-LABEL: test_vqshl_n_s8
  return vqshl_n_s8(a, 0);
  // CHECK: sqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
}

int8x16_t test_vqshlq_n_s8(int8x16_t a) {
  // CHECK-LABEL: test_vqshlq_n_s8
  return vqshlq_n_s8(a, 0);
  // CHECK: sqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
}

int16x4_t test_vqshl_n_s16(int16x4_t a) {
  // CHECK-LABEL: test_vqshl_n_s16
  return vqshl_n_s16(a, 0);
  // CHECK: sqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
}

int16x8_t test_vqshlq_n_s16(int16x8_t a) {
  // CHECK-LABEL: test_vqshlq_n_s16
  return vqshlq_n_s16(a, 0);
  // CHECK: sqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
}

int32x2_t test_vqshl_n_s32(int32x2_t a) {
  // CHECK-LABEL: test_vqshl_n_s32
  return vqshl_n_s32(a, 0);
  // CHECK: sqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
}

int32x4_t test_vqshlq_n_s32(int32x4_t a) {
  // CHECK-LABEL: test_vqshlq_n_s32
  return vqshlq_n_s32(a, 0);
  // CHECK: sqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
}

int64x2_t test_vqshlq_n_s64(int64x2_t a) {
  // CHECK-LABEL: test_vqshlq_n_s64
  return vqshlq_n_s64(a, 0);
  // CHECK: sqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
}

uint8x8_t test_vqshl_n_u8(uint8x8_t a) {
  // CHECK-LABEL: test_vqshl_n_u8
  return vqshl_n_u8(a, 0);
  // CHECK: uqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
}

uint8x16_t test_vqshlq_n_u8(uint8x16_t a) {
  // CHECK-LABEL: test_vqshlq_n_u8
  return vqshlq_n_u8(a, 0);
  // CHECK: uqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
}

uint16x4_t test_vqshl_n_u16(uint16x4_t a) {
  // CHECK-LABEL: test_vqshl_n_u16
  return vqshl_n_u16(a, 0);
  // CHECK: uqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
}

uint16x8_t test_vqshlq_n_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vqshlq_n_u16
  return vqshlq_n_u16(a, 0);
  // CHECK: uqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
}

uint32x2_t test_vqshl_n_u32(uint32x2_t a) {
  // CHECK-LABEL: test_vqshl_n_u32
  return vqshl_n_u32(a, 0);
  // CHECK: uqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
}

uint32x4_t test_vqshlq_n_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vqshlq_n_u32
  return vqshlq_n_u32(a, 0);
  // CHECK: uqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
}

uint64x2_t test_vqshlq_n_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vqshlq_n_u64
  return vqshlq_n_u64(a, 0);
  // CHECK: uqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
}

int64x1_t test_vqshl_n_s64(int64x1_t a) {
// CHECK-LABEL: test_vqshl_n_s64
// CHECK: sqshl d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vqshl_n_s64(a, 1);
}

uint8_t test_vqshlb_n_u8(uint8_t a) {
// CHECK-LABEL: test_vqshlb_n_u8
// CHECK: uqshl {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, #7
  return (uint8_t)vqshlb_n_u8(a, 7);
}

uint16_t test_vqshlh_n_u16(uint16_t a) {
// CHECK-LABEL: test_vqshlh_n_u16
// CHECK: uqshl {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, #15
  return (uint16_t)vqshlh_n_u16(a, 15);
}

uint32_t test_vqshls_n_u32(uint32_t a) {
// CHECK-LABEL: test_vqshls_n_u32
// CHECK: uqshl {{s[0-9]+}}, {{s[0-9]+}}, #31
  return (uint32_t)vqshls_n_u32(a, 31);
}

uint64_t test_vqshld_n_u64(uint64_t a) {
// CHECK-LABEL: test_vqshld_n_u64
// CHECK: uqshl {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (uint64_t)vqshld_n_u64(a, 63);
}

uint64x1_t test_vqshl_n_u64(uint64x1_t a) {
// CHECK-LABEL: test_vqshl_n_u64
// CHECK: uqshl d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vqshl_n_u64(a, 1);
}

int8_t test_vqshlub_n_s8(int8_t a) {
// CHECK-LABEL: test_vqshlub_n_s8
// CHECK: sqshlu {{b[0-9]+|v[0-9]+.8b}}, {{b[0-9]+|v[0-9]+.8b}}, #7
  return (int8_t)vqshlub_n_s8(a, 7);
}

int16_t test_vqshluh_n_s16(int16_t a) {
// CHECK-LABEL: test_vqshluh_n_s16
// CHECK: sqshlu {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, #15
  return (int16_t)vqshluh_n_s16(a, 15);
}

int32_t test_vqshlus_n_s32(int32_t a) {
// CHECK-LABEL: test_vqshlus_n_s32
// CHECK: sqshlu {{s[0-9]+}}, {{s[0-9]+}}, #31
  return (int32_t)vqshlus_n_s32(a, 31);
}

int64_t test_vqshlud_n_s64(int64_t a) {
// CHECK-LABEL: test_vqshlud_n_s64
// CHECK: sqshlu {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (int64_t)vqshlud_n_s64(a, 63);
}

uint64x1_t test_vqshlu_n_s64(int64x1_t a) {
// CHECK-LABEL: test_vqshlu_n_s64
// CHECK: sqshlu d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vqshlu_n_s64(a, 1);
}

int64_t test_vsrid_n_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vsrid_n_s64
// CHECK: sri {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (int64_t)vsrid_n_s64(a, b, 63);
}

int64x1_t test_vsri_n_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vsri_n_s64
// CHECK: sri d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vsri_n_s64(a, b, 1);
}

uint64_t test_vsrid_n_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vsrid_n_u64
// CHECK: sri {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (uint64_t)vsrid_n_u64(a, b, 63);
}

uint64x1_t test_vsri_n_u64(uint64x1_t a, uint64x1_t b) {
// CHECK-LABEL: test_vsri_n_u64
// CHECK: sri d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vsri_n_u64(a, b, 1);
}

int64_t test_vslid_n_s64(int64_t a, int64_t b) {
// CHECK-LABEL: test_vslid_n_s64
// CHECK: sli {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (int64_t)vslid_n_s64(a, b, 63);
}

int64x1_t test_vsli_n_s64(int64x1_t a, int64x1_t b) {
// CHECK-LABEL: test_vsli_n_s64
// CHECK: sli d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vsli_n_s64(a, b, 1);
}

uint64_t test_vslid_n_u64(uint64_t a, uint64_t b) {
// CHECK-LABEL: test_vslid_n_u64
// CHECK: sli {{d[0-9]+}}, {{d[0-9]+}}, #63
  return (uint64_t)vslid_n_u64(a, b, 63);
}

uint64x1_t test_vsli_n_u64(uint64x1_t a, uint64x1_t b) {
// CHECK-LABEL: test_vsli_n_u64
// CHECK: sli d{{[0-9]+}}, d{{[0-9]+}}, #1
  return vsli_n_u64(a, b, 1);
}

int8_t test_vqshrnh_n_s16(int16_t a) {
// CHECK-LABEL: test_vqshrnh_n_s16
// CHECK: sqshrn {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}, #8
  return (int8_t)vqshrnh_n_s16(a, 8);
}

int16_t test_vqshrns_n_s32(int32_t a) {
// CHECK-LABEL: test_vqshrns_n_s32
// CHECK: sqshrn {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}, #16
  return (int16_t)vqshrns_n_s32(a, 16);
}

int32_t test_vqshrnd_n_s64(int64_t a) {
// CHECK-LABEL: test_vqshrnd_n_s64
// CHECK: sqshrn {{s[0-9]+}}, {{d[0-9]+}}, #32
  return (int32_t)vqshrnd_n_s64(a, 32);
}

uint8_t test_vqshrnh_n_u16(uint16_t a) {
// CHECK-LABEL: test_vqshrnh_n_u16
// CHECK: uqshrn {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}, #8
  return (uint8_t)vqshrnh_n_u16(a, 8);
}

uint16_t test_vqshrns_n_u32(uint32_t a) {
// CHECK-LABEL: test_vqshrns_n_u32
// CHECK: uqshrn {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}, #16
  return (uint16_t)vqshrns_n_u32(a, 16);
}

uint32_t test_vqshrnd_n_u64(uint64_t a) {
// CHECK-LABEL: test_vqshrnd_n_u64
// CHECK: uqshrn {{s[0-9]+}}, {{d[0-9]+}}, #32
  return (uint32_t)vqshrnd_n_u64(a, 32);
}

int8_t test_vqrshrnh_n_s16(int16_t a) {
// CHECK-LABEL: test_vqrshrnh_n_s16
// CHECK: sqrshrn {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}, #8
  return (int8_t)vqrshrnh_n_s16(a, 8);
}

int16_t test_vqrshrns_n_s32(int32_t a) {
// CHECK-LABEL: test_vqrshrns_n_s32
// CHECK: sqrshrn {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}, #16
  return (int16_t)vqrshrns_n_s32(a, 16);
}

int32_t test_vqrshrnd_n_s64(int64_t a) {
// CHECK-LABEL: test_vqrshrnd_n_s64
// CHECK: sqrshrn {{s[0-9]+}}, {{d[0-9]+}}, #32
  return (int32_t)vqrshrnd_n_s64(a, 32);
}

uint8_t test_vqrshrnh_n_u16(uint16_t a) {
// CHECK-LABEL: test_vqrshrnh_n_u16
// CHECK: uqrshrn {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}, #8
  return (uint8_t)vqrshrnh_n_u16(a, 8);
}

uint16_t test_vqrshrns_n_u32(uint32_t a) {
// CHECK-LABEL: test_vqrshrns_n_u32
// CHECK: uqrshrn {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}, #16
  return (uint16_t)vqrshrns_n_u32(a, 16);
}

uint32_t test_vqrshrnd_n_u64(uint64_t a) {
// CHECK-LABEL: test_vqrshrnd_n_u64
// CHECK: uqrshrn {{s[0-9]+}}, {{d[0-9]+}}, #32
  return (uint32_t)vqrshrnd_n_u64(a, 32);
}

int8_t test_vqshrunh_n_s16(int16_t a) {
// CHECK-LABEL: test_vqshrunh_n_s16
// CHECK: sqshrun {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}, #8
  return (int8_t)vqshrunh_n_s16(a, 8);
}

int16_t test_vqshruns_n_s32(int32_t a) {
// CHECK-LABEL: test_vqshruns_n_s32
// CHECK: sqshrun {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}, #16
  return (int16_t)vqshruns_n_s32(a, 16);
}

int32_t test_vqshrund_n_s64(int64_t a) {
// CHECK-LABEL: test_vqshrund_n_s64
// CHECK: sqshrun {{s[0-9]+}}, {{d[0-9]+}}, #32
  return (int32_t)vqshrund_n_s64(a, 32);
}

int8_t test_vqrshrunh_n_s16(int16_t a) {
// CHECK-LABEL: test_vqrshrunh_n_s16
// CHECK: sqrshrun {{b[0-9]+|v[0-9]+.8b}}, {{h[0-9]+|v[0-9]+.8h}}, #8
  return (int8_t)vqrshrunh_n_s16(a, 8);
}

int16_t test_vqrshruns_n_s32(int32_t a) {
// CHECK-LABEL: test_vqrshruns_n_s32
// CHECK: sqrshrun {{h[0-9]+|v[0-9]+.4h}}, {{s[0-9]+|v[0-9]+.4s}}, #16
  return (int16_t)vqrshruns_n_s32(a, 16);
}

int32_t test_vqrshrund_n_s64(int64_t a) {
// CHECK-LABEL: test_vqrshrund_n_s64
// CHECK: sqrshrun {{s[0-9]+}}, {{d[0-9]+}}, #32
  return (int32_t)vqrshrund_n_s64(a, 32);
}

float32_t test_vcvts_n_f32_s32(int32_t a) {
// CHECK-LABEL: test_vcvts_n_f32_s32
// CHECK: scvtf {{s[0-9]+}}, {{s[0-9]+}}, #1
  return vcvts_n_f32_s32(a, 1);
}

float64_t test_vcvtd_n_f64_s64(int64_t a) {
// CHECK-LABEL: test_vcvtd_n_f64_s64
// CHECK: scvtf {{d[0-9]+}}, {{d[0-9]+}}, #1
  return vcvtd_n_f64_s64(a, 1);
}

float32_t test_vcvts_n_f32_u32(uint32_t a) {
// CHECK-LABEL: test_vcvts_n_f32_u32
// CHECK: ucvtf {{s[0-9]+}}, {{s[0-9]+}}, #32
  return vcvts_n_f32_u32(a, 32);
}

float64_t test_vcvtd_n_f64_u64(uint64_t a) {
// CHECK-LABEL: test_vcvtd_n_f64_u64
// CHECK: ucvtf {{d[0-9]+}}, {{d[0-9]+}}, #64
  return vcvtd_n_f64_u64(a, 64);
}

int32_t test_vcvts_n_s32_f32(float32_t a) {
// CHECK-LABEL: test_vcvts_n_s32_f32
// CHECK: fcvtzs {{s[0-9]+}}, {{s[0-9]+}}, #1
  return (int32_t)vcvts_n_s32_f32(a, 1);
}

int64_t test_vcvtd_n_s64_f64(float64_t a) {
// CHECK-LABEL: test_vcvtd_n_s64_f64
// CHECK: fcvtzs {{d[0-9]+}}, {{d[0-9]+}}, #1
  return (int64_t)vcvtd_n_s64_f64(a, 1);
}

uint32_t test_vcvts_n_u32_f32(float32_t a) {
// CHECK-LABEL: test_vcvts_n_u32_f32
// CHECK: fcvtzu {{s[0-9]+}}, {{s[0-9]+}}, #32
  return (uint32_t)vcvts_n_u32_f32(a, 32);
}

uint64_t test_vcvtd_n_u64_f64(float64_t a) {
// CHECK-LABEL: test_vcvtd_n_u64_f64
// CHECK: fcvtzu {{d[0-9]+}}, {{d[0-9]+}}, #64
  return (uint64_t)vcvtd_n_u64_f64(a, 64);
}

// CHECK-LABEL: test_vreinterpret_s8_s16:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_s16(int16x4_t a) {
  return vreinterpret_s8_s16(a);
}

// CHECK-LABEL: test_vreinterpret_s8_s32:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_s32(int32x2_t a) {
  return vreinterpret_s8_s32(a);
}

// CHECK-LABEL: test_vreinterpret_s8_s64:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_s64(int64x1_t a) {
  return vreinterpret_s8_s64(a);
}

// CHECK-LABEL: test_vreinterpret_s8_u8:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_u8(uint8x8_t a) {
  return vreinterpret_s8_u8(a);
}

// CHECK-LABEL: test_vreinterpret_s8_u16:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_u16(uint16x4_t a) {
  return vreinterpret_s8_u16(a);
}

// CHECK-LABEL: test_vreinterpret_s8_u32:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_u32(uint32x2_t a) {
  return vreinterpret_s8_u32(a);
}

// CHECK-LABEL: test_vreinterpret_s8_u64:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_u64(uint64x1_t a) {
  return vreinterpret_s8_u64(a);
}

// CHECK-LABEL: test_vreinterpret_s8_f16:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_f16(float16x4_t a) {
  return vreinterpret_s8_f16(a);
}

// CHECK-LABEL: test_vreinterpret_s8_f32:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_f32(float32x2_t a) {
  return vreinterpret_s8_f32(a);
}

// CHECK-LABEL: test_vreinterpret_s8_f64:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_f64(float64x1_t a) {
  return vreinterpret_s8_f64(a);
}

// CHECK-LABEL: test_vreinterpret_s8_p8:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_p8(poly8x8_t a) {
  return vreinterpret_s8_p8(a);
}

// CHECK-LABEL: test_vreinterpret_s8_p16:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_p16(poly16x4_t a) {
  return vreinterpret_s8_p16(a);
}

// CHECK-LABEL: test_vreinterpret_s8_p64:
// CHECK-NEXT: ret
int8x8_t test_vreinterpret_s8_p64(poly64x1_t a) {
  return vreinterpret_s8_p64(a);
}

// CHECK-LABEL: test_vreinterpret_s16_s8:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_s8(int8x8_t a) {
  return vreinterpret_s16_s8(a);
}

// CHECK-LABEL: test_vreinterpret_s16_s32:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_s32(int32x2_t a) {
  return vreinterpret_s16_s32(a);
}

// CHECK-LABEL: test_vreinterpret_s16_s64:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_s64(int64x1_t a) {
  return vreinterpret_s16_s64(a);
}

// CHECK-LABEL: test_vreinterpret_s16_u8:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_u8(uint8x8_t a) {
  return vreinterpret_s16_u8(a);
}

// CHECK-LABEL: test_vreinterpret_s16_u16:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_u16(uint16x4_t a) {
  return vreinterpret_s16_u16(a);
}

// CHECK-LABEL: test_vreinterpret_s16_u32:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_u32(uint32x2_t a) {
  return vreinterpret_s16_u32(a);
}

// CHECK-LABEL: test_vreinterpret_s16_u64:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_u64(uint64x1_t a) {
  return vreinterpret_s16_u64(a);
}

// CHECK-LABEL: test_vreinterpret_s16_f16:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_f16(float16x4_t a) {
  return vreinterpret_s16_f16(a);
}

// CHECK-LABEL: test_vreinterpret_s16_f32:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_f32(float32x2_t a) {
  return vreinterpret_s16_f32(a);
}

// CHECK-LABEL: test_vreinterpret_s16_f64:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_f64(float64x1_t a) {
  return vreinterpret_s16_f64(a);
}

// CHECK-LABEL: test_vreinterpret_s16_p8:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_p8(poly8x8_t a) {
  return vreinterpret_s16_p8(a);
}

// CHECK-LABEL: test_vreinterpret_s16_p16:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_p16(poly16x4_t a) {
  return vreinterpret_s16_p16(a);
}

// CHECK-LABEL: test_vreinterpret_s16_p64:
// CHECK-NEXT: ret
int16x4_t test_vreinterpret_s16_p64(poly64x1_t a) {
  return vreinterpret_s16_p64(a);
}

// CHECK-LABEL: test_vreinterpret_s32_s8:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_s8(int8x8_t a) {
  return vreinterpret_s32_s8(a);
}

// CHECK-LABEL: test_vreinterpret_s32_s16:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_s16(int16x4_t a) {
  return vreinterpret_s32_s16(a);
}

// CHECK-LABEL: test_vreinterpret_s32_s64:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_s64(int64x1_t a) {
  return vreinterpret_s32_s64(a);
}

// CHECK-LABEL: test_vreinterpret_s32_u8:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_u8(uint8x8_t a) {
  return vreinterpret_s32_u8(a);
}

// CHECK-LABEL: test_vreinterpret_s32_u16:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_u16(uint16x4_t a) {
  return vreinterpret_s32_u16(a);
}

// CHECK-LABEL: test_vreinterpret_s32_u32:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_u32(uint32x2_t a) {
  return vreinterpret_s32_u32(a);
}

// CHECK-LABEL: test_vreinterpret_s32_u64:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_u64(uint64x1_t a) {
  return vreinterpret_s32_u64(a);
}

// CHECK-LABEL: test_vreinterpret_s32_f16:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_f16(float16x4_t a) {
  return vreinterpret_s32_f16(a);
}

// CHECK-LABEL: test_vreinterpret_s32_f32:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_f32(float32x2_t a) {
  return vreinterpret_s32_f32(a);
}

// CHECK-LABEL: test_vreinterpret_s32_f64:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_f64(float64x1_t a) {
  return vreinterpret_s32_f64(a);
}

// CHECK-LABEL: test_vreinterpret_s32_p8:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_p8(poly8x8_t a) {
  return vreinterpret_s32_p8(a);
}

// CHECK-LABEL: test_vreinterpret_s32_p16:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_p16(poly16x4_t a) {
  return vreinterpret_s32_p16(a);
}

// CHECK-LABEL: test_vreinterpret_s32_p64:
// CHECK-NEXT: ret
int32x2_t test_vreinterpret_s32_p64(poly64x1_t a) {
  return vreinterpret_s32_p64(a);
}

// CHECK-LABEL: test_vreinterpret_s64_s8:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_s8(int8x8_t a) {
  return vreinterpret_s64_s8(a);
}

// CHECK-LABEL: test_vreinterpret_s64_s16:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_s16(int16x4_t a) {
  return vreinterpret_s64_s16(a);
}

// CHECK-LABEL: test_vreinterpret_s64_s32:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_s32(int32x2_t a) {
  return vreinterpret_s64_s32(a);
}

// CHECK-LABEL: test_vreinterpret_s64_u8:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_u8(uint8x8_t a) {
  return vreinterpret_s64_u8(a);
}

// CHECK-LABEL: test_vreinterpret_s64_u16:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_u16(uint16x4_t a) {
  return vreinterpret_s64_u16(a);
}

// CHECK-LABEL: test_vreinterpret_s64_u32:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_u32(uint32x2_t a) {
  return vreinterpret_s64_u32(a);
}

// CHECK-LABEL: test_vreinterpret_s64_u64:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_u64(uint64x1_t a) {
  return vreinterpret_s64_u64(a);
}

// CHECK-LABEL: test_vreinterpret_s64_f16:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_f16(float16x4_t a) {
  return vreinterpret_s64_f16(a);
}

// CHECK-LABEL: test_vreinterpret_s64_f32:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_f32(float32x2_t a) {
  return vreinterpret_s64_f32(a);
}

// CHECK-LABEL: test_vreinterpret_s64_f64:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_f64(float64x1_t a) {
  return vreinterpret_s64_f64(a);
}

// CHECK-LABEL: test_vreinterpret_s64_p8:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_p8(poly8x8_t a) {
  return vreinterpret_s64_p8(a);
}

// CHECK-LABEL: test_vreinterpret_s64_p16:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_p16(poly16x4_t a) {
  return vreinterpret_s64_p16(a);
}

// CHECK-LABEL: test_vreinterpret_s64_p64:
// CHECK-NEXT: ret
int64x1_t test_vreinterpret_s64_p64(poly64x1_t a) {
  return vreinterpret_s64_p64(a);
}

// CHECK-LABEL: test_vreinterpret_u8_s8:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_s8(int8x8_t a) {
  return vreinterpret_u8_s8(a);
}

// CHECK-LABEL: test_vreinterpret_u8_s16:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_s16(int16x4_t a) {
  return vreinterpret_u8_s16(a);
}

// CHECK-LABEL: test_vreinterpret_u8_s32:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_s32(int32x2_t a) {
  return vreinterpret_u8_s32(a);
}

// CHECK-LABEL: test_vreinterpret_u8_s64:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_s64(int64x1_t a) {
  return vreinterpret_u8_s64(a);
}

// CHECK-LABEL: test_vreinterpret_u8_u16:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_u16(uint16x4_t a) {
  return vreinterpret_u8_u16(a);
}

// CHECK-LABEL: test_vreinterpret_u8_u32:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_u32(uint32x2_t a) {
  return vreinterpret_u8_u32(a);
}

// CHECK-LABEL: test_vreinterpret_u8_u64:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_u64(uint64x1_t a) {
  return vreinterpret_u8_u64(a);
}

// CHECK-LABEL: test_vreinterpret_u8_f16:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_f16(float16x4_t a) {
  return vreinterpret_u8_f16(a);
}

// CHECK-LABEL: test_vreinterpret_u8_f32:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_f32(float32x2_t a) {
  return vreinterpret_u8_f32(a);
}

// CHECK-LABEL: test_vreinterpret_u8_f64:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_f64(float64x1_t a) {
  return vreinterpret_u8_f64(a);
}

// CHECK-LABEL: test_vreinterpret_u8_p8:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_p8(poly8x8_t a) {
  return vreinterpret_u8_p8(a);
}

// CHECK-LABEL: test_vreinterpret_u8_p16:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_p16(poly16x4_t a) {
  return vreinterpret_u8_p16(a);
}

// CHECK-LABEL: test_vreinterpret_u8_p64:
// CHECK-NEXT: ret
uint8x8_t test_vreinterpret_u8_p64(poly64x1_t a) {
  return vreinterpret_u8_p64(a);
}

// CHECK-LABEL: test_vreinterpret_u16_s8:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_s8(int8x8_t a) {
  return vreinterpret_u16_s8(a);
}

// CHECK-LABEL: test_vreinterpret_u16_s16:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_s16(int16x4_t a) {
  return vreinterpret_u16_s16(a);
}

// CHECK-LABEL: test_vreinterpret_u16_s32:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_s32(int32x2_t a) {
  return vreinterpret_u16_s32(a);
}

// CHECK-LABEL: test_vreinterpret_u16_s64:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_s64(int64x1_t a) {
  return vreinterpret_u16_s64(a);
}

// CHECK-LABEL: test_vreinterpret_u16_u8:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_u8(uint8x8_t a) {
  return vreinterpret_u16_u8(a);
}

// CHECK-LABEL: test_vreinterpret_u16_u32:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_u32(uint32x2_t a) {
  return vreinterpret_u16_u32(a);
}

// CHECK-LABEL: test_vreinterpret_u16_u64:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_u64(uint64x1_t a) {
  return vreinterpret_u16_u64(a);
}

// CHECK-LABEL: test_vreinterpret_u16_f16:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_f16(float16x4_t a) {
  return vreinterpret_u16_f16(a);
}

// CHECK-LABEL: test_vreinterpret_u16_f32:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_f32(float32x2_t a) {
  return vreinterpret_u16_f32(a);
}

// CHECK-LABEL: test_vreinterpret_u16_f64:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_f64(float64x1_t a) {
  return vreinterpret_u16_f64(a);
}

// CHECK-LABEL: test_vreinterpret_u16_p8:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_p8(poly8x8_t a) {
  return vreinterpret_u16_p8(a);
}

// CHECK-LABEL: test_vreinterpret_u16_p16:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_p16(poly16x4_t a) {
  return vreinterpret_u16_p16(a);
}

// CHECK-LABEL: test_vreinterpret_u16_p64:
// CHECK-NEXT: ret
uint16x4_t test_vreinterpret_u16_p64(poly64x1_t a) {
  return vreinterpret_u16_p64(a);
}

// CHECK-LABEL: test_vreinterpret_u32_s8:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_s8(int8x8_t a) {
  return vreinterpret_u32_s8(a);
}

// CHECK-LABEL: test_vreinterpret_u32_s16:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_s16(int16x4_t a) {
  return vreinterpret_u32_s16(a);
}

// CHECK-LABEL: test_vreinterpret_u32_s32:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_s32(int32x2_t a) {
  return vreinterpret_u32_s32(a);
}

// CHECK-LABEL: test_vreinterpret_u32_s64:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_s64(int64x1_t a) {
  return vreinterpret_u32_s64(a);
}

// CHECK-LABEL: test_vreinterpret_u32_u8:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_u8(uint8x8_t a) {
  return vreinterpret_u32_u8(a);
}

// CHECK-LABEL: test_vreinterpret_u32_u16:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_u16(uint16x4_t a) {
  return vreinterpret_u32_u16(a);
}

// CHECK-LABEL: test_vreinterpret_u32_u64:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_u64(uint64x1_t a) {
  return vreinterpret_u32_u64(a);
}

// CHECK-LABEL: test_vreinterpret_u32_f16:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_f16(float16x4_t a) {
  return vreinterpret_u32_f16(a);
}

// CHECK-LABEL: test_vreinterpret_u32_f32:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_f32(float32x2_t a) {
  return vreinterpret_u32_f32(a);
}

// CHECK-LABEL: test_vreinterpret_u32_f64:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_f64(float64x1_t a) {
  return vreinterpret_u32_f64(a);
}

// CHECK-LABEL: test_vreinterpret_u32_p8:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_p8(poly8x8_t a) {
  return vreinterpret_u32_p8(a);
}

// CHECK-LABEL: test_vreinterpret_u32_p16:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_p16(poly16x4_t a) {
  return vreinterpret_u32_p16(a);
}

// CHECK-LABEL: test_vreinterpret_u32_p64:
// CHECK-NEXT: ret
uint32x2_t test_vreinterpret_u32_p64(poly64x1_t a) {
  return vreinterpret_u32_p64(a);
}

// CHECK-LABEL: test_vreinterpret_u64_s8:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_s8(int8x8_t a) {
  return vreinterpret_u64_s8(a);
}

// CHECK-LABEL: test_vreinterpret_u64_s16:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_s16(int16x4_t a) {
  return vreinterpret_u64_s16(a);
}

// CHECK-LABEL: test_vreinterpret_u64_s32:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_s32(int32x2_t a) {
  return vreinterpret_u64_s32(a);
}

// CHECK-LABEL: test_vreinterpret_u64_s64:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_s64(int64x1_t a) {
  return vreinterpret_u64_s64(a);
}

// CHECK-LABEL: test_vreinterpret_u64_u8:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_u8(uint8x8_t a) {
  return vreinterpret_u64_u8(a);
}

// CHECK-LABEL: test_vreinterpret_u64_u16:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_u16(uint16x4_t a) {
  return vreinterpret_u64_u16(a);
}

// CHECK-LABEL: test_vreinterpret_u64_u32:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_u32(uint32x2_t a) {
  return vreinterpret_u64_u32(a);
}

// CHECK-LABEL: test_vreinterpret_u64_f16:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_f16(float16x4_t a) {
  return vreinterpret_u64_f16(a);
}

// CHECK-LABEL: test_vreinterpret_u64_f32:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_f32(float32x2_t a) {
  return vreinterpret_u64_f32(a);
}

// CHECK-LABEL: test_vreinterpret_u64_f64:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_f64(float64x1_t a) {
  return vreinterpret_u64_f64(a);
}

// CHECK-LABEL: test_vreinterpret_u64_p8:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_p8(poly8x8_t a) {
  return vreinterpret_u64_p8(a);
}

// CHECK-LABEL: test_vreinterpret_u64_p16:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_p16(poly16x4_t a) {
  return vreinterpret_u64_p16(a);
}

// CHECK-LABEL: test_vreinterpret_u64_p64:
// CHECK-NEXT: ret
uint64x1_t test_vreinterpret_u64_p64(poly64x1_t a) {
  return vreinterpret_u64_p64(a);
}

// CHECK-LABEL: test_vreinterpret_f16_s8:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_s8(int8x8_t a) {
  return vreinterpret_f16_s8(a);
}

// CHECK-LABEL: test_vreinterpret_f16_s16:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_s16(int16x4_t a) {
  return vreinterpret_f16_s16(a);
}

// CHECK-LABEL: test_vreinterpret_f16_s32:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_s32(int32x2_t a) {
  return vreinterpret_f16_s32(a);
}

// CHECK-LABEL: test_vreinterpret_f16_s64:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_s64(int64x1_t a) {
  return vreinterpret_f16_s64(a);
}

// CHECK-LABEL: test_vreinterpret_f16_u8:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_u8(uint8x8_t a) {
  return vreinterpret_f16_u8(a);
}

// CHECK-LABEL: test_vreinterpret_f16_u16:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_u16(uint16x4_t a) {
  return vreinterpret_f16_u16(a);
}

// CHECK-LABEL: test_vreinterpret_f16_u32:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_u32(uint32x2_t a) {
  return vreinterpret_f16_u32(a);
}

// CHECK-LABEL: test_vreinterpret_f16_u64:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_u64(uint64x1_t a) {
  return vreinterpret_f16_u64(a);
}

// CHECK-LABEL: test_vreinterpret_f16_f32:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_f32(float32x2_t a) {
  return vreinterpret_f16_f32(a);
}

// CHECK-LABEL: test_vreinterpret_f16_f64:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_f64(float64x1_t a) {
  return vreinterpret_f16_f64(a);
}

// CHECK-LABEL: test_vreinterpret_f16_p8:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_p8(poly8x8_t a) {
  return vreinterpret_f16_p8(a);
}

// CHECK-LABEL: test_vreinterpret_f16_p16:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_p16(poly16x4_t a) {
  return vreinterpret_f16_p16(a);
}

// CHECK-LABEL: test_vreinterpret_f16_p64:
// CHECK-NEXT: ret
float16x4_t test_vreinterpret_f16_p64(poly64x1_t a) {
  return vreinterpret_f16_p64(a);
}

// CHECK-LABEL: test_vreinterpret_f32_s8:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_s8(int8x8_t a) {
  return vreinterpret_f32_s8(a);
}

// CHECK-LABEL: test_vreinterpret_f32_s16:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_s16(int16x4_t a) {
  return vreinterpret_f32_s16(a);
}

// CHECK-LABEL: test_vreinterpret_f32_s32:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_s32(int32x2_t a) {
  return vreinterpret_f32_s32(a);
}

// CHECK-LABEL: test_vreinterpret_f32_s64:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_s64(int64x1_t a) {
  return vreinterpret_f32_s64(a);
}

// CHECK-LABEL: test_vreinterpret_f32_u8:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_u8(uint8x8_t a) {
  return vreinterpret_f32_u8(a);
}

// CHECK-LABEL: test_vreinterpret_f32_u16:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_u16(uint16x4_t a) {
  return vreinterpret_f32_u16(a);
}

// CHECK-LABEL: test_vreinterpret_f32_u32:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_u32(uint32x2_t a) {
  return vreinterpret_f32_u32(a);
}

// CHECK-LABEL: test_vreinterpret_f32_u64:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_u64(uint64x1_t a) {
  return vreinterpret_f32_u64(a);
}

// CHECK-LABEL: test_vreinterpret_f32_f16:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_f16(float16x4_t a) {
  return vreinterpret_f32_f16(a);
}

// CHECK-LABEL: test_vreinterpret_f32_f64:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_f64(float64x1_t a) {
  return vreinterpret_f32_f64(a);
}

// CHECK-LABEL: test_vreinterpret_f32_p8:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_p8(poly8x8_t a) {
  return vreinterpret_f32_p8(a);
}

// CHECK-LABEL: test_vreinterpret_f32_p16:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_p16(poly16x4_t a) {
  return vreinterpret_f32_p16(a);
}

// CHECK-LABEL: test_vreinterpret_f32_p64:
// CHECK-NEXT: ret
float32x2_t test_vreinterpret_f32_p64(poly64x1_t a) {
  return vreinterpret_f32_p64(a);
}

// CHECK-LABEL: test_vreinterpret_f64_s8:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_s8(int8x8_t a) {
  return vreinterpret_f64_s8(a);
}

// CHECK-LABEL: test_vreinterpret_f64_s16:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_s16(int16x4_t a) {
  return vreinterpret_f64_s16(a);
}

// CHECK-LABEL: test_vreinterpret_f64_s32:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_s32(int32x2_t a) {
  return vreinterpret_f64_s32(a);
}

// CHECK-LABEL: test_vreinterpret_f64_s64:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_s64(int64x1_t a) {
  return vreinterpret_f64_s64(a);
}

// CHECK-LABEL: test_vreinterpret_f64_u8:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_u8(uint8x8_t a) {
  return vreinterpret_f64_u8(a);
}

// CHECK-LABEL: test_vreinterpret_f64_u16:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_u16(uint16x4_t a) {
  return vreinterpret_f64_u16(a);
}

// CHECK-LABEL: test_vreinterpret_f64_u32:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_u32(uint32x2_t a) {
  return vreinterpret_f64_u32(a);
}

// CHECK-LABEL: test_vreinterpret_f64_u64:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_u64(uint64x1_t a) {
  return vreinterpret_f64_u64(a);
}

// CHECK-LABEL: test_vreinterpret_f64_f16:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_f16(float16x4_t a) {
  return vreinterpret_f64_f16(a);
}

// CHECK-LABEL: test_vreinterpret_f64_f32:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_f32(float32x2_t a) {
  return vreinterpret_f64_f32(a);
}

// CHECK-LABEL: test_vreinterpret_f64_p8:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_p8(poly8x8_t a) {
  return vreinterpret_f64_p8(a);
}

// CHECK-LABEL: test_vreinterpret_f64_p16:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_p16(poly16x4_t a) {
  return vreinterpret_f64_p16(a);
}

// CHECK-LABEL: test_vreinterpret_f64_p64:
// CHECK-NEXT: ret
float64x1_t test_vreinterpret_f64_p64(poly64x1_t a) {
  return vreinterpret_f64_p64(a);
}

// CHECK-LABEL: test_vreinterpret_p8_s8:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_s8(int8x8_t a) {
  return vreinterpret_p8_s8(a);
}

// CHECK-LABEL: test_vreinterpret_p8_s16:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_s16(int16x4_t a) {
  return vreinterpret_p8_s16(a);
}

// CHECK-LABEL: test_vreinterpret_p8_s32:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_s32(int32x2_t a) {
  return vreinterpret_p8_s32(a);
}

// CHECK-LABEL: test_vreinterpret_p8_s64:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_s64(int64x1_t a) {
  return vreinterpret_p8_s64(a);
}

// CHECK-LABEL: test_vreinterpret_p8_u8:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_u8(uint8x8_t a) {
  return vreinterpret_p8_u8(a);
}

// CHECK-LABEL: test_vreinterpret_p8_u16:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_u16(uint16x4_t a) {
  return vreinterpret_p8_u16(a);
}

// CHECK-LABEL: test_vreinterpret_p8_u32:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_u32(uint32x2_t a) {
  return vreinterpret_p8_u32(a);
}

// CHECK-LABEL: test_vreinterpret_p8_u64:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_u64(uint64x1_t a) {
  return vreinterpret_p8_u64(a);
}

// CHECK-LABEL: test_vreinterpret_p8_f16:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_f16(float16x4_t a) {
  return vreinterpret_p8_f16(a);
}

// CHECK-LABEL: test_vreinterpret_p8_f32:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_f32(float32x2_t a) {
  return vreinterpret_p8_f32(a);
}

// CHECK-LABEL: test_vreinterpret_p8_f64:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_f64(float64x1_t a) {
  return vreinterpret_p8_f64(a);
}

// CHECK-LABEL: test_vreinterpret_p8_p16:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_p16(poly16x4_t a) {
  return vreinterpret_p8_p16(a);
}

// CHECK-LABEL: test_vreinterpret_p8_p64:
// CHECK-NEXT: ret
poly8x8_t test_vreinterpret_p8_p64(poly64x1_t a) {
  return vreinterpret_p8_p64(a);
}

// CHECK-LABEL: test_vreinterpret_p16_s8:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_s8(int8x8_t a) {
  return vreinterpret_p16_s8(a);
}

// CHECK-LABEL: test_vreinterpret_p16_s16:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_s16(int16x4_t a) {
  return vreinterpret_p16_s16(a);
}

// CHECK-LABEL: test_vreinterpret_p16_s32:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_s32(int32x2_t a) {
  return vreinterpret_p16_s32(a);
}

// CHECK-LABEL: test_vreinterpret_p16_s64:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_s64(int64x1_t a) {
  return vreinterpret_p16_s64(a);
}

// CHECK-LABEL: test_vreinterpret_p16_u8:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_u8(uint8x8_t a) {
  return vreinterpret_p16_u8(a);
}

// CHECK-LABEL: test_vreinterpret_p16_u16:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_u16(uint16x4_t a) {
  return vreinterpret_p16_u16(a);
}

// CHECK-LABEL: test_vreinterpret_p16_u32:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_u32(uint32x2_t a) {
  return vreinterpret_p16_u32(a);
}

// CHECK-LABEL: test_vreinterpret_p16_u64:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_u64(uint64x1_t a) {
  return vreinterpret_p16_u64(a);
}

// CHECK-LABEL: test_vreinterpret_p16_f16:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_f16(float16x4_t a) {
  return vreinterpret_p16_f16(a);
}

// CHECK-LABEL: test_vreinterpret_p16_f32:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_f32(float32x2_t a) {
  return vreinterpret_p16_f32(a);
}

// CHECK-LABEL: test_vreinterpret_p16_f64:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_f64(float64x1_t a) {
  return vreinterpret_p16_f64(a);
}

// CHECK-LABEL: test_vreinterpret_p16_p8:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_p8(poly8x8_t a) {
  return vreinterpret_p16_p8(a);
}

// CHECK-LABEL: test_vreinterpret_p16_p64:
// CHECK-NEXT: ret
poly16x4_t test_vreinterpret_p16_p64(poly64x1_t a) {
  return vreinterpret_p16_p64(a);
}

// CHECK-LABEL: test_vreinterpret_p64_s8:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_s8(int8x8_t a) {
  return vreinterpret_p64_s8(a);
}

// CHECK-LABEL: test_vreinterpret_p64_s16:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_s16(int16x4_t a) {
  return vreinterpret_p64_s16(a);
}

// CHECK-LABEL: test_vreinterpret_p64_s32:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_s32(int32x2_t a) {
  return vreinterpret_p64_s32(a);
}

// CHECK-LABEL: test_vreinterpret_p64_s64:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_s64(int64x1_t a) {
  return vreinterpret_p64_s64(a);
}

// CHECK-LABEL: test_vreinterpret_p64_u8:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_u8(uint8x8_t a) {
  return vreinterpret_p64_u8(a);
}

// CHECK-LABEL: test_vreinterpret_p64_u16:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_u16(uint16x4_t a) {
  return vreinterpret_p64_u16(a);
}

// CHECK-LABEL: test_vreinterpret_p64_u32:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_u32(uint32x2_t a) {
  return vreinterpret_p64_u32(a);
}

// CHECK-LABEL: test_vreinterpret_p64_u64:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_u64(uint64x1_t a) {
  return vreinterpret_p64_u64(a);
}

// CHECK-LABEL: test_vreinterpret_p64_f16:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_f16(float16x4_t a) {
  return vreinterpret_p64_f16(a);
}

// CHECK-LABEL: test_vreinterpret_p64_f32:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_f32(float32x2_t a) {
  return vreinterpret_p64_f32(a);
}

// CHECK-LABEL: test_vreinterpret_p64_f64:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_f64(float64x1_t a) {
  return vreinterpret_p64_f64(a);
}

// CHECK-LABEL: test_vreinterpret_p64_p8:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_p8(poly8x8_t a) {
  return vreinterpret_p64_p8(a);
}

// CHECK-LABEL: test_vreinterpret_p64_p16:
// CHECK-NEXT: ret
poly64x1_t test_vreinterpret_p64_p16(poly16x4_t a) {
  return vreinterpret_p64_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_s16:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_s16(int16x8_t a) {
  return vreinterpretq_s8_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_s32:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_s32(int32x4_t a) {
  return vreinterpretq_s8_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_s64:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_s64(int64x2_t a) {
  return vreinterpretq_s8_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_u8:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_u8(uint8x16_t a) {
  return vreinterpretq_s8_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_u16:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_u16(uint16x8_t a) {
  return vreinterpretq_s8_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_u32:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_u32(uint32x4_t a) {
  return vreinterpretq_s8_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_u64:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_u64(uint64x2_t a) {
  return vreinterpretq_s8_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_f16:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_f16(float16x8_t a) {
  return vreinterpretq_s8_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_f32:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_f32(float32x4_t a) {
  return vreinterpretq_s8_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_f64:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_f64(float64x2_t a) {
  return vreinterpretq_s8_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_p8:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_p8(poly8x16_t a) {
  return vreinterpretq_s8_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_p16:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_p16(poly16x8_t a) {
  return vreinterpretq_s8_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_p64:
// CHECK-NEXT: ret
int8x16_t test_vreinterpretq_s8_p64(poly64x2_t a) {
  return vreinterpretq_s8_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_s8:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_s8(int8x16_t a) {
  return vreinterpretq_s16_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_s32:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_s32(int32x4_t a) {
  return vreinterpretq_s16_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_s64:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_s64(int64x2_t a) {
  return vreinterpretq_s16_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_u8:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_u8(uint8x16_t a) {
  return vreinterpretq_s16_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_u16:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_u16(uint16x8_t a) {
  return vreinterpretq_s16_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_u32:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_u32(uint32x4_t a) {
  return vreinterpretq_s16_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_u64:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_u64(uint64x2_t a) {
  return vreinterpretq_s16_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_f16:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_f16(float16x8_t a) {
  return vreinterpretq_s16_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_f32:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_f32(float32x4_t a) {
  return vreinterpretq_s16_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_f64:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_f64(float64x2_t a) {
  return vreinterpretq_s16_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_p8:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_p8(poly8x16_t a) {
  return vreinterpretq_s16_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_p16:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_p16(poly16x8_t a) {
  return vreinterpretq_s16_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_p64:
// CHECK-NEXT: ret
int16x8_t test_vreinterpretq_s16_p64(poly64x2_t a) {
  return vreinterpretq_s16_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_s8:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_s8(int8x16_t a) {
  return vreinterpretq_s32_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_s16:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_s16(int16x8_t a) {
  return vreinterpretq_s32_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_s64:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_s64(int64x2_t a) {
  return vreinterpretq_s32_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_u8:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_u8(uint8x16_t a) {
  return vreinterpretq_s32_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_u16:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_u16(uint16x8_t a) {
  return vreinterpretq_s32_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_u32:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_u32(uint32x4_t a) {
  return vreinterpretq_s32_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_u64:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_u64(uint64x2_t a) {
  return vreinterpretq_s32_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_f16:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_f16(float16x8_t a) {
  return vreinterpretq_s32_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_f32:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_f32(float32x4_t a) {
  return vreinterpretq_s32_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_f64:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_f64(float64x2_t a) {
  return vreinterpretq_s32_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_p8:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_p8(poly8x16_t a) {
  return vreinterpretq_s32_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_p16:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_p16(poly16x8_t a) {
  return vreinterpretq_s32_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_p64:
// CHECK-NEXT: ret
int32x4_t test_vreinterpretq_s32_p64(poly64x2_t a) {
  return vreinterpretq_s32_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_s8:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_s8(int8x16_t a) {
  return vreinterpretq_s64_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_s16:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_s16(int16x8_t a) {
  return vreinterpretq_s64_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_s32:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_s32(int32x4_t a) {
  return vreinterpretq_s64_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_u8:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_u8(uint8x16_t a) {
  return vreinterpretq_s64_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_u16:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_u16(uint16x8_t a) {
  return vreinterpretq_s64_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_u32:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_u32(uint32x4_t a) {
  return vreinterpretq_s64_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_u64:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_u64(uint64x2_t a) {
  return vreinterpretq_s64_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_f16:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_f16(float16x8_t a) {
  return vreinterpretq_s64_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_f32:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_f32(float32x4_t a) {
  return vreinterpretq_s64_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_f64:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_f64(float64x2_t a) {
  return vreinterpretq_s64_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_p8:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_p8(poly8x16_t a) {
  return vreinterpretq_s64_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_p16:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_p16(poly16x8_t a) {
  return vreinterpretq_s64_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_p64:
// CHECK-NEXT: ret
int64x2_t test_vreinterpretq_s64_p64(poly64x2_t a) {
  return vreinterpretq_s64_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_s8:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_s8(int8x16_t a) {
  return vreinterpretq_u8_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_s16:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_s16(int16x8_t a) {
  return vreinterpretq_u8_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_s32:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_s32(int32x4_t a) {
  return vreinterpretq_u8_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_s64:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_s64(int64x2_t a) {
  return vreinterpretq_u8_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_u16:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_u16(uint16x8_t a) {
  return vreinterpretq_u8_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_u32:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_u32(uint32x4_t a) {
  return vreinterpretq_u8_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_u64:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_u64(uint64x2_t a) {
  return vreinterpretq_u8_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_f16:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_f16(float16x8_t a) {
  return vreinterpretq_u8_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_f32:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_f32(float32x4_t a) {
  return vreinterpretq_u8_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_f64:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_f64(float64x2_t a) {
  return vreinterpretq_u8_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_p8:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_p8(poly8x16_t a) {
  return vreinterpretq_u8_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_p16:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_p16(poly16x8_t a) {
  return vreinterpretq_u8_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_p64:
// CHECK-NEXT: ret
uint8x16_t test_vreinterpretq_u8_p64(poly64x2_t a) {
  return vreinterpretq_u8_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_s8:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_s8(int8x16_t a) {
  return vreinterpretq_u16_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_s16:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_s16(int16x8_t a) {
  return vreinterpretq_u16_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_s32:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_s32(int32x4_t a) {
  return vreinterpretq_u16_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_s64:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_s64(int64x2_t a) {
  return vreinterpretq_u16_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_u8:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_u8(uint8x16_t a) {
  return vreinterpretq_u16_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_u32:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_u32(uint32x4_t a) {
  return vreinterpretq_u16_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_u64:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_u64(uint64x2_t a) {
  return vreinterpretq_u16_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_f16:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_f16(float16x8_t a) {
  return vreinterpretq_u16_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_f32:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_f32(float32x4_t a) {
  return vreinterpretq_u16_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_f64:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_f64(float64x2_t a) {
  return vreinterpretq_u16_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_p8:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_p8(poly8x16_t a) {
  return vreinterpretq_u16_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_p16:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_p16(poly16x8_t a) {
  return vreinterpretq_u16_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_p64:
// CHECK-NEXT: ret
uint16x8_t test_vreinterpretq_u16_p64(poly64x2_t a) {
  return vreinterpretq_u16_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_s8:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_s8(int8x16_t a) {
  return vreinterpretq_u32_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_s16:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_s16(int16x8_t a) {
  return vreinterpretq_u32_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_s32:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_s32(int32x4_t a) {
  return vreinterpretq_u32_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_s64:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_s64(int64x2_t a) {
  return vreinterpretq_u32_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_u8:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_u8(uint8x16_t a) {
  return vreinterpretq_u32_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_u16:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_u16(uint16x8_t a) {
  return vreinterpretq_u32_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_u64:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_u64(uint64x2_t a) {
  return vreinterpretq_u32_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_f16:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_f16(float16x8_t a) {
  return vreinterpretq_u32_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_f32:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_f32(float32x4_t a) {
  return vreinterpretq_u32_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_f64:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_f64(float64x2_t a) {
  return vreinterpretq_u32_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_p8:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_p8(poly8x16_t a) {
  return vreinterpretq_u32_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_p16:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_p16(poly16x8_t a) {
  return vreinterpretq_u32_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_p64:
// CHECK-NEXT: ret
uint32x4_t test_vreinterpretq_u32_p64(poly64x2_t a) {
  return vreinterpretq_u32_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_s8:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_s8(int8x16_t a) {
  return vreinterpretq_u64_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_s16:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_s16(int16x8_t a) {
  return vreinterpretq_u64_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_s32:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_s32(int32x4_t a) {
  return vreinterpretq_u64_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_s64:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_s64(int64x2_t a) {
  return vreinterpretq_u64_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_u8:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_u8(uint8x16_t a) {
  return vreinterpretq_u64_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_u16:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_u16(uint16x8_t a) {
  return vreinterpretq_u64_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_u32:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_u32(uint32x4_t a) {
  return vreinterpretq_u64_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_f16:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_f16(float16x8_t a) {
  return vreinterpretq_u64_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_f32:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_f32(float32x4_t a) {
  return vreinterpretq_u64_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_f64:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_f64(float64x2_t a) {
  return vreinterpretq_u64_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_p8:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_p8(poly8x16_t a) {
  return vreinterpretq_u64_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_p16:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_p16(poly16x8_t a) {
  return vreinterpretq_u64_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_p64:
// CHECK-NEXT: ret
uint64x2_t test_vreinterpretq_u64_p64(poly64x2_t a) {
  return vreinterpretq_u64_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_s8:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_s8(int8x16_t a) {
  return vreinterpretq_f16_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_s16:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_s16(int16x8_t a) {
  return vreinterpretq_f16_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_s32:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_s32(int32x4_t a) {
  return vreinterpretq_f16_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_s64:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_s64(int64x2_t a) {
  return vreinterpretq_f16_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_u8:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_u8(uint8x16_t a) {
  return vreinterpretq_f16_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_u16:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_u16(uint16x8_t a) {
  return vreinterpretq_f16_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_u32:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_u32(uint32x4_t a) {
  return vreinterpretq_f16_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_u64:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_u64(uint64x2_t a) {
  return vreinterpretq_f16_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_f32:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_f32(float32x4_t a) {
  return vreinterpretq_f16_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_f64:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_f64(float64x2_t a) {
  return vreinterpretq_f16_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_p8:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_p8(poly8x16_t a) {
  return vreinterpretq_f16_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_p16:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_p16(poly16x8_t a) {
  return vreinterpretq_f16_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_p64:
// CHECK-NEXT: ret
float16x8_t test_vreinterpretq_f16_p64(poly64x2_t a) {
  return vreinterpretq_f16_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_s8:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_s8(int8x16_t a) {
  return vreinterpretq_f32_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_s16:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_s16(int16x8_t a) {
  return vreinterpretq_f32_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_s32:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_s32(int32x4_t a) {
  return vreinterpretq_f32_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_s64:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_s64(int64x2_t a) {
  return vreinterpretq_f32_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_u8:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_u8(uint8x16_t a) {
  return vreinterpretq_f32_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_u16:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_u16(uint16x8_t a) {
  return vreinterpretq_f32_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_u32:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_u32(uint32x4_t a) {
  return vreinterpretq_f32_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_u64:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_u64(uint64x2_t a) {
  return vreinterpretq_f32_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_f16:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_f16(float16x8_t a) {
  return vreinterpretq_f32_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_f64:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_f64(float64x2_t a) {
  return vreinterpretq_f32_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_p8:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_p8(poly8x16_t a) {
  return vreinterpretq_f32_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_p16:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_p16(poly16x8_t a) {
  return vreinterpretq_f32_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_p64:
// CHECK-NEXT: ret
float32x4_t test_vreinterpretq_f32_p64(poly64x2_t a) {
  return vreinterpretq_f32_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_s8:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_s8(int8x16_t a) {
  return vreinterpretq_f64_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_s16:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_s16(int16x8_t a) {
  return vreinterpretq_f64_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_s32:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_s32(int32x4_t a) {
  return vreinterpretq_f64_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_s64:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_s64(int64x2_t a) {
  return vreinterpretq_f64_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_u8:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_u8(uint8x16_t a) {
  return vreinterpretq_f64_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_u16:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_u16(uint16x8_t a) {
  return vreinterpretq_f64_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_u32:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_u32(uint32x4_t a) {
  return vreinterpretq_f64_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_u64:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_u64(uint64x2_t a) {
  return vreinterpretq_f64_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_f16:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_f16(float16x8_t a) {
  return vreinterpretq_f64_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_f32:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_f32(float32x4_t a) {
  return vreinterpretq_f64_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_p8:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_p8(poly8x16_t a) {
  return vreinterpretq_f64_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_p16:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_p16(poly16x8_t a) {
  return vreinterpretq_f64_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_p64:
// CHECK-NEXT: ret
float64x2_t test_vreinterpretq_f64_p64(poly64x2_t a) {
  return vreinterpretq_f64_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_s8:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_s8(int8x16_t a) {
  return vreinterpretq_p8_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_s16:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_s16(int16x8_t a) {
  return vreinterpretq_p8_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_s32:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_s32(int32x4_t a) {
  return vreinterpretq_p8_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_s64:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_s64(int64x2_t a) {
  return vreinterpretq_p8_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_u8:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_u8(uint8x16_t a) {
  return vreinterpretq_p8_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_u16:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_u16(uint16x8_t a) {
  return vreinterpretq_p8_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_u32:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_u32(uint32x4_t a) {
  return vreinterpretq_p8_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_u64:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_u64(uint64x2_t a) {
  return vreinterpretq_p8_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_f16:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_f16(float16x8_t a) {
  return vreinterpretq_p8_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_f32:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_f32(float32x4_t a) {
  return vreinterpretq_p8_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_f64:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_f64(float64x2_t a) {
  return vreinterpretq_p8_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_p16:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_p16(poly16x8_t a) {
  return vreinterpretq_p8_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_p64:
// CHECK-NEXT: ret
poly8x16_t test_vreinterpretq_p8_p64(poly64x2_t a) {
  return vreinterpretq_p8_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_s8:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_s8(int8x16_t a) {
  return vreinterpretq_p16_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_s16:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_s16(int16x8_t a) {
  return vreinterpretq_p16_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_s32:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_s32(int32x4_t a) {
  return vreinterpretq_p16_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_s64:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_s64(int64x2_t a) {
  return vreinterpretq_p16_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_u8:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_u8(uint8x16_t a) {
  return vreinterpretq_p16_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_u16:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_u16(uint16x8_t a) {
  return vreinterpretq_p16_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_u32:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_u32(uint32x4_t a) {
  return vreinterpretq_p16_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_u64:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_u64(uint64x2_t a) {
  return vreinterpretq_p16_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_f16:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_f16(float16x8_t a) {
  return vreinterpretq_p16_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_f32:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_f32(float32x4_t a) {
  return vreinterpretq_p16_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_f64:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_f64(float64x2_t a) {
  return vreinterpretq_p16_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_p8:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_p8(poly8x16_t a) {
  return vreinterpretq_p16_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_p64:
// CHECK-NEXT: ret
poly16x8_t test_vreinterpretq_p16_p64(poly64x2_t a) {
  return vreinterpretq_p16_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_s8:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_s8(int8x16_t a) {
  return vreinterpretq_p64_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_s16:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_s16(int16x8_t a) {
  return vreinterpretq_p64_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_s32:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_s32(int32x4_t a) {
  return vreinterpretq_p64_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_s64:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_s64(int64x2_t a) {
  return vreinterpretq_p64_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_u8:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_u8(uint8x16_t a) {
  return vreinterpretq_p64_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_u16:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_u16(uint16x8_t a) {
  return vreinterpretq_p64_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_u32:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_u32(uint32x4_t a) {
  return vreinterpretq_p64_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_u64:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_u64(uint64x2_t a) {
  return vreinterpretq_p64_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_f16:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_f16(float16x8_t a) {
  return vreinterpretq_p64_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_f32:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_f32(float32x4_t a) {
  return vreinterpretq_p64_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_f64:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_f64(float64x2_t a) {
  return vreinterpretq_p64_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_p8:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_p8(poly8x16_t a) {
  return vreinterpretq_p64_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_p16:
// CHECK-NEXT: ret
poly64x2_t test_vreinterpretq_p64_p16(poly16x8_t a) {
  return vreinterpretq_p64_p16(a);
}

float32_t test_vabds_f32(float32_t a, float32_t b) {
// CHECK-LABEL: test_vabds_f32
// CHECK: fabd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  return vabds_f32(a, b);
}

float64_t test_vabdd_f64(float64_t a, float64_t b) {
// CHECK-LABEL: test_vabdd_f64
// CHECK: fabd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  return vabdd_f64(a, b);
}

int64x1_t test_vuqadd_s64(int64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vuqadd_s64
  return vuqadd_s64(a, b);
  // CHECK: suqadd d{{[0-9]+}}, d{{[0-9]+}}
}

uint64x1_t test_vsqadd_u64(uint64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vsqadd_u64
  return vsqadd_u64(a, b);
  // CHECK: usqadd d{{[0-9]+}}, d{{[0-9]+}}
}

uint8x8_t test_vsqadd_u8(uint8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vsqadd_u8
  return vsqadd_u8(a, b);
  // CHECK: usqadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vsqaddq_u8(uint8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsqaddq_u8
  return vsqaddq_u8(a, b);
  // CHECK: usqadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vsqadd_u16(uint16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vsqadd_u16
  return vsqadd_u16(a, b);
  // CHECK: usqadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint16x8_t test_vsqaddq_u16(uint16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsqaddq_u16
  return vsqaddq_u16(a, b);
  // CHECK: usqadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x2_t test_vsqadd_u32(uint32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vsqadd_u32
  return vsqadd_u32(a, b);
  // CHECK: usqadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32x4_t test_vsqaddq_u32(uint32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsqaddq_u32
  return vsqaddq_u32(a, b);
  // CHECK: usqadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vsqaddq_u64(uint64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsqaddq_u64
  return vsqaddq_u64(a, b);
  // CHECK: usqadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int64x1_t test_vabs_s64(int64x1_t a) {
  // CHECK-LABEL: test_vabs_s64
  return vabs_s64(a);
  // CHECK: abs d{{[0-9]+}}, d{{[0-9]+}}
}

int64x1_t test_vqabs_s64(int64x1_t a) {
  // CHECK-LABEL: test_vqabs_s64
  return vqabs_s64(a);
  // CHECK: sqabs d{{[0-9]+}}, d{{[0-9]+}}
}

int64x1_t test_vqneg_s64(int64x1_t a) {
  // CHECK-LABEL: test_vqneg_s64
  return vqneg_s64(a);
  // CHECK: sqneg d{{[0-9]+}}, d{{[0-9]+}}
}

int64x1_t test_vneg_s64(int64x1_t a) {
  // CHECK-LABEL: test_vneg_s64
  return vneg_s64(a);
  // CHECK: neg d{{[0-9]+}}, d{{[0-9]+}}
}

float32_t test_vaddv_f32(float32x2_t a) {
  // CHECK-LABEL: test_vaddv_f32
  return vaddv_f32(a);
  // CHECK: faddp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

float32_t test_vaddvq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vaddvq_f32
  return vaddvq_f32(a);
  // CHECK: faddp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: faddp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

float64_t test_vaddvq_f64(float64x2_t a) {
  // CHECK-LABEL: test_vaddvq_f64
  return vaddvq_f64(a);
  // CHECK: faddp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

float32_t test_vmaxv_f32(float32x2_t a) {
  // CHECK-LABEL: test_vmaxv_f32
  return vmaxv_f32(a);
  // CHECK: fmaxp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

float64_t test_vmaxvq_f64(float64x2_t a) {
  // CHECK-LABEL: test_vmaxvq_f64
  return vmaxvq_f64(a);
  // CHECK: fmaxp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

float32_t test_vminv_f32(float32x2_t a) {
  // CHECK-LABEL: test_vminv_f32
  return vminv_f32(a);
  // CHECK: fminp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

float64_t test_vminvq_f64(float64x2_t a) {
  // CHECK-LABEL: test_vminvq_f64
  return vminvq_f64(a);
  // CHECK: fminp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

float64_t test_vmaxnmvq_f64(float64x2_t a) {
  // CHECK-LABEL: test_vmaxnmvq_f64
  return vmaxnmvq_f64(a);
  // CHECK: fmaxnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

float32_t test_vmaxnmv_f32(float32x2_t a) {
  // CHECK-LABEL: test_vmaxnmv_f32
  return vmaxnmv_f32(a);
  // CHECK: fmaxnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

float64_t test_vminnmvq_f64(float64x2_t a) {
  // CHECK-LABEL: test_vminnmvq_f64
  return vminnmvq_f64(a);
  // CHECK: fminnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

float32_t test_vminnmv_f32(float32x2_t a) {
  // CHECK-LABEL: test_vminnmv_f32
  return vminnmv_f32(a);
  // CHECK: fminnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
}

int64x2_t test_vpaddq_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vpaddq_s64
  return vpaddq_s64(a, b);
  // CHECK: addp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vpaddq_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vpaddq_u64
  return vpaddq_u64(a, b);
  // CHECK: addp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64_t test_vpaddd_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vpaddd_u64
  return vpaddd_u64(a);
  // CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

int64_t test_vaddvq_s64(int64x2_t a) {
  // CHECK-LABEL: test_vaddvq_s64
  return vaddvq_s64(a);
  // CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

uint64_t test_vaddvq_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vaddvq_u64
  return vaddvq_u64(a);
  // CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
}

float64x1_t test_vadd_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vadd_f64
  return vadd_f64(a, b);
  // CHECK: fadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vmul_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vmul_f64
  return vmul_f64(a, b);
  // CHECK: fmul d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vdiv_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vdiv_f64
  return vdiv_f64(a, b);
  // CHECK: fdiv d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vmla_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  // CHECK-LABEL: test_vmla_f64
  return vmla_f64(a, b, c);
  // CHECK: fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vmls_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  // CHECK-LABEL: test_vmls_f64
  return vmls_f64(a, b, c);
  // CHECK: fmsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vfma_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  // CHECK-LABEL: test_vfma_f64
  return vfma_f64(a, b, c);
  // CHECK: fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vfms_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  // CHECK-LABEL: test_vfms_f64
  return vfms_f64(a, b, c);
  // CHECK: fmsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vsub_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vsub_f64
  return vsub_f64(a, b);
  // CHECK: fsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vabd_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vabd_f64
  return vabd_f64(a, b);
  // CHECK: fabd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vmax_f64(float64x1_t a, float64x1_t b) {
// CHECK-LABEL: test_vmax_f64
  return vmax_f64(a, b);
// CHECK: fmax d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vmin_f64(float64x1_t a, float64x1_t b) {
// CHECK-LABEL: test_vmin_f64
  return vmin_f64(a, b);
// CHECK: fmin d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vmaxnm_f64(float64x1_t a, float64x1_t b) {
// CHECK-LABEL: test_vmaxnm_f64
  return vmaxnm_f64(a, b);
// CHECK: fmaxnm d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vminnm_f64(float64x1_t a, float64x1_t b) {
// CHECK-LABEL: test_vminnm_f64
  return vminnm_f64(a, b);
// CHECK: fminnm d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vabs_f64(float64x1_t a) {
  // CHECK-LABEL: test_vabs_f64
  return vabs_f64(a);
  // CHECK: fabs d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vneg_f64(float64x1_t a) {
  // CHECK-LABEL: test_vneg_f64
  return vneg_f64(a);
  // CHECK: fneg d{{[0-9]+}}, d{{[0-9]+}}
}

int64x1_t test_vcvt_s64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvt_s64_f64
  return vcvt_s64_f64(a);
  // CHECK: fcvtzs {{[xd][0-9]+}}, d{{[0-9]+}}
}

uint64x1_t test_vcvt_u64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvt_u64_f64
  return vcvt_u64_f64(a);
  // CHECK: fcvtzu {{[xd][0-9]+}}, d{{[0-9]+}}
}

int64x1_t test_vcvtn_s64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvtn_s64_f64
  return vcvtn_s64_f64(a);
  // CHECK: fcvtns d{{[0-9]+}}, d{{[0-9]+}}
}

uint64x1_t test_vcvtn_u64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvtn_u64_f64
  return vcvtn_u64_f64(a);
  // CHECK: fcvtnu d{{[0-9]+}}, d{{[0-9]+}}
}

int64x1_t test_vcvtp_s64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvtp_s64_f64
  return vcvtp_s64_f64(a);
  // CHECK: fcvtps d{{[0-9]+}}, d{{[0-9]+}}
}

uint64x1_t test_vcvtp_u64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvtp_u64_f64
  return vcvtp_u64_f64(a);
  // CHECK: fcvtpu d{{[0-9]+}}, d{{[0-9]+}}
}

int64x1_t test_vcvtm_s64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvtm_s64_f64
  return vcvtm_s64_f64(a);
  // CHECK: fcvtms d{{[0-9]+}}, d{{[0-9]+}}
}

uint64x1_t test_vcvtm_u64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvtm_u64_f64
  return vcvtm_u64_f64(a);
  // CHECK: fcvtmu d{{[0-9]+}}, d{{[0-9]+}}
}

int64x1_t test_vcvta_s64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvta_s64_f64
  return vcvta_s64_f64(a);
  // CHECK: fcvtas d{{[0-9]+}}, d{{[0-9]+}}
}

uint64x1_t test_vcvta_u64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvta_u64_f64
  return vcvta_u64_f64(a);
  // CHECK: fcvtau d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vcvt_f64_s64(int64x1_t a) {
  // CHECK-LABEL: test_vcvt_f64_s64
  return vcvt_f64_s64(a);
  // CHECK: scvtf d{{[0-9]+}}, {{[xd][0-9]+}}
}

float64x1_t test_vcvt_f64_u64(uint64x1_t a) {
  // CHECK-LABEL: test_vcvt_f64_u64
  return vcvt_f64_u64(a);
  // CHECK: ucvtf d{{[0-9]+}}, {{[xd][0-9]+}}
}

int64x1_t test_vcvt_n_s64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvt_n_s64_f64
  return vcvt_n_s64_f64(a, 64);
  // CHECK: fcvtzs d{{[0-9]+}}, d{{[0-9]+}}, #64
}

uint64x1_t test_vcvt_n_u64_f64(float64x1_t a) {
  // CHECK-LABEL: test_vcvt_n_u64_f64
  return vcvt_n_u64_f64(a, 64);
  // CHECK: fcvtzu d{{[0-9]+}}, d{{[0-9]+}}, #64
}

float64x1_t test_vcvt_n_f64_s64(int64x1_t a) {
  // CHECK-LABEL: test_vcvt_n_f64_s64
  return vcvt_n_f64_s64(a, 64);
  // CHECK: scvtf d{{[0-9]+}}, d{{[0-9]+}}, #64
}

float64x1_t test_vcvt_n_f64_u64(uint64x1_t a) {
  // CHECK-LABEL: test_vcvt_n_f64_u64
  return vcvt_n_f64_u64(a, 64);
  // CHECK: ucvtf d{{[0-9]+}}, d{{[0-9]+}}, #64
}

float64x1_t test_vrndn_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrndn_f64
  return vrndn_f64(a);
  // CHECK: frintn d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrnda_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrnda_f64
  return vrnda_f64(a);
  // CHECK: frinta d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrndp_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrndp_f64
  return vrndp_f64(a);
  // CHECK: frintp d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrndm_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrndm_f64
  return vrndm_f64(a);
  // CHECK: frintm d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrndx_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrndx_f64
  return vrndx_f64(a);
  // CHECK: frintx d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrnd_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrnd_f64
  return vrnd_f64(a);
  // CHECK: frintz d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrndi_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrndi_f64
  return vrndi_f64(a);
  // CHECK: frinti d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrsqrte_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrsqrte_f64
  return vrsqrte_f64(a);
  // CHECK: frsqrte d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrecpe_f64(float64x1_t a) {
  // CHECK-LABEL: test_vrecpe_f64
  return vrecpe_f64(a);
  // CHECK: frecpe d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vsqrt_f64(float64x1_t a) {
  // CHECK-LABEL: test_vsqrt_f64
  return vsqrt_f64(a);
  // CHECK: fsqrt d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrecps_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vrecps_f64
  return vrecps_f64(a, b);
  // CHECK: frecps d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

float64x1_t test_vrsqrts_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vrsqrts_f64
  return vrsqrts_f64(a, b);
  // CHECK: frsqrts d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
}

int32_t test_vminv_s32(int32x2_t a) {
  // CHECK-LABEL: test_vminv_s32
  return vminv_s32(a);
  // CHECK: sminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32_t test_vminv_u32(uint32x2_t a) {
  // CHECK-LABEL: test_vminv_u32
  return vminv_u32(a);
  // CHECK: uminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int32_t test_vmaxv_s32(int32x2_t a) {
  // CHECK-LABEL: test_vmaxv_s32
  return vmaxv_s32(a);
  // CHECK: smaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32_t test_vmaxv_u32(uint32x2_t a) {
  // CHECK-LABEL: test_vmaxv_u32
  return vmaxv_u32(a);
  // CHECK: umaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int32_t test_vaddv_s32(int32x2_t a) {
  // CHECK-LABEL: test_vaddv_s32
  return vaddv_s32(a);
  // CHECK: addp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32_t test_vaddv_u32(uint32x2_t a) {
  // CHECK-LABEL: test_vaddv_u32
  return vaddv_u32(a);
  // CHECK: addp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64_t test_vaddlv_s32(int32x2_t a) {
  // CHECK-LABEL: test_vaddlv_s32
  return vaddlv_s32(a);
  // CHECK: saddlp {{v[0-9]+}}.1d, {{v[0-9]+}}.2s
}

uint64_t test_vaddlv_u32(uint32x2_t a) {
  // CHECK-LABEL: test_vaddlv_u32
  return vaddlv_u32(a);
  // CHECK: uaddlp {{v[0-9]+}}.1d, {{v[0-9]+}}.2s
}
