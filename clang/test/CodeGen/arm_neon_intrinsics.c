// RUN: %clang_cc1 -triple thumbv7s-apple-darwin -target-abi apcs-gnu\
// RUN:  -target-cpu swift -ffreestanding -Os -S -o - %s\
// RUN:  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-SWIFT
// RUN: %clang_cc1 -triple armv8-linux-gnu \
// RUN:  -target-cpu cortex-a57 -mfloat-abi soft -ffreestanding -Os -S -o - %s\
// RUN:  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-A57

// REQUIRES: long_tests

#include <arm_neon.h>

// CHECK-LABEL: test_vaba_s8
// CHECK: vaba.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vaba_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
  return vaba_s8(a, b, c);
}

// CHECK-LABEL: test_vaba_s16
// CHECK: vaba.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vaba_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
  return vaba_s16(a, b, c);
}

// CHECK-LABEL: test_vaba_s32
// CHECK: vaba.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vaba_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
  return vaba_s32(a, b, c);
}

// CHECK-LABEL: test_vaba_u8
// CHECK: vaba.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vaba_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c) {
  return vaba_u8(a, b, c);
}

// CHECK-LABEL: test_vaba_u16
// CHECK: vaba.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vaba_u16(uint16x4_t a, uint16x4_t b, uint16x4_t c) {
  return vaba_u16(a, b, c);
}

// CHECK-LABEL: test_vaba_u32
// CHECK: vaba.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vaba_u32(uint32x2_t a, uint32x2_t b, uint32x2_t c) {
  return vaba_u32(a, b, c);
}

// CHECK-LABEL: test_vabaq_s8
// CHECK: vaba.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vabaq_s8(int8x16_t a, int8x16_t b, int8x16_t c) {
  return vabaq_s8(a, b, c);
}

// CHECK-LABEL: test_vabaq_s16
// CHECK: vaba.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vabaq_s16(int16x8_t a, int16x8_t b, int16x8_t c) {
  return vabaq_s16(a, b, c);
}

// CHECK-LABEL: test_vabaq_s32
// CHECK: vaba.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vabaq_s32(int32x4_t a, int32x4_t b, int32x4_t c) {
  return vabaq_s32(a, b, c);
}

// CHECK-LABEL: test_vabaq_u8
// CHECK: vaba.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vabaq_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
  return vabaq_u8(a, b, c);
}

// CHECK-LABEL: test_vabaq_u16
// CHECK: vaba.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vabaq_u16(uint16x8_t a, uint16x8_t b, uint16x8_t c) {
  return vabaq_u16(a, b, c);
}

// CHECK-LABEL: test_vabaq_u32
// CHECK: vaba.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vabaq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  return vabaq_u32(a, b, c);
}


// CHECK-LABEL: test_vabal_s8
// CHECK: vabal.s8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vabal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  return vabal_s8(a, b, c);
}

// CHECK-LABEL: test_vabal_s16
// CHECK: vabal.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vabal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vabal_s16(a, b, c);
}

// CHECK-LABEL: test_vabal_s32
// CHECK: vabal.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vabal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vabal_s32(a, b, c);
}

// CHECK-LABEL: test_vabal_u8
// CHECK: vabal.u8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vabal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  return vabal_u8(a, b, c);
}

// CHECK-LABEL: test_vabal_u16
// CHECK: vabal.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vabal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  return vabal_u16(a, b, c);
}

// CHECK-LABEL: test_vabal_u32
// CHECK: vabal.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vabal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  return vabal_u32(a, b, c);
}


// CHECK-LABEL: test_vabd_s8
// CHECK: vabd.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vabd_s8(int8x8_t a, int8x8_t b) {
  return vabd_s8(a, b);
}

// CHECK-LABEL: test_vabd_s16
// CHECK: vabd.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vabd_s16(int16x4_t a, int16x4_t b) {
  return vabd_s16(a, b);
}

// CHECK-LABEL: test_vabd_s32
// CHECK: vabd.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vabd_s32(int32x2_t a, int32x2_t b) {
  return vabd_s32(a, b);
}

// CHECK-LABEL: test_vabd_u8
// CHECK: vabd.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vabd_u8(uint8x8_t a, uint8x8_t b) {
  return vabd_u8(a, b);
}

// CHECK-LABEL: test_vabd_u16
// CHECK: vabd.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vabd_u16(uint16x4_t a, uint16x4_t b) {
  return vabd_u16(a, b);
}

// CHECK-LABEL: test_vabd_u32
// CHECK: vabd.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vabd_u32(uint32x2_t a, uint32x2_t b) {
  return vabd_u32(a, b);
}

// CHECK-LABEL: test_vabd_f32
// CHECK: vabd.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vabd_f32(float32x2_t a, float32x2_t b) {
  return vabd_f32(a, b);
}

// CHECK-LABEL: test_vabdq_s8
// CHECK: vabd.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vabdq_s8(int8x16_t a, int8x16_t b) {
  return vabdq_s8(a, b);
}

// CHECK-LABEL: test_vabdq_s16
// CHECK: vabd.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vabdq_s16(int16x8_t a, int16x8_t b) {
  return vabdq_s16(a, b);
}

// CHECK-LABEL: test_vabdq_s32
// CHECK: vabd.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vabdq_s32(int32x4_t a, int32x4_t b) {
  return vabdq_s32(a, b);
}

// CHECK-LABEL: test_vabdq_u8
// CHECK: vabd.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vabdq_u8(uint8x16_t a, uint8x16_t b) {
  return vabdq_u8(a, b);
}

// CHECK-LABEL: test_vabdq_u16
// CHECK: vabd.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vabdq_u16(uint16x8_t a, uint16x8_t b) {
  return vabdq_u16(a, b);
}

// CHECK-LABEL: test_vabdq_u32
// CHECK: vabd.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vabdq_u32(uint32x4_t a, uint32x4_t b) {
  return vabdq_u32(a, b);
}

// CHECK-LABEL: test_vabdq_f32
// CHECK: vabd.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vabdq_f32(float32x4_t a, float32x4_t b) {
  return vabdq_f32(a, b);
}


// CHECK-LABEL: test_vabdl_s8
// CHECK: vabdl.s8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vabdl_s8(int8x8_t a, int8x8_t b) {
  return vabdl_s8(a, b);
}

// CHECK-LABEL: test_vabdl_s16
// CHECK: vabdl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vabdl_s16(int16x4_t a, int16x4_t b) {
  return vabdl_s16(a, b);
}

// CHECK-LABEL: test_vabdl_s32
// CHECK: vabdl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vabdl_s32(int32x2_t a, int32x2_t b) {
  return vabdl_s32(a, b);
}

// CHECK-LABEL: test_vabdl_u8
// CHECK: vabdl.u8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vabdl_u8(uint8x8_t a, uint8x8_t b) {
  return vabdl_u8(a, b);
}

// CHECK-LABEL: test_vabdl_u16
// CHECK: vabdl.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vabdl_u16(uint16x4_t a, uint16x4_t b) {
  return vabdl_u16(a, b);
}

// CHECK-LABEL: test_vabdl_u32
// CHECK: vabdl.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vabdl_u32(uint32x2_t a, uint32x2_t b) {
  return vabdl_u32(a, b);
}


// CHECK-LABEL: test_vabs_s8
// CHECK: vabs.s8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vabs_s8(int8x8_t a) {
  return vabs_s8(a);
}

// CHECK-LABEL: test_vabs_s16
// CHECK: vabs.s16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vabs_s16(int16x4_t a) {
  return vabs_s16(a);
}

// CHECK-LABEL: test_vabs_s32
// CHECK: vabs.s32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vabs_s32(int32x2_t a) {
  return vabs_s32(a);
}

// CHECK-LABEL: test_vabs_f32
// CHECK: vabs.f32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vabs_f32(float32x2_t a) {
  return vabs_f32(a);
}

// CHECK-LABEL: test_vabsq_s8
// CHECK: vabs.s8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vabsq_s8(int8x16_t a) {
  return vabsq_s8(a);
}

// CHECK-LABEL: test_vabsq_s16
// CHECK: vabs.s16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vabsq_s16(int16x8_t a) {
  return vabsq_s16(a);
}

// CHECK-LABEL: test_vabsq_s32
// CHECK: vabs.s32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vabsq_s32(int32x4_t a) {
  return vabsq_s32(a);
}

// CHECK-LABEL: test_vabsq_f32
// CHECK: vabs.f32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vabsq_f32(float32x4_t a) {
  return vabsq_f32(a);
}


// CHECK-LABEL: test_vadd_s8
// CHECK: vadd.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vadd_s8(int8x8_t a, int8x8_t b) {
  return vadd_s8(a, b);
}

// CHECK-LABEL: test_vadd_s16
// CHECK: vadd.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vadd_s16(int16x4_t a, int16x4_t b) {
  return vadd_s16(a, b);
}

// CHECK-LABEL: test_vadd_s32
// CHECK: vadd.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vadd_s32(int32x2_t a, int32x2_t b) {
  return vadd_s32(a, b);
}

// CHECK-LABEL: test_vadd_s64
// CHECK: vadd.i64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vadd_s64(int64x1_t a, int64x1_t b) {
  return vadd_s64(a, b);
}

// CHECK-LABEL: test_vadd_f32
// CHECK: vadd.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vadd_f32(float32x2_t a, float32x2_t b) {
  return vadd_f32(a, b);
}

// CHECK-LABEL: test_vadd_u8
// CHECK: vadd.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vadd_u8(uint8x8_t a, uint8x8_t b) {
  return vadd_u8(a, b);
}

// CHECK-LABEL: test_vadd_u16
// CHECK: vadd.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vadd_u16(uint16x4_t a, uint16x4_t b) {
  return vadd_u16(a, b);
}

// CHECK-LABEL: test_vadd_u32
// CHECK: vadd.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vadd_u32(uint32x2_t a, uint32x2_t b) {
  return vadd_u32(a, b);
}

// CHECK-LABEL: test_vadd_u64
// CHECK: vadd.i64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vadd_u64(uint64x1_t a, uint64x1_t b) {
  return vadd_u64(a, b);
}

// CHECK-LABEL: test_vaddq_s8
// CHECK: vadd.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vaddq_s8(int8x16_t a, int8x16_t b) {
  return vaddq_s8(a, b);
}

// CHECK-LABEL: test_vaddq_s16
// CHECK: vadd.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vaddq_s16(int16x8_t a, int16x8_t b) {
  return vaddq_s16(a, b);
}

// CHECK-LABEL: test_vaddq_s32
// CHECK: vadd.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vaddq_s32(int32x4_t a, int32x4_t b) {
  return vaddq_s32(a, b);
}

// CHECK-LABEL: test_vaddq_s64
// CHECK: vadd.i64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vaddq_s64(int64x2_t a, int64x2_t b) {
  return vaddq_s64(a, b);
}

// CHECK-LABEL: test_vaddq_f32
// CHECK: vadd.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vaddq_f32(float32x4_t a, float32x4_t b) {
  return vaddq_f32(a, b);
}

// CHECK-LABEL: test_vaddq_u8
// CHECK: vadd.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vaddq_u8(uint8x16_t a, uint8x16_t b) {
  return vaddq_u8(a, b);
}

// CHECK-LABEL: test_vaddq_u16
// CHECK: vadd.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vaddq_u16(uint16x8_t a, uint16x8_t b) {
  return vaddq_u16(a, b);
}

// CHECK-LABEL: test_vaddq_u32
// CHECK: vadd.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vaddq_u32(uint32x4_t a, uint32x4_t b) {
  return vaddq_u32(a, b);
}

// CHECK-LABEL: test_vaddq_u64
// CHECK: vadd.i64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vaddq_u64(uint64x2_t a, uint64x2_t b) {
  return vaddq_u64(a, b);
}


// CHECK-LABEL: test_vaddhn_s16
// CHECK: vaddhn.i16 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x8_t test_vaddhn_s16(int16x8_t a, int16x8_t b) {
  return vaddhn_s16(a, b);
}

// CHECK-LABEL: test_vaddhn_s32
// CHECK: vaddhn.i32 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x4_t test_vaddhn_s32(int32x4_t a, int32x4_t b) {
  return vaddhn_s32(a, b);
}

// CHECK-LABEL: test_vaddhn_s64
// CHECK: vaddhn.i64 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x2_t test_vaddhn_s64(int64x2_t a, int64x2_t b) {
  return vaddhn_s64(a, b);
}

// CHECK-LABEL: test_vaddhn_u16
// CHECK: vaddhn.i16 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x8_t test_vaddhn_u16(uint16x8_t a, uint16x8_t b) {
  return vaddhn_u16(a, b);
}

// CHECK-LABEL: test_vaddhn_u32
// CHECK: vaddhn.i32 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x4_t test_vaddhn_u32(uint32x4_t a, uint32x4_t b) {
  return vaddhn_u32(a, b);
}

// CHECK-LABEL: test_vaddhn_u64
// CHECK: vaddhn.i64 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x2_t test_vaddhn_u64(uint64x2_t a, uint64x2_t b) {
  return vaddhn_u64(a, b);
}


// CHECK-LABEL: test_vaddl_s8
// CHECK: vaddl.s8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vaddl_s8(int8x8_t a, int8x8_t b) {
  return vaddl_s8(a, b);
}

// CHECK-LABEL: test_vaddl_s16
// CHECK: vaddl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vaddl_s16(int16x4_t a, int16x4_t b) {
  return vaddl_s16(a, b);
}

// CHECK-LABEL: test_vaddl_s32
// CHECK: vaddl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vaddl_s32(int32x2_t a, int32x2_t b) {
  return vaddl_s32(a, b);
}

// CHECK-LABEL: test_vaddl_u8
// CHECK: vaddl.u8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vaddl_u8(uint8x8_t a, uint8x8_t b) {
  return vaddl_u8(a, b);
}

// CHECK-LABEL: test_vaddl_u16
// CHECK: vaddl.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vaddl_u16(uint16x4_t a, uint16x4_t b) {
  return vaddl_u16(a, b);
}

// CHECK-LABEL: test_vaddl_u32
// CHECK: vaddl.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vaddl_u32(uint32x2_t a, uint32x2_t b) {
  return vaddl_u32(a, b);
}


// CHECK-LABEL: test_vaddw_s8
// CHECK: vaddw.s8 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vaddw_s8(int16x8_t a, int8x8_t b) {
  return vaddw_s8(a, b);
}

// CHECK-LABEL: test_vaddw_s16
// CHECK: vaddw.s16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vaddw_s16(int32x4_t a, int16x4_t b) {
  return vaddw_s16(a, b);
}

// CHECK-LABEL: test_vaddw_s32
// CHECK: vaddw.s32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vaddw_s32(int64x2_t a, int32x2_t b) {
  return vaddw_s32(a, b);
}

// CHECK-LABEL: test_vaddw_u8
// CHECK: vaddw.u8 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vaddw_u8(uint16x8_t a, uint8x8_t b) {
  return vaddw_u8(a, b);
}

// CHECK-LABEL: test_vaddw_u16
// CHECK: vaddw.u16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vaddw_u16(uint32x4_t a, uint16x4_t b) {
  return vaddw_u16(a, b);
}

// CHECK-LABEL: test_vaddw_u32
// CHECK: vaddw.u32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vaddw_u32(uint64x2_t a, uint32x2_t b) {
  return vaddw_u32(a, b);
}


// CHECK-LABEL: test_vand_s8
// CHECK: vand d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vand_s8(int8x8_t a, int8x8_t b) {
  return vand_s8(a, b);
}

// CHECK-LABEL: test_vand_s16
// CHECK: vand d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vand_s16(int16x4_t a, int16x4_t b) {
  return vand_s16(a, b);
}

// CHECK-LABEL: test_vand_s32
// CHECK: vand d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vand_s32(int32x2_t a, int32x2_t b) {
  return vand_s32(a, b);
}

// CHECK-LABEL: test_vand_s64
// CHECK: vand d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vand_s64(int64x1_t a, int64x1_t b) {
  return vand_s64(a, b);
}

// CHECK-LABEL: test_vand_u8
// CHECK: vand d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vand_u8(uint8x8_t a, uint8x8_t b) {
  return vand_u8(a, b);
}

// CHECK-LABEL: test_vand_u16
// CHECK: vand d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vand_u16(uint16x4_t a, uint16x4_t b) {
  return vand_u16(a, b);
}

// CHECK-LABEL: test_vand_u32
// CHECK: vand d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vand_u32(uint32x2_t a, uint32x2_t b) {
  return vand_u32(a, b);
}

// CHECK-LABEL: test_vand_u64
// CHECK: vand d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vand_u64(uint64x1_t a, uint64x1_t b) {
  return vand_u64(a, b);
}

// CHECK-LABEL: test_vandq_s8
// CHECK: vand q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vandq_s8(int8x16_t a, int8x16_t b) {
  return vandq_s8(a, b);
}

// CHECK-LABEL: test_vandq_s16
// CHECK: vand q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vandq_s16(int16x8_t a, int16x8_t b) {
  return vandq_s16(a, b);
}

// CHECK-LABEL: test_vandq_s32
// CHECK: vand q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vandq_s32(int32x4_t a, int32x4_t b) {
  return vandq_s32(a, b);
}

// CHECK-LABEL: test_vandq_s64
// CHECK: vand q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vandq_s64(int64x2_t a, int64x2_t b) {
  return vandq_s64(a, b);
}

// CHECK-LABEL: test_vandq_u8
// CHECK: vand q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vandq_u8(uint8x16_t a, uint8x16_t b) {
  return vandq_u8(a, b);
}

// CHECK-LABEL: test_vandq_u16
// CHECK: vand q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vandq_u16(uint16x8_t a, uint16x8_t b) {
  return vandq_u16(a, b);
}

// CHECK-LABEL: test_vandq_u32
// CHECK: vand q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vandq_u32(uint32x4_t a, uint32x4_t b) {
  return vandq_u32(a, b);
}

// CHECK-LABEL: test_vandq_u64
// CHECK: vand q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vandq_u64(uint64x2_t a, uint64x2_t b) {
  return vandq_u64(a, b);
}


// CHECK-LABEL: test_vbic_s8
// CHECK: vbic d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vbic_s8(int8x8_t a, int8x8_t b) {
  return vbic_s8(a, b);
}

// CHECK-LABEL: test_vbic_s16
// CHECK: vbic d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vbic_s16(int16x4_t a, int16x4_t b) {
  return vbic_s16(a, b);
}

// CHECK-LABEL: test_vbic_s32
// CHECK: vbic d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vbic_s32(int32x2_t a, int32x2_t b) {
  return vbic_s32(a, b);
}

// CHECK-LABEL: test_vbic_s64
// CHECK: vbic d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vbic_s64(int64x1_t a, int64x1_t b) {
  return vbic_s64(a, b);
}

// CHECK-LABEL: test_vbic_u8
// CHECK: vbic d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vbic_u8(uint8x8_t a, uint8x8_t b) {
  return vbic_u8(a, b);
}

// CHECK-LABEL: test_vbic_u16
// CHECK: vbic d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vbic_u16(uint16x4_t a, uint16x4_t b) {
  return vbic_u16(a, b);
}

// CHECK-LABEL: test_vbic_u32
// CHECK: vbic d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vbic_u32(uint32x2_t a, uint32x2_t b) {
  return vbic_u32(a, b);
}

// CHECK-LABEL: test_vbic_u64
// CHECK: vbic d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vbic_u64(uint64x1_t a, uint64x1_t b) {
  return vbic_u64(a, b);
}

// CHECK-LABEL: test_vbicq_s8
// CHECK: vbic q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vbicq_s8(int8x16_t a, int8x16_t b) {
  return vbicq_s8(a, b);
}

// CHECK-LABEL: test_vbicq_s16
// CHECK: vbic q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vbicq_s16(int16x8_t a, int16x8_t b) {
  return vbicq_s16(a, b);
}

// CHECK-LABEL: test_vbicq_s32
// CHECK: vbic q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vbicq_s32(int32x4_t a, int32x4_t b) {
  return vbicq_s32(a, b);
}

// CHECK-LABEL: test_vbicq_s64
// CHECK: vbic q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vbicq_s64(int64x2_t a, int64x2_t b) {
  return vbicq_s64(a, b);
}

// CHECK-LABEL: test_vbicq_u8
// CHECK: vbic q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vbicq_u8(uint8x16_t a, uint8x16_t b) {
  return vbicq_u8(a, b);
}

// CHECK-LABEL: test_vbicq_u16
// CHECK: vbic q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vbicq_u16(uint16x8_t a, uint16x8_t b) {
  return vbicq_u16(a, b);
}

// CHECK-LABEL: test_vbicq_u32
// CHECK: vbic q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vbicq_u32(uint32x4_t a, uint32x4_t b) {
  return vbicq_u32(a, b);
}

// CHECK-LABEL: test_vbicq_u64
// CHECK: vbic q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vbicq_u64(uint64x2_t a, uint64x2_t b) {
  return vbicq_u64(a, b);
}


// CHECK-LABEL: test_vbsl_s8
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vbsl_s8(uint8x8_t a, int8x8_t b, int8x8_t c) {
  return vbsl_s8(a, b, c);
}

// CHECK-LABEL: test_vbsl_s16
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vbsl_s16(uint16x4_t a, int16x4_t b, int16x4_t c) {
  return vbsl_s16(a, b, c);
}

// CHECK-LABEL: test_vbsl_s32
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vbsl_s32(uint32x2_t a, int32x2_t b, int32x2_t c) {
  return vbsl_s32(a, b, c);
}

// CHECK-LABEL: test_vbsl_s64
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vbsl_s64(uint64x1_t a, int64x1_t b, int64x1_t c) {
  return vbsl_s64(a, b, c);
}

// CHECK-LABEL: test_vbsl_u8
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vbsl_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c) {
  return vbsl_u8(a, b, c);
}

// CHECK-LABEL: test_vbsl_u16
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vbsl_u16(uint16x4_t a, uint16x4_t b, uint16x4_t c) {
  return vbsl_u16(a, b, c);
}

// CHECK-LABEL: test_vbsl_u32
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vbsl_u32(uint32x2_t a, uint32x2_t b, uint32x2_t c) {
  return vbsl_u32(a, b, c);
}

// CHECK-LABEL: test_vbsl_u64
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vbsl_u64(uint64x1_t a, uint64x1_t b, uint64x1_t c) {
  return vbsl_u64(a, b, c);
}

// CHECK-LABEL: test_vbsl_f32
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vbsl_f32(uint32x2_t a, float32x2_t b, float32x2_t c) {
  return vbsl_f32(a, b, c);
}

// CHECK-LABEL: test_vbsl_p8
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
poly8x8_t test_vbsl_p8(uint8x8_t a, poly8x8_t b, poly8x8_t c) {
  return vbsl_p8(a, b, c);
}

// CHECK-LABEL: test_vbsl_p16
// CHECK: vbsl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
poly16x4_t test_vbsl_p16(uint16x4_t a, poly16x4_t b, poly16x4_t c) {
  return vbsl_p16(a, b, c);
}

// CHECK-LABEL: test_vbslq_s8
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vbslq_s8(uint8x16_t a, int8x16_t b, int8x16_t c) {
  return vbslq_s8(a, b, c);
}

// CHECK-LABEL: test_vbslq_s16
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vbslq_s16(uint16x8_t a, int16x8_t b, int16x8_t c) {
  return vbslq_s16(a, b, c);
}

// CHECK-LABEL: test_vbslq_s32
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vbslq_s32(uint32x4_t a, int32x4_t b, int32x4_t c) {
  return vbslq_s32(a, b, c);
}

// CHECK-LABEL: test_vbslq_s64
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vbslq_s64(uint64x2_t a, int64x2_t b, int64x2_t c) {
  return vbslq_s64(a, b, c);
}

// CHECK-LABEL: test_vbslq_u8
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vbslq_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
  return vbslq_u8(a, b, c);
}

// CHECK-LABEL: test_vbslq_u16
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vbslq_u16(uint16x8_t a, uint16x8_t b, uint16x8_t c) {
  return vbslq_u16(a, b, c);
}

// CHECK-LABEL: test_vbslq_u32
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vbslq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  return vbslq_u32(a, b, c);
}

// CHECK-LABEL: test_vbslq_u64
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vbslq_u64(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
  return vbslq_u64(a, b, c);
}

// CHECK-LABEL: test_vbslq_f32
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vbslq_f32(uint32x4_t a, float32x4_t b, float32x4_t c) {
  return vbslq_f32(a, b, c);
}

// CHECK-LABEL: test_vbslq_p8
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
poly8x16_t test_vbslq_p8(uint8x16_t a, poly8x16_t b, poly8x16_t c) {
  return vbslq_p8(a, b, c);
}

// CHECK-LABEL: test_vbslq_p16
// CHECK: vbsl q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
poly16x8_t test_vbslq_p16(uint16x8_t a, poly16x8_t b, poly16x8_t c) {
  return vbslq_p16(a, b, c);
}


// CHECK-LABEL: test_vcage_f32
// CHECK: vacge.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcage_f32(float32x2_t a, float32x2_t b) {
  return vcage_f32(a, b);
}

// CHECK-LABEL: test_vcageq_f32
// CHECK: vacge.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcageq_f32(float32x4_t a, float32x4_t b) {
  return vcageq_f32(a, b);
}


// CHECK-LABEL: test_vcagt_f32
// CHECK: vacgt.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcagt_f32(float32x2_t a, float32x2_t b) {
  return vcagt_f32(a, b);
}

// CHECK-LABEL: test_vcagtq_f32
// CHECK: vacgt.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcagtq_f32(float32x4_t a, float32x4_t b) {
  return vcagtq_f32(a, b);
}


// CHECK-LABEL: test_vcale_f32
// CHECK: vacge.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcale_f32(float32x2_t a, float32x2_t b) {
  return vcale_f32(a, b);
}

// CHECK-LABEL: test_vcaleq_f32
// CHECK: vacge.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcaleq_f32(float32x4_t a, float32x4_t b) {
  return vcaleq_f32(a, b);
}


// CHECK-LABEL: test_vcalt_f32
// CHECK: vacgt.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcalt_f32(float32x2_t a, float32x2_t b) {
  return vcalt_f32(a, b);
}

// CHECK-LABEL: test_vcaltq_f32
// CHECK: vacgt.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcaltq_f32(float32x4_t a, float32x4_t b) {
  return vcaltq_f32(a, b);
}


// CHECK-LABEL: test_vceq_s8
// CHECK: vceq.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vceq_s8(int8x8_t a, int8x8_t b) {
  return vceq_s8(a, b);
}

// CHECK-LABEL: test_vceq_s16
// CHECK: vceq.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vceq_s16(int16x4_t a, int16x4_t b) {
  return vceq_s16(a, b);
}

// CHECK-LABEL: test_vceq_s32
// CHECK: vceq.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vceq_s32(int32x2_t a, int32x2_t b) {
  return vceq_s32(a, b);
}

// CHECK-LABEL: test_vceq_f32
// CHECK: vceq.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vceq_f32(float32x2_t a, float32x2_t b) {
  return vceq_f32(a, b);
}

// CHECK-LABEL: test_vceq_u8
// CHECK: vceq.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vceq_u8(uint8x8_t a, uint8x8_t b) {
  return vceq_u8(a, b);
}

// CHECK-LABEL: test_vceq_u16
// CHECK: vceq.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vceq_u16(uint16x4_t a, uint16x4_t b) {
  return vceq_u16(a, b);
}

// CHECK-LABEL: test_vceq_u32
// CHECK: vceq.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vceq_u32(uint32x2_t a, uint32x2_t b) {
  return vceq_u32(a, b);
}

// CHECK-LABEL: test_vceq_p8
// CHECK: vceq.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vceq_p8(poly8x8_t a, poly8x8_t b) {
  return vceq_p8(a, b);
}

// CHECK-LABEL: test_vceqq_s8
// CHECK: vceq.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vceqq_s8(int8x16_t a, int8x16_t b) {
  return vceqq_s8(a, b);
}

// CHECK-LABEL: test_vceqq_s16
// CHECK: vceq.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vceqq_s16(int16x8_t a, int16x8_t b) {
  return vceqq_s16(a, b);
}

// CHECK-LABEL: test_vceqq_s32
// CHECK: vceq.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vceqq_s32(int32x4_t a, int32x4_t b) {
  return vceqq_s32(a, b);
}

// CHECK-LABEL: test_vceqq_f32
// CHECK: vceq.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vceqq_f32(float32x4_t a, float32x4_t b) {
  return vceqq_f32(a, b);
}

// CHECK-LABEL: test_vceqq_u8
// CHECK: vceq.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vceqq_u8(uint8x16_t a, uint8x16_t b) {
  return vceqq_u8(a, b);
}

// CHECK-LABEL: test_vceqq_u16
// CHECK: vceq.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vceqq_u16(uint16x8_t a, uint16x8_t b) {
  return vceqq_u16(a, b);
}

// CHECK-LABEL: test_vceqq_u32
// CHECK: vceq.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vceqq_u32(uint32x4_t a, uint32x4_t b) {
  return vceqq_u32(a, b);
}

// CHECK-LABEL: test_vceqq_p8
// CHECK: vceq.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vceqq_p8(poly8x16_t a, poly8x16_t b) {
  return vceqq_p8(a, b);
}


// CHECK-LABEL: test_vcge_s8
// CHECK: vcge.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vcge_s8(int8x8_t a, int8x8_t b) {
  return vcge_s8(a, b);
}

// CHECK-LABEL: test_vcge_s16
// CHECK: vcge.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vcge_s16(int16x4_t a, int16x4_t b) {
  return vcge_s16(a, b);
}

// CHECK-LABEL: test_vcge_s32
// CHECK: vcge.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcge_s32(int32x2_t a, int32x2_t b) {
  return vcge_s32(a, b);
}

// CHECK-LABEL: test_vcge_f32
// CHECK: vcge.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcge_f32(float32x2_t a, float32x2_t b) {
  return vcge_f32(a, b);
}

// CHECK-LABEL: test_vcge_u8
// CHECK: vcge.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vcge_u8(uint8x8_t a, uint8x8_t b) {
  return vcge_u8(a, b);
}

// CHECK-LABEL: test_vcge_u16
// CHECK: vcge.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vcge_u16(uint16x4_t a, uint16x4_t b) {
  return vcge_u16(a, b);
}

// CHECK-LABEL: test_vcge_u32
// CHECK: vcge.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcge_u32(uint32x2_t a, uint32x2_t b) {
  return vcge_u32(a, b);
}

// CHECK-LABEL: test_vcgeq_s8
// CHECK: vcge.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcgeq_s8(int8x16_t a, int8x16_t b) {
  return vcgeq_s8(a, b);
}

// CHECK-LABEL: test_vcgeq_s16
// CHECK: vcge.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vcgeq_s16(int16x8_t a, int16x8_t b) {
  return vcgeq_s16(a, b);
}

// CHECK-LABEL: test_vcgeq_s32
// CHECK: vcge.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcgeq_s32(int32x4_t a, int32x4_t b) {
  return vcgeq_s32(a, b);
}

// CHECK-LABEL: test_vcgeq_f32
// CHECK: vcge.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcgeq_f32(float32x4_t a, float32x4_t b) {
  return vcgeq_f32(a, b);
}

// CHECK-LABEL: test_vcgeq_u8
// CHECK: vcge.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcgeq_u8(uint8x16_t a, uint8x16_t b) {
  return vcgeq_u8(a, b);
}

// CHECK-LABEL: test_vcgeq_u16
// CHECK: vcge.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vcgeq_u16(uint16x8_t a, uint16x8_t b) {
  return vcgeq_u16(a, b);
}

// CHECK-LABEL: test_vcgeq_u32
// CHECK: vcge.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcgeq_u32(uint32x4_t a, uint32x4_t b) {
  return vcgeq_u32(a, b);
}


// CHECK-LABEL: test_vcgt_s8
// CHECK: vcgt.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vcgt_s8(int8x8_t a, int8x8_t b) {
  return vcgt_s8(a, b);
}

// CHECK-LABEL: test_vcgt_s16
// CHECK: vcgt.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vcgt_s16(int16x4_t a, int16x4_t b) {
  return vcgt_s16(a, b);
}

// CHECK-LABEL: test_vcgt_s32
// CHECK: vcgt.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcgt_s32(int32x2_t a, int32x2_t b) {
  return vcgt_s32(a, b);
}

// CHECK-LABEL: test_vcgt_f32
// CHECK: vcgt.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcgt_f32(float32x2_t a, float32x2_t b) {
  return vcgt_f32(a, b);
}

// CHECK-LABEL: test_vcgt_u8
// CHECK: vcgt.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vcgt_u8(uint8x8_t a, uint8x8_t b) {
  return vcgt_u8(a, b);
}

// CHECK-LABEL: test_vcgt_u16
// CHECK: vcgt.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vcgt_u16(uint16x4_t a, uint16x4_t b) {
  return vcgt_u16(a, b);
}

// CHECK-LABEL: test_vcgt_u32
// CHECK: vcgt.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcgt_u32(uint32x2_t a, uint32x2_t b) {
  return vcgt_u32(a, b);
}

// CHECK-LABEL: test_vcgtq_s8
// CHECK: vcgt.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcgtq_s8(int8x16_t a, int8x16_t b) {
  return vcgtq_s8(a, b);
}

// CHECK-LABEL: test_vcgtq_s16
// CHECK: vcgt.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vcgtq_s16(int16x8_t a, int16x8_t b) {
  return vcgtq_s16(a, b);
}

// CHECK-LABEL: test_vcgtq_s32
// CHECK: vcgt.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcgtq_s32(int32x4_t a, int32x4_t b) {
  return vcgtq_s32(a, b);
}

// CHECK-LABEL: test_vcgtq_f32
// CHECK: vcgt.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcgtq_f32(float32x4_t a, float32x4_t b) {
  return vcgtq_f32(a, b);
}

// CHECK-LABEL: test_vcgtq_u8
// CHECK: vcgt.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcgtq_u8(uint8x16_t a, uint8x16_t b) {
  return vcgtq_u8(a, b);
}

// CHECK-LABEL: test_vcgtq_u16
// CHECK: vcgt.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vcgtq_u16(uint16x8_t a, uint16x8_t b) {
  return vcgtq_u16(a, b);
}

// CHECK-LABEL: test_vcgtq_u32
// CHECK: vcgt.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcgtq_u32(uint32x4_t a, uint32x4_t b) {
  return vcgtq_u32(a, b);
}


// CHECK-LABEL: test_vcle_s8
// CHECK: vcge.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vcle_s8(int8x8_t a, int8x8_t b) {
  return vcle_s8(a, b);
}

// CHECK-LABEL: test_vcle_s16
// CHECK: vcge.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vcle_s16(int16x4_t a, int16x4_t b) {
  return vcle_s16(a, b);
}

// CHECK-LABEL: test_vcle_s32
// CHECK: vcge.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcle_s32(int32x2_t a, int32x2_t b) {
  return vcle_s32(a, b);
}

// CHECK-LABEL: test_vcle_f32
// CHECK: vcge.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcle_f32(float32x2_t a, float32x2_t b) {
  return vcle_f32(a, b);
}

// CHECK-LABEL: test_vcle_u8
// CHECK: vcge.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vcle_u8(uint8x8_t a, uint8x8_t b) {
  return vcle_u8(a, b);
}

// CHECK-LABEL: test_vcle_u16
// CHECK: vcge.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vcle_u16(uint16x4_t a, uint16x4_t b) {
  return vcle_u16(a, b);
}

// CHECK-LABEL: test_vcle_u32
// CHECK: vcge.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcle_u32(uint32x2_t a, uint32x2_t b) {
  return vcle_u32(a, b);
}

// CHECK-LABEL: test_vcleq_s8
// CHECK: vcge.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcleq_s8(int8x16_t a, int8x16_t b) {
  return vcleq_s8(a, b);
}

// CHECK-LABEL: test_vcleq_s16
// CHECK: vcge.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vcleq_s16(int16x8_t a, int16x8_t b) {
  return vcleq_s16(a, b);
}

// CHECK-LABEL: test_vcleq_s32
// CHECK: vcge.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcleq_s32(int32x4_t a, int32x4_t b) {
  return vcleq_s32(a, b);
}

// CHECK-LABEL: test_vcleq_f32
// CHECK: vcge.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcleq_f32(float32x4_t a, float32x4_t b) {
  return vcleq_f32(a, b);
}

// CHECK-LABEL: test_vcleq_u8
// CHECK: vcge.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcleq_u8(uint8x16_t a, uint8x16_t b) {
  return vcleq_u8(a, b);
}

// CHECK-LABEL: test_vcleq_u16
// CHECK: vcge.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vcleq_u16(uint16x8_t a, uint16x8_t b) {
  return vcleq_u16(a, b);
}

// CHECK-LABEL: test_vcleq_u32
// CHECK: vcge.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcleq_u32(uint32x4_t a, uint32x4_t b) {
  return vcleq_u32(a, b);
}


// CHECK-LABEL: test_vcls_s8
// CHECK: vcls.s8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vcls_s8(int8x8_t a) {
  return vcls_s8(a);
}

// CHECK-LABEL: test_vcls_s16
// CHECK: vcls.s16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vcls_s16(int16x4_t a) {
  return vcls_s16(a);
}

// CHECK-LABEL: test_vcls_s32
// CHECK: vcls.s32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vcls_s32(int32x2_t a) {
  return vcls_s32(a);
}

// CHECK-LABEL: test_vclsq_s8
// CHECK: vcls.s8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vclsq_s8(int8x16_t a) {
  return vclsq_s8(a);
}

// CHECK-LABEL: test_vclsq_s16
// CHECK: vcls.s16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vclsq_s16(int16x8_t a) {
  return vclsq_s16(a);
}

// CHECK-LABEL: test_vclsq_s32
// CHECK: vcls.s32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vclsq_s32(int32x4_t a) {
  return vclsq_s32(a);
}


// CHECK-LABEL: test_vclt_s8
// CHECK: vcgt.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vclt_s8(int8x8_t a, int8x8_t b) {
  return vclt_s8(a, b);
}

// CHECK-LABEL: test_vclt_s16
// CHECK: vcgt.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vclt_s16(int16x4_t a, int16x4_t b) {
  return vclt_s16(a, b);
}

// CHECK-LABEL: test_vclt_s32
// CHECK: vcgt.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vclt_s32(int32x2_t a, int32x2_t b) {
  return vclt_s32(a, b);
}

// CHECK-LABEL: test_vclt_f32
// CHECK: vcgt.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vclt_f32(float32x2_t a, float32x2_t b) {
  return vclt_f32(a, b);
}

// CHECK-LABEL: test_vclt_u8
// CHECK: vcgt.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vclt_u8(uint8x8_t a, uint8x8_t b) {
  return vclt_u8(a, b);
}

// CHECK-LABEL: test_vclt_u16
// CHECK: vcgt.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vclt_u16(uint16x4_t a, uint16x4_t b) {
  return vclt_u16(a, b);
}

// CHECK-LABEL: test_vclt_u32
// CHECK: vcgt.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vclt_u32(uint32x2_t a, uint32x2_t b) {
  return vclt_u32(a, b);
}

// CHECK-LABEL: test_vcltq_s8
// CHECK: vcgt.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcltq_s8(int8x16_t a, int8x16_t b) {
  return vcltq_s8(a, b);
}

// CHECK-LABEL: test_vcltq_s16
// CHECK: vcgt.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vcltq_s16(int16x8_t a, int16x8_t b) {
  return vcltq_s16(a, b);
}

// CHECK-LABEL: test_vcltq_s32
// CHECK: vcgt.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcltq_s32(int32x4_t a, int32x4_t b) {
  return vcltq_s32(a, b);
}

// CHECK-LABEL: test_vcltq_f32
// CHECK: vcgt.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcltq_f32(float32x4_t a, float32x4_t b) {
  return vcltq_f32(a, b);
}

// CHECK-LABEL: test_vcltq_u8
// CHECK: vcgt.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcltq_u8(uint8x16_t a, uint8x16_t b) {
  return vcltq_u8(a, b);
}

// CHECK-LABEL: test_vcltq_u16
// CHECK: vcgt.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vcltq_u16(uint16x8_t a, uint16x8_t b) {
  return vcltq_u16(a, b);
}

// CHECK-LABEL: test_vcltq_u32
// CHECK: vcgt.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcltq_u32(uint32x4_t a, uint32x4_t b) {
  return vcltq_u32(a, b);
}


// CHECK-LABEL: test_vclz_s8
// CHECK: vclz.i8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vclz_s8(int8x8_t a) {
  return vclz_s8(a);
}

// CHECK-LABEL: test_vclz_s16
// CHECK: vclz.i16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vclz_s16(int16x4_t a) {
  return vclz_s16(a);
}

// CHECK-LABEL: test_vclz_s32
// CHECK: vclz.i32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vclz_s32(int32x2_t a) {
  return vclz_s32(a);
}

// CHECK-LABEL: test_vclz_u8
// CHECK: vclz.i8 d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vclz_u8(uint8x8_t a) {
  return vclz_u8(a);
}

// CHECK-LABEL: test_vclz_u16
// CHECK: vclz.i16 d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vclz_u16(uint16x4_t a) {
  return vclz_u16(a);
}

// CHECK-LABEL: test_vclz_u32
// CHECK: vclz.i32 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vclz_u32(uint32x2_t a) {
  return vclz_u32(a);
}

// CHECK-LABEL: test_vclzq_s8
// CHECK: vclz.i8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vclzq_s8(int8x16_t a) {
  return vclzq_s8(a);
}

// CHECK-LABEL: test_vclzq_s16
// CHECK: vclz.i16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vclzq_s16(int16x8_t a) {
  return vclzq_s16(a);
}

// CHECK-LABEL: test_vclzq_s32
// CHECK: vclz.i32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vclzq_s32(int32x4_t a) {
  return vclzq_s32(a);
}

// CHECK-LABEL: test_vclzq_u8
// CHECK: vclz.i8 q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vclzq_u8(uint8x16_t a) {
  return vclzq_u8(a);
}

// CHECK-LABEL: test_vclzq_u16
// CHECK: vclz.i16 q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vclzq_u16(uint16x8_t a) {
  return vclzq_u16(a);
}

// CHECK-LABEL: test_vclzq_u32
// CHECK: vclz.i32 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vclzq_u32(uint32x4_t a) {
  return vclzq_u32(a);
}


// CHECK-LABEL: test_vcnt_u8
// CHECK: vcnt.8 d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vcnt_u8(uint8x8_t a) {
  return vcnt_u8(a);
}

// CHECK-LABEL: test_vcnt_s8
// CHECK: vcnt.8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vcnt_s8(int8x8_t a) {
  return vcnt_s8(a);
}

// CHECK-LABEL: test_vcnt_p8
// CHECK: vcnt.8 d{{[0-9]+}}, d{{[0-9]+}}
poly8x8_t test_vcnt_p8(poly8x8_t a) {
  return vcnt_p8(a);
}

// CHECK-LABEL: test_vcntq_u8
// CHECK: vcnt.8 q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vcntq_u8(uint8x16_t a) {
  return vcntq_u8(a);
}

// CHECK-LABEL: test_vcntq_s8
// CHECK: vcnt.8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vcntq_s8(int8x16_t a) {
  return vcntq_s8(a);
}

// CHECK-LABEL: test_vcntq_p8
// CHECK: vcnt.8 q{{[0-9]+}}, q{{[0-9]+}}
poly8x16_t test_vcntq_p8(poly8x16_t a) {
  return vcntq_p8(a);
}


// CHECK-LABEL: test_vcombine_s8
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
int8x16_t test_vcombine_s8(int8x8_t a, int8x8_t b) {
  return vcombine_s8(a, b);
}

// CHECK-LABEL: test_vcombine_s16
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
int16x8_t test_vcombine_s16(int16x4_t a, int16x4_t b) {
  return vcombine_s16(a, b);
}

// CHECK-LABEL: test_vcombine_s32
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
int32x4_t test_vcombine_s32(int32x2_t a, int32x2_t b) {
  return vcombine_s32(a, b);
}

// CHECK-LABEL: test_vcombine_s64
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
int64x2_t test_vcombine_s64(int64x1_t a, int64x1_t b) {
  return vcombine_s64(a, b);
}

// CHECK-LABEL: test_vcombine_f16
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
float16x8_t test_vcombine_f16(float16x4_t a, float16x4_t b) {
  return vcombine_f16(a, b);
}

// CHECK-LABEL: test_vcombine_f32
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
float32x4_t test_vcombine_f32(float32x2_t a, float32x2_t b) {
  return vcombine_f32(a, b);
}

// CHECK-LABEL: test_vcombine_u8
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
uint8x16_t test_vcombine_u8(uint8x8_t a, uint8x8_t b) {
  return vcombine_u8(a, b);
}

// CHECK-LABEL: test_vcombine_u16
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
uint16x8_t test_vcombine_u16(uint16x4_t a, uint16x4_t b) {
  return vcombine_u16(a, b);
}

// CHECK-LABEL: test_vcombine_u32
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
uint32x4_t test_vcombine_u32(uint32x2_t a, uint32x2_t b) {
  return vcombine_u32(a, b);
}

// CHECK-LABEL: test_vcombine_u64
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
uint64x2_t test_vcombine_u64(uint64x1_t a, uint64x1_t b) {
  return vcombine_u64(a, b);
}

// CHECK-LABEL: test_vcombine_p8
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
poly8x16_t test_vcombine_p8(poly8x8_t a, poly8x8_t b) {
  return vcombine_p8(a, b);
}

// CHECK-LABEL: test_vcombine_p16
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: vmov d{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
poly16x8_t test_vcombine_p16(poly16x4_t a, poly16x4_t b) {
  return vcombine_p16(a, b);
}


// CHECK-LABEL: test_vcreate_s8
// CHECK: vmov [[REG:d[0-9]+]], r0, r1
// CHECK: vclz.i8 d{{[0-9]+}}, [[REG]]
int8x8_t test_vcreate_s8(uint64_t a) {
  return vclz_s8(vcreate_s8(a));
}

// CHECK-LABEL: test_vcreate_s16
// CHECK: vmov [[REG:d[0-9]+]], r0, r1
// CHECK: vclz.i16 d{{[0-9]+}}, [[REG]]
int16x4_t test_vcreate_s16(uint64_t a) {
  return vclz_s16(vcreate_s16(a));
}

// CHECK-LABEL: test_vcreate_s32
// CHECK: vmov [[REG:d[0-9]+]], r0, r1
// CHECK: vclz.i32 d{{[0-9]+}}, [[REG]]
int32x2_t test_vcreate_s32(uint64_t a) {
  return vclz_s32(vcreate_s32(a));
}

// CHECK-LABEL: test_vcreate_f16
float16x4_t test_vcreate_f16(uint64_t a) {
  return vcreate_f16(a);
}

// CHECK-LABEL: test_vcreate_f32
float32x2_t test_vcreate_f32(uint64_t a) {
  return vcreate_f32(a);
}

// CHECK-LABEL: test_vcreate_u8
// CHECK: vmov [[REG:d[0-9]+]], r0, r1
// CHECK: vclz.i8 d{{[0-9]+}}, [[REG]]
uint8x8_t test_vcreate_u8(uint64_t a) {
  return vclz_s8(vcreate_u8(a));
}

// CHECK-LABEL: test_vcreate_u16
// CHECK: vmov [[REG:d[0-9]+]], r0, r1
// CHECK: vclz.i16 d{{[0-9]+}}, [[REG]]
uint16x4_t test_vcreate_u16(uint64_t a) {
  return vclz_s16(vcreate_u16(a));
}

// CHECK-LABEL: test_vcreate_u32
// CHECK: vmov [[REG:d[0-9]+]], r0, r1
// CHECK: vclz.i32 d{{[0-9]+}}, [[REG]]
uint32x2_t test_vcreate_u32(uint64_t a) {
  return vclz_s32(vcreate_u32(a));
}


// We have two ways of lowering that.  Either with one 'vmov d, r, r' or
// with two 'vmov d[],r'.  LLVM does the latter. We may want to be less
// strict about the matching pattern if it starts causing problem.
// CHECK-LABEL: test_vcreate_u64
// CHECK: vmov.32 [[REG:d[0-9]+]][0], r0
// CHECK: vmov.32 [[REG]][1], r1
uint64x1_t test_vcreate_u64(uint64_t a) {
  uint64x1_t tmp = vcreate_u64(a);
  return vadd_u64(tmp, tmp);

}

// CHECK-LABEL: test_vcreate_p8
// CHECK: vmov [[REG:d[0-9]+]], r0, r1
// CHECK: vcnt.8 d{{[0-9]+}}, [[REG]]
poly8x8_t test_vcreate_p8(uint64_t a) {
  return vcnt_p8(vcreate_p8(a));
}

// CHECK-LABEL: test_vcreate_p16
// CHECK: vmov [[REG:d[0-9]+]], r0, r1
poly16x4_t test_vcreate_p16(uint64_t a) {
  poly16x4_t tmp = vcreate_p16(a);
  return vbsl_p16(tmp, tmp, tmp);
}

// CHECK-LABEL: test_vcreate_s64
// CHECK: vmov.32 [[REG:d[0-9]+]][0], r0
// CHECK: vmov.32 [[REG]][1], r1
int64x1_t test_vcreate_s64(uint64_t a) {
  int64x1_t tmp = vcreate_s64(a);
  return vadd_s64(tmp, tmp);
}


// CHECK-LABEL: test_vcvt_f16_f32
// CHECK: vcvt.f16.f32 d{{[0-9]+}}, q{{[0-9]+}}
float16x4_t test_vcvt_f16_f32(float32x4_t a) {
  return vcvt_f16_f32(a);
}


// CHECK-LABEL: test_vcvt_f32_s32
// CHECK: vcvt.f32.s32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vcvt_f32_s32(int32x2_t a) {
  return vcvt_f32_s32(a);
}

// CHECK-LABEL: test_vcvt_f32_u32
// CHECK: vcvt.f32.u32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vcvt_f32_u32(uint32x2_t a) {
  return vcvt_f32_u32(a);
}

// CHECK-LABEL: test_vcvtq_f32_s32
// CHECK: vcvt.f32.s32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vcvtq_f32_s32(int32x4_t a) {
  return vcvtq_f32_s32(a);
}

// CHECK-LABEL: test_vcvtq_f32_u32
// CHECK: vcvt.f32.u32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vcvtq_f32_u32(uint32x4_t a) {
  return vcvtq_f32_u32(a);
}


// CHECK-LABEL: test_vcvt_f32_f16
// CHECK: vcvt.f32.f16
float32x4_t test_vcvt_f32_f16(float16x4_t a) {
  return vcvt_f32_f16(a);
}


// CHECK-LABEL: test_vcvt_n_f32_s32
// CHECK: vcvt.f32.s32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
float32x2_t test_vcvt_n_f32_s32(int32x2_t a) {
  return vcvt_n_f32_s32(a, 1);
}

// CHECK-LABEL: test_vcvt_n_f32_u32
// CHECK: vcvt.f32.u32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
float32x2_t test_vcvt_n_f32_u32(uint32x2_t a) {
  return vcvt_n_f32_u32(a, 1);
}

// CHECK-LABEL: test_vcvtq_n_f32_s32
// CHECK: vcvt.f32.s32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
float32x4_t test_vcvtq_n_f32_s32(int32x4_t a) {
  return vcvtq_n_f32_s32(a, 3);
}

// CHECK-LABEL: test_vcvtq_n_f32_u32
// CHECK: vcvt.f32.u32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
float32x4_t test_vcvtq_n_f32_u32(uint32x4_t a) {
  return vcvtq_n_f32_u32(a, 3);
}


// CHECK-LABEL: test_vcvt_n_s32_f32
// CHECK: vcvt.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vcvt_n_s32_f32(float32x2_t a) {
  return vcvt_n_s32_f32(a, 1);
}

// CHECK-LABEL: test_vcvtq_n_s32_f32
// CHECK: vcvt.s32.f32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vcvtq_n_s32_f32(float32x4_t a) {
  return vcvtq_n_s32_f32(a, 3);
}


// CHECK-LABEL: test_vcvt_n_u32_f32
// CHECK: vcvt.u32.f32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vcvt_n_u32_f32(float32x2_t a) {
  return vcvt_n_u32_f32(a, 1);
}

// CHECK-LABEL: test_vcvtq_n_u32_f32
// CHECK: vcvt.u32.f32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vcvtq_n_u32_f32(float32x4_t a) {
  return vcvtq_n_u32_f32(a, 3);
}


// CHECK-LABEL: test_vcvt_s32_f32
// CHECK: vcvt.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vcvt_s32_f32(float32x2_t a) {
  return vcvt_s32_f32(a);
}

// CHECK-LABEL: test_vcvtq_s32_f32
// CHECK: vcvt.s32.f32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vcvtq_s32_f32(float32x4_t a) {
  return vcvtq_s32_f32(a);
}


// CHECK-LABEL: test_vcvt_u32_f32
// CHECK: vcvt.u32.f32 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vcvt_u32_f32(float32x2_t a) {
  return vcvt_u32_f32(a);
}

// CHECK-LABEL: test_vcvtq_u32_f32
// CHECK: vcvt.u32.f32 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vcvtq_u32_f32(float32x4_t a) {
  return vcvtq_u32_f32(a);
}


// CHECK-LABEL: test_vdup_lane_u8
// CHECK: vdup.8 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint8x8_t test_vdup_lane_u8(uint8x8_t a) {
  return vdup_lane_u8(a, 7);
}

// CHECK-LABEL: test_vdup_lane_u16
// CHECK: vdup.16 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint16x4_t test_vdup_lane_u16(uint16x4_t a) {
  return vdup_lane_u16(a, 3);
}

// CHECK-LABEL: test_vdup_lane_u32
// CHECK: vdup.32 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x2_t test_vdup_lane_u32(uint32x2_t a) {
  return vdup_lane_u32(a, 1);
}

// CHECK-LABEL: test_vdup_lane_s8
// CHECK: vdup.8 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int8x8_t test_vdup_lane_s8(int8x8_t a) {
  return vdup_lane_s8(a, 7);
}

// CHECK-LABEL: test_vdup_lane_s16
// CHECK: vdup.16 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x4_t test_vdup_lane_s16(int16x4_t a) {
  return vdup_lane_s16(a, 3);
}

// CHECK-LABEL: test_vdup_lane_s32
// CHECK: vdup.32 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x2_t test_vdup_lane_s32(int32x2_t a) {
  return vdup_lane_s32(a, 1);
}

// CHECK-LABEL: test_vdup_lane_p8
// CHECK: vdup.8 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
poly8x8_t test_vdup_lane_p8(poly8x8_t a) {
  return vdup_lane_p8(a, 7);
}

// CHECK-LABEL: test_vdup_lane_p16
// CHECK: vdup.16 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
poly16x4_t test_vdup_lane_p16(poly16x4_t a) {
  return vdup_lane_p16(a, 3);
}

// CHECK-LABEL: test_vdup_lane_f32
// CHECK: vdup.32 d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
float32x2_t test_vdup_lane_f32(float32x2_t a) {
  return vdup_lane_f32(a, 1);
}

// CHECK-LABEL: test_vdupq_lane_u8
// CHECK: vdup.8 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint8x16_t test_vdupq_lane_u8(uint8x8_t a) {
  return vdupq_lane_u8(a, 7);
}

// CHECK-LABEL: test_vdupq_lane_u16
// CHECK: vdup.16 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint16x8_t test_vdupq_lane_u16(uint16x4_t a) {
  return vdupq_lane_u16(a, 3);
}

// CHECK-LABEL: test_vdupq_lane_u32
// CHECK: vdup.32 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x4_t test_vdupq_lane_u32(uint32x2_t a) {
  return vdupq_lane_u32(a, 1);
}

// CHECK-LABEL: test_vdupq_lane_s8
// CHECK: vdup.8 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int8x16_t test_vdupq_lane_s8(int8x8_t a) {
  return vdupq_lane_s8(a, 7);
}

// CHECK-LABEL: test_vdupq_lane_s16
// CHECK: vdup.16 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x8_t test_vdupq_lane_s16(int16x4_t a) {
  return vdupq_lane_s16(a, 3);
}

// CHECK-LABEL: test_vdupq_lane_s32
// CHECK: vdup.32 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vdupq_lane_s32(int32x2_t a) {
  return vdupq_lane_s32(a, 1);
}

// CHECK-LABEL: test_vdupq_lane_p8
// CHECK: vdup.8 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
poly8x16_t test_vdupq_lane_p8(poly8x8_t a) {
  return vdupq_lane_p8(a, 7);
}

// CHECK-LABEL: test_vdupq_lane_p16
// CHECK: vdup.16 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
poly16x8_t test_vdupq_lane_p16(poly16x4_t a) {
  return vdupq_lane_p16(a, 3);
}

// CHECK-LABEL: test_vdupq_lane_f32
// CHECK: vdup.32 q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
float32x4_t test_vdupq_lane_f32(float32x2_t a) {
  return vdupq_lane_f32(a, 1);
}

// CHECK-LABEL: test_vdup_lane_s64
int64x1_t test_vdup_lane_s64(int64x1_t a) {
  return vdup_lane_s64(a, 0);
}

// CHECK-LABEL: test_vdup_lane_u64
uint64x1_t test_vdup_lane_u64(uint64x1_t a) {
  return vdup_lane_u64(a, 0);
}

// CHECK-LABEL: test_vdupq_lane_s64
// CHECK: {{vmov|vdup}}
int64x2_t test_vdupq_lane_s64(int64x1_t a) {
  return vdupq_lane_s64(a, 0);
}

// CHECK-LABEL: test_vdupq_lane_u64
// CHECK: {{vmov|vdup}}
uint64x2_t test_vdupq_lane_u64(uint64x1_t a) {
  return vdupq_lane_u64(a, 0);
}


// CHECK-LABEL: test_vdup_n_u8
// CHECK: vmov 
uint8x8_t test_vdup_n_u8(uint8_t a) {
  return vdup_n_u8(a);
}

// CHECK-LABEL: test_vdup_n_u16
// CHECK: vmov 
uint16x4_t test_vdup_n_u16(uint16_t a) {
  return vdup_n_u16(a);
}

// CHECK-LABEL: test_vdup_n_u32
// CHECK: mov 
uint32x2_t test_vdup_n_u32(uint32_t a) {
  return vdup_n_u32(a);
}

// CHECK-LABEL: test_vdup_n_s8
// CHECK: vmov 
int8x8_t test_vdup_n_s8(int8_t a) {
  return vdup_n_s8(a);
}

// CHECK-LABEL: test_vdup_n_s16
// CHECK: vmov 
int16x4_t test_vdup_n_s16(int16_t a) {
  return vdup_n_s16(a);
}

// CHECK-LABEL: test_vdup_n_s32
// CHECK: mov 
int32x2_t test_vdup_n_s32(int32_t a) {
  return vdup_n_s32(a);
}

// CHECK-LABEL: test_vdup_n_p8
// CHECK: vmov 
poly8x8_t test_vdup_n_p8(poly8_t a) {
  return vdup_n_p8(a);
}

// CHECK-LABEL: test_vdup_n_p16
// CHECK: vmov 
poly16x4_t test_vdup_n_p16(poly16_t a) {
  return vdup_n_p16(a);
}

// CHECK-LABEL: test_vdup_n_f16
// CHECK: vld1.16 {{{d[0-9]+\[\]}}}
float16x4_t test_vdup_n_f16(float16_t *a) {
  return vdup_n_f16(*a);
}

// CHECK-LABEL: test_vdup_n_f32
// CHECK: mov 
float32x2_t test_vdup_n_f32(float32_t a) {
  return vdup_n_f32(a);
}

// CHECK-LABEL: test_vdupq_n_u8
// CHECK: vmov 
uint8x16_t test_vdupq_n_u8(uint8_t a) {
  return vdupq_n_u8(a);
}

// CHECK-LABEL: test_vdupq_n_u16
// CHECK: vmov 
uint16x8_t test_vdupq_n_u16(uint16_t a) {
  return vdupq_n_u16(a);
}

// CHECK-LABEL: test_vdupq_n_u32
// CHECK: vmov 
uint32x4_t test_vdupq_n_u32(uint32_t a) {
  return vdupq_n_u32(a);
}

// CHECK-LABEL: test_vdupq_n_s8
// CHECK: vmov 
int8x16_t test_vdupq_n_s8(int8_t a) {
  return vdupq_n_s8(a);
}

// CHECK-LABEL: test_vdupq_n_s16
// CHECK: vmov 
int16x8_t test_vdupq_n_s16(int16_t a) {
  return vdupq_n_s16(a);
}

// CHECK-LABEL: test_vdupq_n_s32
// CHECK: vmov 
int32x4_t test_vdupq_n_s32(int32_t a) {
  return vdupq_n_s32(a);
}

// CHECK-LABEL: test_vdupq_n_p8
// CHECK: vmov 
poly8x16_t test_vdupq_n_p8(poly8_t a) {
  return vdupq_n_p8(a);
}

// CHECK-LABEL: test_vdupq_n_p16
// CHECK: vmov 
poly16x8_t test_vdupq_n_p16(poly16_t a) {
  return vdupq_n_p16(a);
}

// CHECK-LABEL: test_vdupq_n_f16
// CHECK: vld1.16 {{{d[0-9]+\[\], d[0-9]+\[\]}}}
float16x8_t test_vdupq_n_f16(float16_t *a) {
  return vdupq_n_f16(*a);
}

// CHECK-LABEL: test_vdupq_n_f32
// CHECK: vmov 
float32x4_t test_vdupq_n_f32(float32_t a) {
  return vdupq_n_f32(a);
}

// CHECK-LABEL: test_vdup_n_s64
// CHECK: vmov
int64x1_t test_vdup_n_s64(int64_t a) {
  int64x1_t tmp = vdup_n_s64(a);
  return vadd_s64(tmp, tmp);
}

// CHECK-LABEL: test_vdup_n_u64
// CHECK: vmov
uint64x1_t test_vdup_n_u64(uint64_t a) {
  int64x1_t tmp = vdup_n_u64(a);
  return vadd_s64(tmp, tmp);

}

// CHECK-LABEL: test_vdupq_n_s64
// CHECK: vmov
int64x2_t test_vdupq_n_s64(int64_t a) {
  int64x2_t tmp = vdupq_n_s64(a);
  return vaddq_s64(tmp, tmp);
}

// CHECK-LABEL: test_vdupq_n_u64
// CHECK: vmov
uint64x2_t test_vdupq_n_u64(uint64_t a) {
  int64x2_t tmp = vdupq_n_u64(a);
  return vaddq_u64(tmp, tmp);
}


// CHECK-LABEL: test_veor_s8
// CHECK: veor d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_veor_s8(int8x8_t a, int8x8_t b) {
  return veor_s8(a, b);
}

// CHECK-LABEL: test_veor_s16
// CHECK: veor d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_veor_s16(int16x4_t a, int16x4_t b) {
  return veor_s16(a, b);
}

// CHECK-LABEL: test_veor_s32
// CHECK: veor d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_veor_s32(int32x2_t a, int32x2_t b) {
  return veor_s32(a, b);
}

// CHECK-LABEL: test_veor_s64
// CHECK: veor d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_veor_s64(int64x1_t a, int64x1_t b) {
  return veor_s64(a, b);
}

// CHECK-LABEL: test_veor_u8
// CHECK: veor d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_veor_u8(uint8x8_t a, uint8x8_t b) {
  return veor_u8(a, b);
}

// CHECK-LABEL: test_veor_u16
// CHECK: veor d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_veor_u16(uint16x4_t a, uint16x4_t b) {
  return veor_u16(a, b);
}

// CHECK-LABEL: test_veor_u32
// CHECK: veor d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_veor_u32(uint32x2_t a, uint32x2_t b) {
  return veor_u32(a, b);
}

// CHECK-LABEL: test_veor_u64
// CHECK: veor d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_veor_u64(uint64x1_t a, uint64x1_t b) {
  return veor_u64(a, b);
}

// CHECK-LABEL: test_veorq_s8
// CHECK: veor q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_veorq_s8(int8x16_t a, int8x16_t b) {
  return veorq_s8(a, b);
}

// CHECK-LABEL: test_veorq_s16
// CHECK: veor q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_veorq_s16(int16x8_t a, int16x8_t b) {
  return veorq_s16(a, b);
}

// CHECK-LABEL: test_veorq_s32
// CHECK: veor q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_veorq_s32(int32x4_t a, int32x4_t b) {
  return veorq_s32(a, b);
}

// CHECK-LABEL: test_veorq_s64
// CHECK: veor q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_veorq_s64(int64x2_t a, int64x2_t b) {
  return veorq_s64(a, b);
}

// CHECK-LABEL: test_veorq_u8
// CHECK: veor q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_veorq_u8(uint8x16_t a, uint8x16_t b) {
  return veorq_u8(a, b);
}

// CHECK-LABEL: test_veorq_u16
// CHECK: veor q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_veorq_u16(uint16x8_t a, uint16x8_t b) {
  return veorq_u16(a, b);
}

// CHECK-LABEL: test_veorq_u32
// CHECK: veor q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_veorq_u32(uint32x4_t a, uint32x4_t b) {
  return veorq_u32(a, b);
}

// CHECK-LABEL: test_veorq_u64
// CHECK: veor q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_veorq_u64(uint64x2_t a, uint64x2_t b) {
  return veorq_u64(a, b);
}


// CHECK-LABEL: test_vext_s8
// CHECK: vext.8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vext_s8(int8x8_t a, int8x8_t b) {
  return vext_s8(a, b, 7);
}

// CHECK-LABEL: test_vext_u8
// CHECK: vext.8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vext_u8(uint8x8_t a, uint8x8_t b) {
  return vext_u8(a, b, 7);
}

// CHECK-LABEL: test_vext_p8
// CHECK: vext.8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
poly8x8_t test_vext_p8(poly8x8_t a, poly8x8_t b) {
  return vext_p8(a, b, 7);
}

// CHECK-LABEL: test_vext_s16
// CHECK: vext.16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vext_s16(int16x4_t a, int16x4_t b) {
  return vext_s16(a, b, 3);
}

// CHECK-LABEL: test_vext_u16
// CHECK: vext.16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vext_u16(uint16x4_t a, uint16x4_t b) {
  return vext_u16(a, b, 3);
}

// CHECK-LABEL: test_vext_p16
// CHECK: vext.16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
poly16x4_t test_vext_p16(poly16x4_t a, poly16x4_t b) {
  return vext_p16(a, b, 3);
}

// CHECK-LABEL: test_vext_s32
// CHECK: vext.32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vext_s32(int32x2_t a, int32x2_t b) {
  return vext_s32(a, b, 1);
}

// CHECK-LABEL: test_vext_u32
// CHECK: vext.32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vext_u32(uint32x2_t a, uint32x2_t b) {
  return vext_u32(a, b, 1);
}

// CHECK-LABEL: test_vext_s64
int64x1_t test_vext_s64(int64x1_t a, int64x1_t b) {
  return vext_s64(a, b, 0);
}

// CHECK-LABEL: test_vext_u64
uint64x1_t test_vext_u64(uint64x1_t a, uint64x1_t b) {
  return vext_u64(a, b, 0);
}

// CHECK-LABEL: test_vext_f32
// CHECK: vext.32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
float32x2_t test_vext_f32(float32x2_t a, float32x2_t b) {
  return vext_f32(a, b, 1);
}

// CHECK-LABEL: test_vextq_s8
// CHECK: vext.8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vextq_s8(int8x16_t a, int8x16_t b) {
  return vextq_s8(a, b, 15);
}

// CHECK-LABEL: test_vextq_u8
// CHECK: vext.8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vextq_u8(uint8x16_t a, uint8x16_t b) {
  return vextq_u8(a, b, 15);
}

// CHECK-LABEL: test_vextq_p8
// CHECK: vext.8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
poly8x16_t test_vextq_p8(poly8x16_t a, poly8x16_t b) {
  return vextq_p8(a, b, 15);
}

// CHECK-LABEL: test_vextq_s16
// CHECK: vext.16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vextq_s16(int16x8_t a, int16x8_t b) {
  return vextq_s16(a, b, 7);
}

// CHECK-LABEL: test_vextq_u16
// CHECK: vext.16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vextq_u16(uint16x8_t a, uint16x8_t b) {
  return vextq_u16(a, b, 7);
}

// CHECK-LABEL: test_vextq_p16
// CHECK: vext.16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
poly16x8_t test_vextq_p16(poly16x8_t a, poly16x8_t b) {
  return vextq_p16(a, b, 7);
}

// CHECK-LABEL: test_vextq_s32
// CHECK: vext.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vextq_s32(int32x4_t a, int32x4_t b) {
  return vextq_s32(a, b, 3);
}

// CHECK-LABEL: test_vextq_u32
// CHECK: vext.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vextq_u32(uint32x4_t a, uint32x4_t b) {
  return vextq_u32(a, b, 3);
}

// CHECK-LABEL: test_vextq_s64
// CHECK: {{vmov|vdup}}
int64x2_t test_vextq_s64(int64x2_t a, int64x2_t b) {
  return vextq_s64(a, b, 1);
}

// CHECK-LABEL: test_vextq_u64
// CHECK: {{vmov|vdup}}
uint64x2_t test_vextq_u64(uint64x2_t a, uint64x2_t b) {
  return vextq_u64(a, b, 1);
}

// CHECK-LABEL: test_vextq_f32
// CHECK: vext.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
float32x4_t test_vextq_f32(float32x4_t a, float32x4_t b) {
  return vextq_f32(a, b, 3);
}


// CHECK-LABEL: test_vfma_f32
// CHECK: vfma.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vfma_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
  return vfma_f32(a, b, c);
}

// CHECK-LABEL: test_vfmaq_f32
// CHECK: vfma.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return vfmaq_f32(a, b, c);
}


// CHECK-LABEL: test_vget_high_s8
int8x8_t test_vget_high_s8(int8x16_t a) {
  return vget_high_s8(a);
}

// CHECK-LABEL: test_vget_high_s16
int16x4_t test_vget_high_s16(int16x8_t a) {
  return vget_high_s16(a);
}

// CHECK-LABEL: test_vget_high_s32
int32x2_t test_vget_high_s32(int32x4_t a) {
  return vget_high_s32(a);
}

// CHECK-LABEL: test_vget_high_s64
int64x1_t test_vget_high_s64(int64x2_t a) {
  return vget_high_s64(a);
}

// CHECK-LABEL: test_vget_high_f16
float16x4_t test_vget_high_f16(float16x8_t a) {
  return vget_high_f16(a);
}

// CHECK-LABEL: test_vget_high_f32
float32x2_t test_vget_high_f32(float32x4_t a) {
  return vget_high_f32(a);
}

// CHECK-LABEL: test_vget_high_u8
uint8x8_t test_vget_high_u8(uint8x16_t a) {
  return vget_high_u8(a);
}

// CHECK-LABEL: test_vget_high_u16
uint16x4_t test_vget_high_u16(uint16x8_t a) {
  return vget_high_u16(a);
}

// CHECK-LABEL: test_vget_high_u32
uint32x2_t test_vget_high_u32(uint32x4_t a) {
  return vget_high_u32(a);
}

// CHECK-LABEL: test_vget_high_u64
uint64x1_t test_vget_high_u64(uint64x2_t a) {
  return vget_high_u64(a);
}

// CHECK-LABEL: test_vget_high_p8
poly8x8_t test_vget_high_p8(poly8x16_t a) {
  return vget_high_p8(a);
}

// CHECK-LABEL: test_vget_high_p16
poly16x4_t test_vget_high_p16(poly16x8_t a) {
  return vget_high_p16(a);
}


// CHECK-LABEL: test_vget_lane_u8
// CHECK: vmov 
uint8_t test_vget_lane_u8(uint8x8_t a) {
  return vget_lane_u8(a, 7);
}

// CHECK-LABEL: test_vget_lane_u16
// CHECK: vmov 
uint16_t test_vget_lane_u16(uint16x4_t a) {
  return vget_lane_u16(a, 3);
}

// CHECK-LABEL: test_vget_lane_u32
// CHECK: mov 
uint32_t test_vget_lane_u32(uint32x2_t a) {
  return vget_lane_u32(a, 1);
}

// CHECK-LABEL: test_vget_lane_s8
// CHECK: vmov 
int8_t test_vget_lane_s8(int8x8_t a) {
  return vget_lane_s8(a, 7);
}

// CHECK-LABEL: test_vget_lane_s16
// CHECK: vmov 
int16_t test_vget_lane_s16(int16x4_t a) {
  return vget_lane_s16(a, 3);
}

// CHECK-LABEL: test_vget_lane_s32
// CHECK: mov 
int32_t test_vget_lane_s32(int32x2_t a) {
  return vget_lane_s32(a, 1);
}

// CHECK-LABEL: test_vget_lane_p8
// CHECK: vmov 
poly8_t test_vget_lane_p8(poly8x8_t a) {
  return vget_lane_p8(a, 7);
}

// CHECK-LABEL: test_vget_lane_p16
// CHECK: vmov 
poly16_t test_vget_lane_p16(poly16x4_t a) {
  return vget_lane_p16(a, 3);
}

// CHECK-LABEL: test_vget_lane_f32
// CHECK: vmov 
float32_t test_vget_lane_f32(float32x2_t a) {
  return vget_lane_f32(a, 1);
}

// CHECK-LABEL: test_vget_lane_f16
// CHECK: vmov 
float32_t test_vget_lane_f16(float16x4_t a) {
  return vget_lane_f16(a, 1);
}

// CHECK-LABEL: test_vgetq_lane_u8
// CHECK: vmov 
uint8_t test_vgetq_lane_u8(uint8x16_t a) {
  return vgetq_lane_u8(a, 15);
}

// CHECK-LABEL: test_vgetq_lane_u16
// CHECK: vmov 
uint16_t test_vgetq_lane_u16(uint16x8_t a) {
  return vgetq_lane_u16(a, 7);
}

// CHECK-LABEL: test_vgetq_lane_u32
// CHECK: vmov 
uint32_t test_vgetq_lane_u32(uint32x4_t a) {
  return vgetq_lane_u32(a, 3);
}

// CHECK-LABEL: test_vgetq_lane_s8
// CHECK: vmov 
int8_t test_vgetq_lane_s8(int8x16_t a) {
  return vgetq_lane_s8(a, 15);
}

// CHECK-LABEL: test_vgetq_lane_s16
// CHECK: vmov 
int16_t test_vgetq_lane_s16(int16x8_t a) {
  return vgetq_lane_s16(a, 7);
}

// CHECK-LABEL: test_vgetq_lane_s32
// CHECK: vmov 
int32_t test_vgetq_lane_s32(int32x4_t a) {
  return vgetq_lane_s32(a, 3);
}

// CHECK-LABEL: test_vgetq_lane_p8
// CHECK: vmov 
poly8_t test_vgetq_lane_p8(poly8x16_t a) {
  return vgetq_lane_p8(a, 15);
}

// CHECK-LABEL: test_vgetq_lane_p16
// CHECK: vmov 
poly16_t test_vgetq_lane_p16(poly16x8_t a) {
  return vgetq_lane_p16(a, 7);
}

// CHECK-LABEL: test_vgetq_lane_f32
// CHECK: vmov 
float32_t test_vgetq_lane_f32(float32x4_t a) {
  return vgetq_lane_f32(a, 3);
}

// CHECK-LABEL: test_vgetq_lane_f16
// CHECK: vmov 
float32_t test_vgetq_lane_f16(float16x8_t a) {
  return vgetq_lane_f16(a, 3);
}

// CHECK-LABEL: test_vget_lane_s64
// The optimizer is able to remove all moves now.
int64_t test_vget_lane_s64(int64x1_t a) {
  return vget_lane_s64(a, 0);
}

// CHECK-LABEL: test_vget_lane_u64
// The optimizer is able to remove all moves now.
uint64_t test_vget_lane_u64(uint64x1_t a) {
  return vget_lane_u64(a, 0);
}

// CHECK-LABEL: test_vgetq_lane_s64
// CHECK: vmov 
int64_t test_vgetq_lane_s64(int64x2_t a) {
  return vgetq_lane_s64(a, 1);
}

// CHECK-LABEL: test_vgetq_lane_u64
// CHECK: vmov 
uint64_t test_vgetq_lane_u64(uint64x2_t a) {
  return vgetq_lane_u64(a, 1);
}


// CHECK-LABEL: test_vget_low_s8
int8x8_t test_vget_low_s8(int8x16_t a) {
  return vget_low_s8(a);
}

// CHECK-LABEL: test_vget_low_s16
int16x4_t test_vget_low_s16(int16x8_t a) {
  return vget_low_s16(a);
}

// CHECK-LABEL: test_vget_low_s32
int32x2_t test_vget_low_s32(int32x4_t a) {
  return vget_low_s32(a);
}

// CHECK-LABEL: test_vget_low_s64
int64x1_t test_vget_low_s64(int64x2_t a) {
  return vget_low_s64(a);
}

// CHECK-LABEL: test_vget_low_f16
float16x4_t test_vget_low_f16(float16x8_t a) {
  return vget_low_f16(a);
}

// CHECK-LABEL: test_vget_low_f32
float32x2_t test_vget_low_f32(float32x4_t a) {
  return vget_low_f32(a);
}

// CHECK-LABEL: test_vget_low_u8
uint8x8_t test_vget_low_u8(uint8x16_t a) {
  return vget_low_u8(a);
}

// CHECK-LABEL: test_vget_low_u16
uint16x4_t test_vget_low_u16(uint16x8_t a) {
  return vget_low_u16(a);
}

// CHECK-LABEL: test_vget_low_u32
uint32x2_t test_vget_low_u32(uint32x4_t a) {
  return vget_low_u32(a);
}

// CHECK-LABEL: test_vget_low_u64
uint64x1_t test_vget_low_u64(uint64x2_t a) {
  return vget_low_u64(a);
}

// CHECK-LABEL: test_vget_low_p8
poly8x8_t test_vget_low_p8(poly8x16_t a) {
  return vget_low_p8(a);
}

// CHECK-LABEL: test_vget_low_p16
poly16x4_t test_vget_low_p16(poly16x8_t a) {
  return vget_low_p16(a);
}


// CHECK-LABEL: test_vhadd_s8
// CHECK: vhadd.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vhadd_s8(int8x8_t a, int8x8_t b) {
  return vhadd_s8(a, b);
}

// CHECK-LABEL: test_vhadd_s16
// CHECK: vhadd.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vhadd_s16(int16x4_t a, int16x4_t b) {
  return vhadd_s16(a, b);
}

// CHECK-LABEL: test_vhadd_s32
// CHECK: vhadd.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vhadd_s32(int32x2_t a, int32x2_t b) {
  return vhadd_s32(a, b);
}

// CHECK-LABEL: test_vhadd_u8
// CHECK: vhadd.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vhadd_u8(uint8x8_t a, uint8x8_t b) {
  return vhadd_u8(a, b);
}

// CHECK-LABEL: test_vhadd_u16
// CHECK: vhadd.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vhadd_u16(uint16x4_t a, uint16x4_t b) {
  return vhadd_u16(a, b);
}

// CHECK-LABEL: test_vhadd_u32
// CHECK: vhadd.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vhadd_u32(uint32x2_t a, uint32x2_t b) {
  return vhadd_u32(a, b);
}

// CHECK-LABEL: test_vhaddq_s8
// CHECK: vhadd.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vhaddq_s8(int8x16_t a, int8x16_t b) {
  return vhaddq_s8(a, b);
}

// CHECK-LABEL: test_vhaddq_s16
// CHECK: vhadd.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vhaddq_s16(int16x8_t a, int16x8_t b) {
  return vhaddq_s16(a, b);
}

// CHECK-LABEL: test_vhaddq_s32
// CHECK: vhadd.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vhaddq_s32(int32x4_t a, int32x4_t b) {
  return vhaddq_s32(a, b);
}

// CHECK-LABEL: test_vhaddq_u8
// CHECK: vhadd.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vhaddq_u8(uint8x16_t a, uint8x16_t b) {
  return vhaddq_u8(a, b);
}

// CHECK-LABEL: test_vhaddq_u16
// CHECK: vhadd.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vhaddq_u16(uint16x8_t a, uint16x8_t b) {
  return vhaddq_u16(a, b);
}

// CHECK-LABEL: test_vhaddq_u32
// CHECK: vhadd.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vhaddq_u32(uint32x4_t a, uint32x4_t b) {
  return vhaddq_u32(a, b);
}


// CHECK-LABEL: test_vhsub_s8
// CHECK: vhsub.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vhsub_s8(int8x8_t a, int8x8_t b) {
  return vhsub_s8(a, b);
}

// CHECK-LABEL: test_vhsub_s16
// CHECK: vhsub.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vhsub_s16(int16x4_t a, int16x4_t b) {
  return vhsub_s16(a, b);
}

// CHECK-LABEL: test_vhsub_s32
// CHECK: vhsub.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vhsub_s32(int32x2_t a, int32x2_t b) {
  return vhsub_s32(a, b);
}

// CHECK-LABEL: test_vhsub_u8
// CHECK: vhsub.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vhsub_u8(uint8x8_t a, uint8x8_t b) {
  return vhsub_u8(a, b);
}

// CHECK-LABEL: test_vhsub_u16
// CHECK: vhsub.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vhsub_u16(uint16x4_t a, uint16x4_t b) {
  return vhsub_u16(a, b);
}

// CHECK-LABEL: test_vhsub_u32
// CHECK: vhsub.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vhsub_u32(uint32x2_t a, uint32x2_t b) {
  return vhsub_u32(a, b);
}

// CHECK-LABEL: test_vhsubq_s8
// CHECK: vhsub.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vhsubq_s8(int8x16_t a, int8x16_t b) {
  return vhsubq_s8(a, b);
}

// CHECK-LABEL: test_vhsubq_s16
// CHECK: vhsub.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vhsubq_s16(int16x8_t a, int16x8_t b) {
  return vhsubq_s16(a, b);
}

// CHECK-LABEL: test_vhsubq_s32
// CHECK: vhsub.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vhsubq_s32(int32x4_t a, int32x4_t b) {
  return vhsubq_s32(a, b);
}

// CHECK-LABEL: test_vhsubq_u8
// CHECK: vhsub.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vhsubq_u8(uint8x16_t a, uint8x16_t b) {
  return vhsubq_u8(a, b);
}

// CHECK-LABEL: test_vhsubq_u16
// CHECK: vhsub.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vhsubq_u16(uint16x8_t a, uint16x8_t b) {
  return vhsubq_u16(a, b);
}

// CHECK-LABEL: test_vhsubq_u32
// CHECK: vhsub.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vhsubq_u32(uint32x4_t a, uint32x4_t b) {
  return vhsubq_u32(a, b);
}


// CHECK-LABEL: test_vld1q_u8
// CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint8x16_t test_vld1q_u8(uint8_t const * a) {
  return vld1q_u8(a);
}

// CHECK-LABEL: test_vld1q_u16
// CHECK: vld1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint16x8_t test_vld1q_u16(uint16_t const * a) {
  return vld1q_u16(a);
}

// CHECK-LABEL: test_vld1q_u32
// CHECK: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint32x4_t test_vld1q_u32(uint32_t const * a) {
  return vld1q_u32(a);
}

// CHECK-LABEL: test_vld1q_u64
// CHECK: vld1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}{{(:64)?}}]
uint64x2_t test_vld1q_u64(uint64_t const * a) {
  return vld1q_u64(a);
}

// CHECK-LABEL: test_vld1q_s8
// CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int8x16_t test_vld1q_s8(int8_t const * a) {
  return vld1q_s8(a);
}

// CHECK-LABEL: test_vld1q_s16
// CHECK: vld1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int16x8_t test_vld1q_s16(int16_t const * a) {
  return vld1q_s16(a);
}

// CHECK-LABEL: test_vld1q_s32
// CHECK: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int32x4_t test_vld1q_s32(int32_t const * a) {
  return vld1q_s32(a);
}

// CHECK-LABEL: test_vld1q_s64
// CHECK: vld1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}{{(:64)?}}]
int64x2_t test_vld1q_s64(int64_t const * a) {
  return vld1q_s64(a);
}

// CHECK-LABEL: test_vld1q_f16
// CHECK: vld1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float16x8_t test_vld1q_f16(float16_t const * a) {
  return vld1q_f16(a);
}

// CHECK-LABEL: test_vld1q_f32
// CHECK: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float32x4_t test_vld1q_f32(float32_t const * a) {
  return vld1q_f32(a);
}

// CHECK-LABEL: test_vld1q_p8
// CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly8x16_t test_vld1q_p8(poly8_t const * a) {
  return vld1q_p8(a);
}

// CHECK-LABEL: test_vld1q_p16
// CHECK: vld1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly16x8_t test_vld1q_p16(poly16_t const * a) {
  return vld1q_p16(a);
}

// CHECK-LABEL: test_vld1_u8
// CHECK: vld1.8 {d{{[0-9]+}}}, [r{{[0-9]+}}]
uint8x8_t test_vld1_u8(uint8_t const * a) {
  return vld1_u8(a);
}

// CHECK-LABEL: test_vld1_u16
// CHECK: vld1.16 {d{{[0-9]+}}}, [r{{[0-9]+}}]
uint16x4_t test_vld1_u16(uint16_t const * a) {
  return vld1_u16(a);
}

// CHECK-LABEL: test_vld1_u32
// CHECK: vld1.32 {d{{[0-9]+}}}, [r{{[0-9]+}}]
uint32x2_t test_vld1_u32(uint32_t const * a) {
  return vld1_u32(a);
}

// CHECK-LABEL: test_vld1_u64
// CHECK: vld1.64 {d{{[0-9]+}}}, [r{{[0-9]+}}{{(:64)?}}]
uint64x1_t test_vld1_u64(uint64_t const * a) {
  return vld1_u64(a);
}

// CHECK-LABEL: test_vld1_s8
// CHECK: vld1.8 {d{{[0-9]+}}}, [r{{[0-9]+}}]
int8x8_t test_vld1_s8(int8_t const * a) {
  return vld1_s8(a);
}

// CHECK-LABEL: test_vld1_s16
// CHECK: vld1.16 {d{{[0-9]+}}}, [r{{[0-9]+}}]
int16x4_t test_vld1_s16(int16_t const * a) {
  return vld1_s16(a);
}

// CHECK-LABEL: test_vld1_s32
// CHECK: vld1.32 {d{{[0-9]+}}}, [r{{[0-9]+}}]
int32x2_t test_vld1_s32(int32_t const * a) {
  return vld1_s32(a);
}

// CHECK-LABEL: test_vld1_s64
// CHECK: vld1.64 {d{{[0-9]+}}}, [r{{[0-9]+}}{{(:64)?}}]
int64x1_t test_vld1_s64(int64_t const * a) {
  return vld1_s64(a);
}

// CHECK-LABEL: test_vld1_f16
// CHECK: vld1.16 {d{{[0-9]+}}}, [r{{[0-9]+}}]
float16x4_t test_vld1_f16(float16_t const * a) {
  return vld1_f16(a);
}

// CHECK-LABEL: test_vld1_f32
// CHECK: vld1.32 {d{{[0-9]+}}}, [r{{[0-9]+}}]
float32x2_t test_vld1_f32(float32_t const * a) {
  return vld1_f32(a);
}

// CHECK-LABEL: test_vld1_p8
// CHECK: vld1.8 {d{{[0-9]+}}}, [r{{[0-9]+}}]
poly8x8_t test_vld1_p8(poly8_t const * a) {
  return vld1_p8(a);
}

// CHECK-LABEL: test_vld1_p16
// CHECK: vld1.16 {d{{[0-9]+}}}, [r{{[0-9]+}}]
poly16x4_t test_vld1_p16(poly16_t const * a) {
  return vld1_p16(a);
}


// CHECK-LABEL: test_vld1q_dup_u8
// CHECK: vld1.8 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint8x16_t test_vld1q_dup_u8(uint8_t const * a) {
  return vld1q_dup_u8(a);
}

// CHECK-LABEL: test_vld1q_dup_u16
// CHECK: vld1.16 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}:16]
uint16x8_t test_vld1q_dup_u16(uint16_t const * a) {
  return vld1q_dup_u16(a);
}

// CHECK-LABEL: test_vld1q_dup_u32
// CHECK: vld1.32 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}:32]
uint32x4_t test_vld1q_dup_u32(uint32_t const * a) {
  return vld1q_dup_u32(a);
}

// CHECK-LABEL: test_vld1q_dup_u64
// CHECK: {{ldr|vldr|vmov}}
uint64x2_t test_vld1q_dup_u64(uint64_t const * a) {
  return vld1q_dup_u64(a);
}

// CHECK-LABEL: test_vld1q_dup_s8
// CHECK: vld1.8 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int8x16_t test_vld1q_dup_s8(int8_t const * a) {
  return vld1q_dup_s8(a);
}

// CHECK-LABEL: test_vld1q_dup_s16
// CHECK: vld1.16 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}:16]
int16x8_t test_vld1q_dup_s16(int16_t const * a) {
  return vld1q_dup_s16(a);
}

// CHECK-LABEL: test_vld1q_dup_s32
// CHECK: vld1.32 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}:32]
int32x4_t test_vld1q_dup_s32(int32_t const * a) {
  return vld1q_dup_s32(a);
}

// CHECK-LABEL: test_vld1q_dup_s64
// CHECK: {{ldr|vldr|vmov}}
int64x2_t test_vld1q_dup_s64(int64_t const * a) {
  return vld1q_dup_s64(a);
}

// CHECK-LABEL: test_vld1q_dup_f16
// CHECK: vld1.16 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}:16]
float16x8_t test_vld1q_dup_f16(float16_t const * a) {
  return vld1q_dup_f16(a);
}

// CHECK-LABEL: test_vld1q_dup_f32
// CHECK: vld1.32 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}:32]
float32x4_t test_vld1q_dup_f32(float32_t const * a) {
  return vld1q_dup_f32(a);
}

// CHECK-LABEL: test_vld1q_dup_p8
// CHECK: vld1.8 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
poly8x16_t test_vld1q_dup_p8(poly8_t const * a) {
  return vld1q_dup_p8(a);
}

// CHECK-LABEL: test_vld1q_dup_p16
// CHECK: vld1.16 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}:16]
poly16x8_t test_vld1q_dup_p16(poly16_t const * a) {
  return vld1q_dup_p16(a);
}

// CHECK-LABEL: test_vld1_dup_u8
// CHECK: vld1.8 {d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint8x8_t test_vld1_dup_u8(uint8_t const * a) {
  return vld1_dup_u8(a);
}

// CHECK-LABEL: test_vld1_dup_u16
// CHECK: vld1.16 {d{{[0-9]+}}[]}, [r{{[0-9]+}}:16]
uint16x4_t test_vld1_dup_u16(uint16_t const * a) {
  return vld1_dup_u16(a);
}

// CHECK-LABEL: test_vld1_dup_u32
// CHECK: vld1.32 {d{{[0-9]+}}[]}, [r{{[0-9]+}}:32]
uint32x2_t test_vld1_dup_u32(uint32_t const * a) {
  return vld1_dup_u32(a);
}

// CHECK-LABEL: test_vld1_dup_u64
// CHECK: {{ldr|vldr|vmov}}
uint64x1_t test_vld1_dup_u64(uint64_t const * a) {
  return vld1_dup_u64(a);
}

// CHECK-LABEL: test_vld1_dup_s8
// CHECK: vld1.8 {d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int8x8_t test_vld1_dup_s8(int8_t const * a) {
  return vld1_dup_s8(a);
}

// CHECK-LABEL: test_vld1_dup_s16
// CHECK: vld1.16 {d{{[0-9]+}}[]}, [r{{[0-9]+}}:16]
int16x4_t test_vld1_dup_s16(int16_t const * a) {
  return vld1_dup_s16(a);
}

// CHECK-LABEL: test_vld1_dup_s32
// CHECK: vld1.32 {d{{[0-9]+}}[]}, [r{{[0-9]+}}:32]
int32x2_t test_vld1_dup_s32(int32_t const * a) {
  return vld1_dup_s32(a);
}

// CHECK-LABEL: test_vld1_dup_s64
// CHECK: {{ldr|vldr|vmov}}
int64x1_t test_vld1_dup_s64(int64_t const * a) {
  return vld1_dup_s64(a);
}

// CHECK-LABEL: test_vld1_dup_f16
// CHECK: vld1.16 {d{{[0-9]+}}[]}, [r{{[0-9]+}}:16]
float16x4_t test_vld1_dup_f16(float16_t const * a) {
  return vld1_dup_f16(a);
}

// CHECK-LABEL: test_vld1_dup_f32
// CHECK: vld1.32 {d{{[0-9]+}}[]}, [r{{[0-9]+}}:32]
float32x2_t test_vld1_dup_f32(float32_t const * a) {
  return vld1_dup_f32(a);
}

// CHECK-LABEL: test_vld1_dup_p8
// CHECK: vld1.8 {d{{[0-9]+}}[]}, [r{{[0-9]+}}]
poly8x8_t test_vld1_dup_p8(poly8_t const * a) {
  return vld1_dup_p8(a);
}

// CHECK-LABEL: test_vld1_dup_p16
// CHECK: vld1.16 {d{{[0-9]+}}[]}, [r{{[0-9]+}}:16]
poly16x4_t test_vld1_dup_p16(poly16_t const * a) {
  return vld1_dup_p16(a);
}


// CHECK-LABEL: test_vld1q_lane_u8
// CHECK: vld1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint8x16_t test_vld1q_lane_u8(uint8_t const * a, uint8x16_t b) {
  return vld1q_lane_u8(a, b, 15);
}

// CHECK-LABEL: test_vld1q_lane_u16
// CHECK: vld1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
uint16x8_t test_vld1q_lane_u16(uint16_t const * a, uint16x8_t b) {
  return vld1q_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vld1q_lane_u32
// CHECK: vld1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
uint32x4_t test_vld1q_lane_u32(uint32_t const * a, uint32x4_t b) {
  return vld1q_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vld1q_lane_u64
// CHECK: {{ldr|vldr|vmov}}
uint64x2_t test_vld1q_lane_u64(uint64_t const * a, uint64x2_t b) {
  return vld1q_lane_u64(a, b, 1);
}

// CHECK-LABEL: test_vld1q_lane_s8
// CHECK: vld1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int8x16_t test_vld1q_lane_s8(int8_t const * a, int8x16_t b) {
  return vld1q_lane_s8(a, b, 15);
}

// CHECK-LABEL: test_vld1q_lane_s16
// CHECK: vld1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
int16x8_t test_vld1q_lane_s16(int16_t const * a, int16x8_t b) {
  return vld1q_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vld1q_lane_s32
// CHECK: vld1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
int32x4_t test_vld1q_lane_s32(int32_t const * a, int32x4_t b) {
  return vld1q_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vld1q_lane_s64
// CHECK: {{ldr|vldr|vmov}}
int64x2_t test_vld1q_lane_s64(int64_t const * a, int64x2_t b) {
  return vld1q_lane_s64(a, b, 1);
}

// CHECK-LABEL: test_vld1q_lane_f16
// CHECK: vld1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
float16x8_t test_vld1q_lane_f16(float16_t const * a, float16x8_t b) {
  return vld1q_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vld1q_lane_f32
// CHECK: vld1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
float32x4_t test_vld1q_lane_f32(float32_t const * a, float32x4_t b) {
  return vld1q_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vld1q_lane_p8
// CHECK: vld1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly8x16_t test_vld1q_lane_p8(poly8_t const * a, poly8x16_t b) {
  return vld1q_lane_p8(a, b, 15);
}

// CHECK-LABEL: test_vld1q_lane_p16
// CHECK: vld1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
poly16x8_t test_vld1q_lane_p16(poly16_t const * a, poly16x8_t b) {
  return vld1q_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vld1_lane_u8
// CHECK: vld1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint8x8_t test_vld1_lane_u8(uint8_t const * a, uint8x8_t b) {
  return vld1_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vld1_lane_u16
// CHECK: vld1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
uint16x4_t test_vld1_lane_u16(uint16_t const * a, uint16x4_t b) {
  return vld1_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vld1_lane_u32
// CHECK: vld1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
uint32x2_t test_vld1_lane_u32(uint32_t const * a, uint32x2_t b) {
  return vld1_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vld1_lane_u64
// CHECK: {{ldr|vldr|vmov}}
uint64x1_t test_vld1_lane_u64(uint64_t const * a, uint64x1_t b) {
  return vld1_lane_u64(a, b, 0);
}

// CHECK-LABEL: test_vld1_lane_s8
// CHECK: vld1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int8x8_t test_vld1_lane_s8(int8_t const * a, int8x8_t b) {
  return vld1_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vld1_lane_s16
// CHECK: vld1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
int16x4_t test_vld1_lane_s16(int16_t const * a, int16x4_t b) {
  return vld1_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vld1_lane_s32
// CHECK: vld1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
int32x2_t test_vld1_lane_s32(int32_t const * a, int32x2_t b) {
  return vld1_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vld1_lane_s64
// CHECK: {{ldr|vldr|vmov}}
int64x1_t test_vld1_lane_s64(int64_t const * a, int64x1_t b) {
  return vld1_lane_s64(a, b, 0);
}

// CHECK-LABEL: test_vld1_lane_f16
// CHECK: vld1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
float16x4_t test_vld1_lane_f16(float16_t const * a, float16x4_t b) {
  return vld1_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vld1_lane_f32
// CHECK: vld1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
float32x2_t test_vld1_lane_f32(float32_t const * a, float32x2_t b) {
  return vld1_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vld1_lane_p8
// CHECK: vld1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly8x8_t test_vld1_lane_p8(poly8_t const * a, poly8x8_t b) {
  return vld1_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vld1_lane_p16
// CHECK: vld1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
poly16x4_t test_vld1_lane_p16(poly16_t const * a, poly16x4_t b) {
  return vld1_lane_p16(a, b, 3);
}


// CHECK-LABEL: test_vld2q_u8
// CHECK: vld2.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint8x16x2_t test_vld2q_u8(uint8_t const * a) {
  return vld2q_u8(a);
}

// CHECK-LABEL: test_vld2q_u16
// CHECK: vld2.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint16x8x2_t test_vld2q_u16(uint16_t const * a) {
  return vld2q_u16(a);
}

// CHECK-LABEL: test_vld2q_u32
// CHECK: vld2.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint32x4x2_t test_vld2q_u32(uint32_t const * a) {
  return vld2q_u32(a);
}

// CHECK-LABEL: test_vld2q_s8
// CHECK: vld2.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int8x16x2_t test_vld2q_s8(int8_t const * a) {
  return vld2q_s8(a);
}

// CHECK-LABEL: test_vld2q_s16
// CHECK: vld2.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int16x8x2_t test_vld2q_s16(int16_t const * a) {
  return vld2q_s16(a);
}

// CHECK-LABEL: test_vld2q_s32
// CHECK: vld2.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int32x4x2_t test_vld2q_s32(int32_t const * a) {
  return vld2q_s32(a);
}

// CHECK-LABEL: test_vld2q_f16
// CHECK: vld2.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float16x8x2_t test_vld2q_f16(float16_t const * a) {
  return vld2q_f16(a);
}

// CHECK-LABEL: test_vld2q_f32
// CHECK: vld2.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float32x4x2_t test_vld2q_f32(float32_t const * a) {
  return vld2q_f32(a);
}

// CHECK-LABEL: test_vld2q_p8
// CHECK: vld2.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly8x16x2_t test_vld2q_p8(poly8_t const * a) {
  return vld2q_p8(a);
}

// CHECK-LABEL: test_vld2q_p16
// CHECK: vld2.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly16x8x2_t test_vld2q_p16(poly16_t const * a) {
  return vld2q_p16(a);
}

// CHECK-LABEL: test_vld2_u8
// CHECK: vld2.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint8x8x2_t test_vld2_u8(uint8_t const * a) {
  return vld2_u8(a);
}

// CHECK-LABEL: test_vld2_u16
// CHECK: vld2.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint16x4x2_t test_vld2_u16(uint16_t const * a) {
  return vld2_u16(a);
}

// CHECK-LABEL: test_vld2_u32
// CHECK: vld2.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint32x2x2_t test_vld2_u32(uint32_t const * a) {
  return vld2_u32(a);
}

// CHECK-LABEL: test_vld2_u64
// CHECK: vld1.64
uint64x1x2_t test_vld2_u64(uint64_t const * a) {
  return vld2_u64(a);
}

// CHECK-LABEL: test_vld2_s8
// CHECK: vld2.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int8x8x2_t test_vld2_s8(int8_t const * a) {
  return vld2_s8(a);
}

// CHECK-LABEL: test_vld2_s16
// CHECK: vld2.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int16x4x2_t test_vld2_s16(int16_t const * a) {
  return vld2_s16(a);
}

// CHECK-LABEL: test_vld2_s32
// CHECK: vld2.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int32x2x2_t test_vld2_s32(int32_t const * a) {
  return vld2_s32(a);
}

// CHECK-LABEL: test_vld2_s64
// CHECK: vld1.64
int64x1x2_t test_vld2_s64(int64_t const * a) {
  return vld2_s64(a);
}

// CHECK-LABEL: test_vld2_f16
// CHECK: vld2.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float16x4x2_t test_vld2_f16(float16_t const * a) {
  return vld2_f16(a);
}

// CHECK-LABEL: test_vld2_f32
// CHECK: vld2.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float32x2x2_t test_vld2_f32(float32_t const * a) {
  return vld2_f32(a);
}

// CHECK-LABEL: test_vld2_p8
// CHECK: vld2.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly8x8x2_t test_vld2_p8(poly8_t const * a) {
  return vld2_p8(a);
}

// CHECK-LABEL: test_vld2_p16
// CHECK: vld2.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly16x4x2_t test_vld2_p16(poly16_t const * a) {
  return vld2_p16(a);
}


// CHECK-LABEL: test_vld2_dup_u8
// CHECK: vld2.8 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint8x8x2_t test_vld2_dup_u8(uint8_t const * a) {
  return vld2_dup_u8(a);
}

// CHECK-LABEL: test_vld2_dup_u16
// CHECK: vld2.16 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint16x4x2_t test_vld2_dup_u16(uint16_t const * a) {
  return vld2_dup_u16(a);
}

// CHECK-LABEL: test_vld2_dup_u32
// CHECK: vld2.32 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint32x2x2_t test_vld2_dup_u32(uint32_t const * a) {
  return vld2_dup_u32(a);
}

// CHECK-LABEL: test_vld2_dup_u64
// CHECK: vld1.64
uint64x1x2_t test_vld2_dup_u64(uint64_t const * a) {
  return vld2_dup_u64(a);
}

// CHECK-LABEL: test_vld2_dup_s8
// CHECK: vld2.8 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int8x8x2_t test_vld2_dup_s8(int8_t const * a) {
  return vld2_dup_s8(a);
}

// CHECK-LABEL: test_vld2_dup_s16
// CHECK: vld2.16 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int16x4x2_t test_vld2_dup_s16(int16_t const * a) {
  return vld2_dup_s16(a);
}

// CHECK-LABEL: test_vld2_dup_s32
// CHECK: vld2.32 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int32x2x2_t test_vld2_dup_s32(int32_t const * a) {
  return vld2_dup_s32(a);
}

// CHECK-LABEL: test_vld2_dup_s64
// CHECK: vld1.64
int64x1x2_t test_vld2_dup_s64(int64_t const * a) {
  return vld2_dup_s64(a);
}

// CHECK-LABEL: test_vld2_dup_f16
// CHECK: vld2.16 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
float16x4x2_t test_vld2_dup_f16(float16_t const * a) {
  return vld2_dup_f16(a);
}

// CHECK-LABEL: test_vld2_dup_f32
// CHECK: vld2.32 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
float32x2x2_t test_vld2_dup_f32(float32_t const * a) {
  return vld2_dup_f32(a);
}

// CHECK-LABEL: test_vld2_dup_p8
// CHECK: vld2.8 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
poly8x8x2_t test_vld2_dup_p8(poly8_t const * a) {
  return vld2_dup_p8(a);
}

// CHECK-LABEL: test_vld2_dup_p16
// CHECK: vld2.16 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
poly16x4x2_t test_vld2_dup_p16(poly16_t const * a) {
  return vld2_dup_p16(a);
}


// CHECK-LABEL: test_vld2q_lane_u16
// CHECK: vld2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint16x8x2_t test_vld2q_lane_u16(uint16_t const * a, uint16x8x2_t b) {
  return vld2q_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vld2q_lane_u32
// CHECK: vld2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint32x4x2_t test_vld2q_lane_u32(uint32_t const * a, uint32x4x2_t b) {
  return vld2q_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vld2q_lane_s16
// CHECK: vld2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int16x8x2_t test_vld2q_lane_s16(int16_t const * a, int16x8x2_t b) {
  return vld2q_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vld2q_lane_s32
// CHECK: vld2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int32x4x2_t test_vld2q_lane_s32(int32_t const * a, int32x4x2_t b) {
  return vld2q_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vld2q_lane_f16
// CHECK: vld2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
float16x8x2_t test_vld2q_lane_f16(float16_t const * a, float16x8x2_t b) {
  return vld2q_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vld2q_lane_f32
// CHECK: vld2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
float32x4x2_t test_vld2q_lane_f32(float32_t const * a, float32x4x2_t b) {
  return vld2q_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vld2q_lane_p16
// CHECK: vld2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly16x8x2_t test_vld2q_lane_p16(poly16_t const * a, poly16x8x2_t b) {
  return vld2q_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vld2_lane_u8
// CHECK: vld2.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint8x8x2_t test_vld2_lane_u8(uint8_t const * a, uint8x8x2_t b) {
  return vld2_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vld2_lane_u16
// CHECK: vld2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint16x4x2_t test_vld2_lane_u16(uint16_t const * a, uint16x4x2_t b) {
  return vld2_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vld2_lane_u32
// CHECK: vld2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint32x2x2_t test_vld2_lane_u32(uint32_t const * a, uint32x2x2_t b) {
  return vld2_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vld2_lane_s8
// CHECK: vld2.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int8x8x2_t test_vld2_lane_s8(int8_t const * a, int8x8x2_t b) {
  return vld2_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vld2_lane_s16
// CHECK: vld2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int16x4x2_t test_vld2_lane_s16(int16_t const * a, int16x4x2_t b) {
  return vld2_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vld2_lane_s32
// CHECK: vld2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int32x2x2_t test_vld2_lane_s32(int32_t const * a, int32x2x2_t b) {
  return vld2_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vld2_lane_f16
// CHECK: vld2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
float16x4x2_t test_vld2_lane_f16(float16_t const * a, float16x4x2_t b) {
  return vld2_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vld2_lane_f32
// CHECK: vld2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
float32x2x2_t test_vld2_lane_f32(float32_t const * a, float32x2x2_t b) {
  return vld2_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vld2_lane_p8
// CHECK: vld2.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly8x8x2_t test_vld2_lane_p8(poly8_t const * a, poly8x8x2_t b) {
  return vld2_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vld2_lane_p16
// CHECK: vld2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly16x4x2_t test_vld2_lane_p16(poly16_t const * a, poly16x4x2_t b) {
  return vld2_lane_p16(a, b, 3);
}


// CHECK-LABEL: test_vld3q_u8
// CHECK: vld3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
uint8x16x3_t test_vld3q_u8(uint8_t const * a) {
  return vld3q_u8(a);
}

// CHECK-LABEL: test_vld3q_u16
// CHECK: vld3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
uint16x8x3_t test_vld3q_u16(uint16_t const * a) {
  return vld3q_u16(a);
}

// CHECK-LABEL: test_vld3q_u32
// CHECK: vld3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
uint32x4x3_t test_vld3q_u32(uint32_t const * a) {
  return vld3q_u32(a);
}

// CHECK-LABEL: test_vld3q_s8
// CHECK: vld3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
int8x16x3_t test_vld3q_s8(int8_t const * a) {
  return vld3q_s8(a);
}

// CHECK-LABEL: test_vld3q_s16
// CHECK: vld3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
int16x8x3_t test_vld3q_s16(int16_t const * a) {
  return vld3q_s16(a);
}

// CHECK-LABEL: test_vld3q_s32
// CHECK: vld3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
int32x4x3_t test_vld3q_s32(int32_t const * a) {
  return vld3q_s32(a);
}

// CHECK-LABEL: test_vld3q_f16
// CHECK: vld3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
float16x8x3_t test_vld3q_f16(float16_t const * a) {
  return vld3q_f16(a);
}

// CHECK-LABEL: test_vld3q_f32
// CHECK: vld3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
float32x4x3_t test_vld3q_f32(float32_t const * a) {
  return vld3q_f32(a);
}

// CHECK-LABEL: test_vld3q_p8
// CHECK: vld3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
poly8x16x3_t test_vld3q_p8(poly8_t const * a) {
  return vld3q_p8(a);
}

// CHECK-LABEL: test_vld3q_p16
// CHECK: vld3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
poly16x8x3_t test_vld3q_p16(poly16_t const * a) {
  return vld3q_p16(a);
}

// CHECK-LABEL: test_vld3_u8
// CHECK: vld3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint8x8x3_t test_vld3_u8(uint8_t const * a) {
  return vld3_u8(a);
}

// CHECK-LABEL: test_vld3_u16
// CHECK: vld3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint16x4x3_t test_vld3_u16(uint16_t const * a) {
  return vld3_u16(a);
}

// CHECK-LABEL: test_vld3_u32
// CHECK: vld3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint32x2x3_t test_vld3_u32(uint32_t const * a) {
  return vld3_u32(a);
}

// CHECK-LABEL: test_vld3_u64
// CHECK: vld1.64
uint64x1x3_t test_vld3_u64(uint64_t const * a) {
  return vld3_u64(a);
}

// CHECK-LABEL: test_vld3_s8
// CHECK: vld3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int8x8x3_t test_vld3_s8(int8_t const * a) {
  return vld3_s8(a);
}

// CHECK-LABEL: test_vld3_s16
// CHECK: vld3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int16x4x3_t test_vld3_s16(int16_t const * a) {
  return vld3_s16(a);
}

// CHECK-LABEL: test_vld3_s32
// CHECK: vld3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int32x2x3_t test_vld3_s32(int32_t const * a) {
  return vld3_s32(a);
}

// CHECK-LABEL: test_vld3_s64
// CHECK: vld1.64
int64x1x3_t test_vld3_s64(int64_t const * a) {
  return vld3_s64(a);
}

// CHECK-LABEL: test_vld3_f16
// CHECK: vld3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float16x4x3_t test_vld3_f16(float16_t const * a) {
  return vld3_f16(a);
}

// CHECK-LABEL: test_vld3_f32
// CHECK: vld3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float32x2x3_t test_vld3_f32(float32_t const * a) {
  return vld3_f32(a);
}

// CHECK-LABEL: test_vld3_p8
// CHECK: vld3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly8x8x3_t test_vld3_p8(poly8_t const * a) {
  return vld3_p8(a);
}

// CHECK-LABEL: test_vld3_p16
// CHECK: vld3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly16x4x3_t test_vld3_p16(poly16_t const * a) {
  return vld3_p16(a);
}


// CHECK-LABEL: test_vld3_dup_u8
// CHECK: vld3.8 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint8x8x3_t test_vld3_dup_u8(uint8_t const * a) {
  return vld3_dup_u8(a);
}

// CHECK-LABEL: test_vld3_dup_u16
// CHECK: vld3.16 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint16x4x3_t test_vld3_dup_u16(uint16_t const * a) {
  return vld3_dup_u16(a);
}

// CHECK-LABEL: test_vld3_dup_u32
// CHECK: vld3.32 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint32x2x3_t test_vld3_dup_u32(uint32_t const * a) {
  return vld3_dup_u32(a);
}

// CHECK-LABEL: test_vld3_dup_u64
// CHECK: vld1.64
uint64x1x3_t test_vld3_dup_u64(uint64_t const * a) {
  return vld3_dup_u64(a);
}

// CHECK-LABEL: test_vld3_dup_s8
// CHECK: vld3.8 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int8x8x3_t test_vld3_dup_s8(int8_t const * a) {
  return vld3_dup_s8(a);
}

// CHECK-LABEL: test_vld3_dup_s16
// CHECK: vld3.16 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int16x4x3_t test_vld3_dup_s16(int16_t const * a) {
  return vld3_dup_s16(a);
}

// CHECK-LABEL: test_vld3_dup_s32
// CHECK: vld3.32 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int32x2x3_t test_vld3_dup_s32(int32_t const * a) {
  return vld3_dup_s32(a);
}

// CHECK-LABEL: test_vld3_dup_s64
// CHECK: vld1.64
int64x1x3_t test_vld3_dup_s64(int64_t const * a) {
  return vld3_dup_s64(a);
}

// CHECK-LABEL: test_vld3_dup_f16
// CHECK: vld3.16 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
float16x4x3_t test_vld3_dup_f16(float16_t const * a) {
  return vld3_dup_f16(a);
}

// CHECK-LABEL: test_vld3_dup_f32
// CHECK: vld3.32 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
float32x2x3_t test_vld3_dup_f32(float32_t const * a) {
  return vld3_dup_f32(a);
}

// CHECK-LABEL: test_vld3_dup_p8
// CHECK: vld3.8 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
poly8x8x3_t test_vld3_dup_p8(poly8_t const * a) {
  return vld3_dup_p8(a);
}

// CHECK-LABEL: test_vld3_dup_p16
// CHECK: vld3.16 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
poly16x4x3_t test_vld3_dup_p16(poly16_t const * a) {
  return vld3_dup_p16(a);
}


// CHECK-LABEL: test_vld3q_lane_u16
// CHECK: vld3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
uint16x8x3_t test_vld3q_lane_u16(uint16_t const * a, uint16x8x3_t b) {
  return vld3q_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vld3q_lane_u32
// CHECK: vld3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
uint32x4x3_t test_vld3q_lane_u32(uint32_t const * a, uint32x4x3_t b) {
  return vld3q_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vld3q_lane_s16
// CHECK: vld3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
int16x8x3_t test_vld3q_lane_s16(int16_t const * a, int16x8x3_t b) {
  return vld3q_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vld3q_lane_s32
// CHECK: vld3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
int32x4x3_t test_vld3q_lane_s32(int32_t const * a, int32x4x3_t b) {
  return vld3q_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vld3q_lane_f16
// CHECK: vld3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
float16x8x3_t test_vld3q_lane_f16(float16_t const * a, float16x8x3_t b) {
  return vld3q_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vld3q_lane_f32
// CHECK: vld3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
float32x4x3_t test_vld3q_lane_f32(float32_t const * a, float32x4x3_t b) {
  return vld3q_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vld3q_lane_p16
// CHECK: vld3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
poly16x8x3_t test_vld3q_lane_p16(poly16_t const * a, poly16x8x3_t b) {
  return vld3q_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vld3_lane_u8
// CHECK: vld3.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint8x8x3_t test_vld3_lane_u8(uint8_t const * a, uint8x8x3_t b) {
  return vld3_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vld3_lane_u16
// CHECK: vld3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint16x4x3_t test_vld3_lane_u16(uint16_t const * a, uint16x4x3_t b) {
  return vld3_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vld3_lane_u32
// CHECK: vld3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint32x2x3_t test_vld3_lane_u32(uint32_t const * a, uint32x2x3_t b) {
  return vld3_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vld3_lane_s8
// CHECK: vld3.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int8x8x3_t test_vld3_lane_s8(int8_t const * a, int8x8x3_t b) {
  return vld3_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vld3_lane_s16
// CHECK: vld3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int16x4x3_t test_vld3_lane_s16(int16_t const * a, int16x4x3_t b) {
  return vld3_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vld3_lane_s32
// CHECK: vld3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int32x2x3_t test_vld3_lane_s32(int32_t const * a, int32x2x3_t b) {
  return vld3_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vld3_lane_f16
// CHECK: vld3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
float16x4x3_t test_vld3_lane_f16(float16_t const * a, float16x4x3_t b) {
  return vld3_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vld3_lane_f32
// CHECK: vld3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
float32x2x3_t test_vld3_lane_f32(float32_t const * a, float32x2x3_t b) {
  return vld3_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vld3_lane_p8
// CHECK: vld3.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly8x8x3_t test_vld3_lane_p8(poly8_t const * a, poly8x8x3_t b) {
  return vld3_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vld3_lane_p16
// CHECK: vld3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly16x4x3_t test_vld3_lane_p16(poly16_t const * a, poly16x4x3_t b) {
  return vld3_lane_p16(a, b, 3);
}


// CHECK-LABEL: test_vld4q_u8
// CHECK: vld4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
uint8x16x4_t test_vld4q_u8(uint8_t const * a) {
  return vld4q_u8(a);
}

// CHECK-LABEL: test_vld4q_u16
// CHECK: vld4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
uint16x8x4_t test_vld4q_u16(uint16_t const * a) {
  return vld4q_u16(a);
}

// CHECK-LABEL: test_vld4q_u32
// CHECK: vld4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
uint32x4x4_t test_vld4q_u32(uint32_t const * a) {
  return vld4q_u32(a);
}

// CHECK-LABEL: test_vld4q_s8
// CHECK: vld4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
int8x16x4_t test_vld4q_s8(int8_t const * a) {
  return vld4q_s8(a);
}

// CHECK-LABEL: test_vld4q_s16
// CHECK: vld4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
int16x8x4_t test_vld4q_s16(int16_t const * a) {
  return vld4q_s16(a);
}

// CHECK-LABEL: test_vld4q_s32
// CHECK: vld4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
int32x4x4_t test_vld4q_s32(int32_t const * a) {
  return vld4q_s32(a);
}

// CHECK-LABEL: test_vld4q_f16
// CHECK: vld4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
float16x8x4_t test_vld4q_f16(float16_t const * a) {
  return vld4q_f16(a);
}

// CHECK-LABEL: test_vld4q_f32
// CHECK: vld4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
float32x4x4_t test_vld4q_f32(float32_t const * a) {
  return vld4q_f32(a);
}

// CHECK-LABEL: test_vld4q_p8
// CHECK: vld4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
poly8x16x4_t test_vld4q_p8(poly8_t const * a) {
  return vld4q_p8(a);
}

// CHECK-LABEL: test_vld4q_p16
// CHECK: vld4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
poly16x8x4_t test_vld4q_p16(poly16_t const * a) {
  return vld4q_p16(a);
}

// CHECK-LABEL: test_vld4_u8
// CHECK: vld4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint8x8x4_t test_vld4_u8(uint8_t const * a) {
  return vld4_u8(a);
}

// CHECK-LABEL: test_vld4_u16
// CHECK: vld4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint16x4x4_t test_vld4_u16(uint16_t const * a) {
  return vld4_u16(a);
}

// CHECK-LABEL: test_vld4_u32
// CHECK: vld4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
uint32x2x4_t test_vld4_u32(uint32_t const * a) {
  return vld4_u32(a);
}

// CHECK-LABEL: test_vld4_u64
// CHECK: vld1.64
uint64x1x4_t test_vld4_u64(uint64_t const * a) {
  return vld4_u64(a);
}

// CHECK-LABEL: test_vld4_s8
// CHECK: vld4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int8x8x4_t test_vld4_s8(int8_t const * a) {
  return vld4_s8(a);
}

// CHECK-LABEL: test_vld4_s16
// CHECK: vld4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int16x4x4_t test_vld4_s16(int16_t const * a) {
  return vld4_s16(a);
}

// CHECK-LABEL: test_vld4_s32
// CHECK: vld4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
int32x2x4_t test_vld4_s32(int32_t const * a) {
  return vld4_s32(a);
}

// CHECK-LABEL: test_vld4_s64
// CHECK: vld1.64
int64x1x4_t test_vld4_s64(int64_t const * a) {
  return vld4_s64(a);
}

// CHECK-LABEL: test_vld4_f16
// CHECK: vld4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float16x4x4_t test_vld4_f16(float16_t const * a) {
  return vld4_f16(a);
}

// CHECK-LABEL: test_vld4_f32
// CHECK: vld4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
float32x2x4_t test_vld4_f32(float32_t const * a) {
  return vld4_f32(a);
}

// CHECK-LABEL: test_vld4_p8
// CHECK: vld4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly8x8x4_t test_vld4_p8(poly8_t const * a) {
  return vld4_p8(a);
}

// CHECK-LABEL: test_vld4_p16
// CHECK: vld4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
poly16x4x4_t test_vld4_p16(poly16_t const * a) {
  return vld4_p16(a);
}


// CHECK-LABEL: test_vld4_dup_u8
// CHECK: vld4.8 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint8x8x4_t test_vld4_dup_u8(uint8_t const * a) {
  return vld4_dup_u8(a);
}

// CHECK-LABEL: test_vld4_dup_u16
// CHECK: vld4.16 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint16x4x4_t test_vld4_dup_u16(uint16_t const * a) {
  return vld4_dup_u16(a);
}

// CHECK-LABEL: test_vld4_dup_u32
// CHECK: vld4.32 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
uint32x2x4_t test_vld4_dup_u32(uint32_t const * a) {
  return vld4_dup_u32(a);
}

// CHECK-LABEL: test_vld4_dup_u64
// CHECK: vld1.64
uint64x1x4_t test_vld4_dup_u64(uint64_t const * a) {
  return vld4_dup_u64(a);
}

// CHECK-LABEL: test_vld4_dup_s8
// CHECK: vld4.8 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int8x8x4_t test_vld4_dup_s8(int8_t const * a) {
  return vld4_dup_s8(a);
}

// CHECK-LABEL: test_vld4_dup_s16
// CHECK: vld4.16 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int16x4x4_t test_vld4_dup_s16(int16_t const * a) {
  return vld4_dup_s16(a);
}

// CHECK-LABEL: test_vld4_dup_s32
// CHECK: vld4.32 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
int32x2x4_t test_vld4_dup_s32(int32_t const * a) {
  return vld4_dup_s32(a);
}

// CHECK-LABEL: test_vld4_dup_s64
// CHECK: vld1.64
int64x1x4_t test_vld4_dup_s64(int64_t const * a) {
  return vld4_dup_s64(a);
}

// CHECK-LABEL: test_vld4_dup_f16
// CHECK: vld4.16 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
float16x4x4_t test_vld4_dup_f16(float16_t const * a) {
  return vld4_dup_f16(a);
}

// CHECK-LABEL: test_vld4_dup_f32
// CHECK: vld4.32 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
float32x2x4_t test_vld4_dup_f32(float32_t const * a) {
  return vld4_dup_f32(a);
}

// CHECK-LABEL: test_vld4_dup_p8
// CHECK: vld4.8 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
poly8x8x4_t test_vld4_dup_p8(poly8_t const * a) {
  return vld4_dup_p8(a);
}

// CHECK-LABEL: test_vld4_dup_p16
// CHECK: vld4.16 {d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[], d{{[0-9]+}}[]}, [r{{[0-9]+}}]
poly16x4x4_t test_vld4_dup_p16(poly16_t const * a) {
  return vld4_dup_p16(a);
}


// CHECK-LABEL: test_vld4q_lane_u16
// CHECK: vld4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
uint16x8x4_t test_vld4q_lane_u16(uint16_t const * a, uint16x8x4_t b) {
  return vld4q_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vld4q_lane_u32
// CHECK: vld4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
uint32x4x4_t test_vld4q_lane_u32(uint32_t const * a, uint32x4x4_t b) {
  return vld4q_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vld4q_lane_s16
// CHECK: vld4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
int16x8x4_t test_vld4q_lane_s16(int16_t const * a, int16x8x4_t b) {
  return vld4q_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vld4q_lane_s32
// CHECK: vld4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
int32x4x4_t test_vld4q_lane_s32(int32_t const * a, int32x4x4_t b) {
  return vld4q_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vld4q_lane_f16
// CHECK: vld4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
float16x8x4_t test_vld4q_lane_f16(float16_t const * a, float16x8x4_t b) {
  return vld4q_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vld4q_lane_f32
// CHECK: vld4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
float32x4x4_t test_vld4q_lane_f32(float32_t const * a, float32x4x4_t b) {
  return vld4q_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vld4q_lane_p16
// CHECK: vld4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
poly16x8x4_t test_vld4q_lane_p16(poly16_t const * a, poly16x8x4_t b) {
  return vld4q_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vld4_lane_u8
// CHECK: vld4.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint8x8x4_t test_vld4_lane_u8(uint8_t const * a, uint8x8x4_t b) {
  return vld4_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vld4_lane_u16
// CHECK: vld4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint16x4x4_t test_vld4_lane_u16(uint16_t const * a, uint16x4x4_t b) {
  return vld4_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vld4_lane_u32
// CHECK: vld4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
uint32x2x4_t test_vld4_lane_u32(uint32_t const * a, uint32x2x4_t b) {
  return vld4_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vld4_lane_s8
// CHECK: vld4.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int8x8x4_t test_vld4_lane_s8(int8_t const * a, int8x8x4_t b) {
  return vld4_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vld4_lane_s16
// CHECK: vld4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int16x4x4_t test_vld4_lane_s16(int16_t const * a, int16x4x4_t b) {
  return vld4_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vld4_lane_s32
// CHECK: vld4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
int32x2x4_t test_vld4_lane_s32(int32_t const * a, int32x2x4_t b) {
  return vld4_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vld4_lane_f16
// CHECK: vld4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
float16x4x4_t test_vld4_lane_f16(float16_t const * a, float16x4x4_t b) {
  return vld4_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vld4_lane_f32
// CHECK: vld4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
float32x2x4_t test_vld4_lane_f32(float32_t const * a, float32x2x4_t b) {
  return vld4_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vld4_lane_p8
// CHECK: vld4.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly8x8x4_t test_vld4_lane_p8(poly8_t const * a, poly8x8x4_t b) {
  return vld4_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vld4_lane_p16
// CHECK: vld4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
poly16x4x4_t test_vld4_lane_p16(poly16_t const * a, poly16x4x4_t b) {
  return vld4_lane_p16(a, b, 3);
}


// CHECK-LABEL: test_vmax_s8
// CHECK: vmax.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vmax_s8(int8x8_t a, int8x8_t b) {
  return vmax_s8(a, b);
}

// CHECK-LABEL: test_vmax_s16
// CHECK: vmax.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmax_s16(int16x4_t a, int16x4_t b) {
  return vmax_s16(a, b);
}

// CHECK-LABEL: test_vmax_s32
// CHECK: vmax.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmax_s32(int32x2_t a, int32x2_t b) {
  return vmax_s32(a, b);
}

// CHECK-LABEL: test_vmax_u8
// CHECK: vmax.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vmax_u8(uint8x8_t a, uint8x8_t b) {
  return vmax_u8(a, b);
}

// CHECK-LABEL: test_vmax_u16
// CHECK: vmax.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmax_u16(uint16x4_t a, uint16x4_t b) {
  return vmax_u16(a, b);
}

// CHECK-LABEL: test_vmax_u32
// CHECK: vmax.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmax_u32(uint32x2_t a, uint32x2_t b) {
  return vmax_u32(a, b);
}

// CHECK-LABEL: test_vmax_f32
// CHECK: vmax.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vmax_f32(float32x2_t a, float32x2_t b) {
  return vmax_f32(a, b);
}

// CHECK-LABEL: test_vmaxq_s8
// CHECK: vmax.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vmaxq_s8(int8x16_t a, int8x16_t b) {
  return vmaxq_s8(a, b);
}

// CHECK-LABEL: test_vmaxq_s16
// CHECK: vmax.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vmaxq_s16(int16x8_t a, int16x8_t b) {
  return vmaxq_s16(a, b);
}

// CHECK-LABEL: test_vmaxq_s32
// CHECK: vmax.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vmaxq_s32(int32x4_t a, int32x4_t b) {
  return vmaxq_s32(a, b);
}

// CHECK-LABEL: test_vmaxq_u8
// CHECK: vmax.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vmaxq_u8(uint8x16_t a, uint8x16_t b) {
  return vmaxq_u8(a, b);
}

// CHECK-LABEL: test_vmaxq_u16
// CHECK: vmax.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vmaxq_u16(uint16x8_t a, uint16x8_t b) {
  return vmaxq_u16(a, b);
}

// CHECK-LABEL: test_vmaxq_u32
// CHECK: vmax.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vmaxq_u32(uint32x4_t a, uint32x4_t b) {
  return vmaxq_u32(a, b);
}

// CHECK-LABEL: test_vmaxq_f32
// CHECK: vmax.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vmaxq_f32(float32x4_t a, float32x4_t b) {
  return vmaxq_f32(a, b);
}


// CHECK-LABEL: test_vmin_s8
// CHECK: vmin.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vmin_s8(int8x8_t a, int8x8_t b) {
  return vmin_s8(a, b);
}

// CHECK-LABEL: test_vmin_s16
// CHECK: vmin.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmin_s16(int16x4_t a, int16x4_t b) {
  return vmin_s16(a, b);
}

// CHECK-LABEL: test_vmin_s32
// CHECK: vmin.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmin_s32(int32x2_t a, int32x2_t b) {
  return vmin_s32(a, b);
}

// CHECK-LABEL: test_vmin_u8
// CHECK: vmin.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vmin_u8(uint8x8_t a, uint8x8_t b) {
  return vmin_u8(a, b);
}

// CHECK-LABEL: test_vmin_u16
// CHECK: vmin.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmin_u16(uint16x4_t a, uint16x4_t b) {
  return vmin_u16(a, b);
}

// CHECK-LABEL: test_vmin_u32
// CHECK: vmin.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmin_u32(uint32x2_t a, uint32x2_t b) {
  return vmin_u32(a, b);
}

// CHECK-LABEL: test_vmin_f32
// CHECK: vmin.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vmin_f32(float32x2_t a, float32x2_t b) {
  return vmin_f32(a, b);
}

// CHECK-LABEL: test_vminq_s8
// CHECK: vmin.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vminq_s8(int8x16_t a, int8x16_t b) {
  return vminq_s8(a, b);
}

// CHECK-LABEL: test_vminq_s16
// CHECK: vmin.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vminq_s16(int16x8_t a, int16x8_t b) {
  return vminq_s16(a, b);
}

// CHECK-LABEL: test_vminq_s32
// CHECK: vmin.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vminq_s32(int32x4_t a, int32x4_t b) {
  return vminq_s32(a, b);
}

// CHECK-LABEL: test_vminq_u8
// CHECK: vmin.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vminq_u8(uint8x16_t a, uint8x16_t b) {
  return vminq_u8(a, b);
}

// CHECK-LABEL: test_vminq_u16
// CHECK: vmin.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vminq_u16(uint16x8_t a, uint16x8_t b) {
  return vminq_u16(a, b);
}

// CHECK-LABEL: test_vminq_u32
// CHECK: vmin.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vminq_u32(uint32x4_t a, uint32x4_t b) {
  return vminq_u32(a, b);
}

// CHECK-LABEL: test_vminq_f32
// CHECK: vmin.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vminq_f32(float32x4_t a, float32x4_t b) {
  return vminq_f32(a, b);
}


// CHECK-LABEL: test_vmla_s8
// CHECK: vmla.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vmla_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
  return vmla_s8(a, b, c);
}

// CHECK-LABEL: test_vmla_s16
// CHECK: vmla.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmla_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
  return vmla_s16(a, b, c);
}

// CHECK-LABEL: test_vmla_s32
// CHECK: vmla.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmla_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
  return vmla_s32(a, b, c);
}

// CHECK-LABEL: test_vmla_f32
// CHECK-SWIFT: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-SWIFT: vadd.f32
// CHECK-A57: vmla.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vmla_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
  return vmla_f32(a, b, c);
}

// CHECK-LABEL: test_vmla_u8
// CHECK: vmla.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vmla_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c) {
  return vmla_u8(a, b, c);
}

// CHECK-LABEL: test_vmla_u16
// CHECK: vmla.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmla_u16(uint16x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmla_u16(a, b, c);
}

// CHECK-LABEL: test_vmla_u32
// CHECK: vmla.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmla_u32(uint32x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmla_u32(a, b, c);
}

// CHECK-LABEL: test_vmlaq_s8
// CHECK: vmla.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vmlaq_s8(int8x16_t a, int8x16_t b, int8x16_t c) {
  return vmlaq_s8(a, b, c);
}

// CHECK-LABEL: test_vmlaq_s16
// CHECK: vmla.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vmlaq_s16(int16x8_t a, int16x8_t b, int16x8_t c) {
  return vmlaq_s16(a, b, c);
}

// CHECK-LABEL: test_vmlaq_s32
// CHECK: vmla.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vmlaq_s32(int32x4_t a, int32x4_t b, int32x4_t c) {
  return vmlaq_s32(a, b, c);
}

// CHECK-LABEL: test_vmlaq_f32
// CHECK-SWIFT: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// CHECK-SWIFT: vadd.f32
// CHECK-A57: vmla.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return vmlaq_f32(a, b, c);
}

// CHECK-LABEL: test_vmlaq_u8
// CHECK: vmla.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vmlaq_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
  return vmlaq_u8(a, b, c);
}

// CHECK-LABEL: test_vmlaq_u16
// CHECK: vmla.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vmlaq_u16(uint16x8_t a, uint16x8_t b, uint16x8_t c) {
  return vmlaq_u16(a, b, c);
}

// CHECK-LABEL: test_vmlaq_u32
// CHECK: vmla.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vmlaq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  return vmlaq_u32(a, b, c);
}


// CHECK-LABEL: test_vmlal_s8
// CHECK: vmlal.s8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vmlal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  return vmlal_s8(a, b, c);
}

// CHECK-LABEL: test_vmlal_s16
// CHECK: vmlal.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vmlal_s16(a, b, c);
}

// CHECK-LABEL: test_vmlal_s32
// CHECK: vmlal.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vmlal_s32(a, b, c);
}

// CHECK-LABEL: test_vmlal_u8
// CHECK: vmlal.u8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vmlal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  return vmlal_u8(a, b, c);
}

// CHECK-LABEL: test_vmlal_u16
// CHECK: vmlal.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vmlal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmlal_u16(a, b, c);
}

// CHECK-LABEL: test_vmlal_u32
// CHECK: vmlal.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vmlal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmlal_u32(a, b, c);
}


// CHECK-LABEL: test_vmlal_lane_s16
// CHECK: vmlal.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vmlal_lane_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vmlal_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vmlal_lane_s32
// CHECK: vmlal.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int64x2_t test_vmlal_lane_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vmlal_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vmlal_lane_u16
// CHECK: vmlal.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x4_t test_vmlal_lane_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmlal_lane_u16(a, b, c, 3);
}

// CHECK-LABEL: test_vmlal_lane_u32
// CHECK: vmlal.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint64x2_t test_vmlal_lane_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmlal_lane_u32(a, b, c, 1);
}


// CHECK-LABEL: test_vmlal_n_s16
// CHECK: vmlal.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vmlal_n_s16(int32x4_t a, int16x4_t b, int16_t c) {
  return vmlal_n_s16(a, b, c);
}

// CHECK-LABEL: test_vmlal_n_s32
// CHECK: vmlal.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vmlal_n_s32(int64x2_t a, int32x2_t b, int32_t c) {
  return vmlal_n_s32(a, b, c);
}

// CHECK-LABEL: test_vmlal_n_u16
// CHECK: vmlal.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vmlal_n_u16(uint32x4_t a, uint16x4_t b, uint16_t c) {
  return vmlal_n_u16(a, b, c);
}

// CHECK-LABEL: test_vmlal_n_u32
// CHECK: vmlal.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vmlal_n_u32(uint64x2_t a, uint32x2_t b, uint32_t c) {
  return vmlal_n_u32(a, b, c);
}


// CHECK-LABEL: test_vmla_lane_s16
// CHECK: vmla.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x4_t test_vmla_lane_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
  return vmla_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vmla_lane_s32
// CHECK: vmla.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x2_t test_vmla_lane_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
  return vmla_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vmla_lane_u16
// CHECK: vmla.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint16x4_t test_vmla_lane_u16(uint16x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmla_lane_u16(a, b, c, 3);
}

// CHECK-LABEL: test_vmla_lane_u32
// CHECK: vmla.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x2_t test_vmla_lane_u32(uint32x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmla_lane_u32(a, b, c, 1);
}

// CHECK-LABEL: test_vmla_lane_f32
// CHECK-SWIFT: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
// CHECK-SWIFT: vadd.f32
// CHECK-A57: vmla.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
float32x2_t test_vmla_lane_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
  return vmla_lane_f32(a, b, c, 1);
}

// CHECK-LABEL: test_vmlaq_lane_s16
// CHECK: vmla.i16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x8_t test_vmlaq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t c) {
  return vmlaq_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vmlaq_lane_s32
// CHECK: vmla.i32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vmlaq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t c) {
  return vmlaq_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vmlaq_lane_u16
// CHECK: vmla.i16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint16x8_t test_vmlaq_lane_u16(uint16x8_t a, uint16x8_t b, uint16x4_t c) {
  return vmlaq_lane_u16(a, b, c, 3);
}

// CHECK-LABEL: test_vmlaq_lane_u32
// CHECK: vmla.i32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x4_t test_vmlaq_lane_u32(uint32x4_t a, uint32x4_t b, uint32x2_t c) {
  return vmlaq_lane_u32(a, b, c, 1);
}

// CHECK-LABEL: test_vmlaq_lane_f32
// CHECK-SWIFT: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
// CHECK-SWIFT: vadd.f32
// CHECK-A57: vmla.f32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
float32x4_t test_vmlaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t c) {
  return vmlaq_lane_f32(a, b, c, 1);
}


// CHECK-LABEL: test_vmla_n_s16
// CHECK: vmla.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmla_n_s16(int16x4_t a, int16x4_t b, int16_t c) {
  return vmla_n_s16(a, b, c);
}

// CHECK-LABEL: test_vmla_n_s32
// CHECK: vmla.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmla_n_s32(int32x2_t a, int32x2_t b, int32_t c) {
  return vmla_n_s32(a, b, c);
}

// CHECK-LABEL: test_vmla_n_u16
// CHECK: vmla.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmla_n_u16(uint16x4_t a, uint16x4_t b, uint16_t c) {
  return vmla_n_u16(a, b, c);
}

// CHECK-LABEL: test_vmla_n_u32
// CHECK: vmla.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmla_n_u32(uint32x2_t a, uint32x2_t b, uint32_t c) {
  return vmla_n_u32(a, b, c);
}

// CHECK-LABEL: test_vmla_n_f32
// CHECK-SWIFT: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-SWIFT: vadd.f32
// CHECK-A57: vmla.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vmla_n_f32(float32x2_t a, float32x2_t b, float32_t c) {
  return vmla_n_f32(a, b, c);
}

// CHECK-LABEL: test_vmlaq_n_s16
// CHECK: vmla.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vmlaq_n_s16(int16x8_t a, int16x8_t b, int16_t c) {
  return vmlaq_n_s16(a, b, c);
}

// CHECK-LABEL: test_vmlaq_n_s32
// CHECK: vmla.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vmlaq_n_s32(int32x4_t a, int32x4_t b, int32_t c) {
  return vmlaq_n_s32(a, b, c);
}

// CHECK-LABEL: test_vmlaq_n_u16
// CHECK: vmla.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vmlaq_n_u16(uint16x8_t a, uint16x8_t b, uint16_t c) {
  return vmlaq_n_u16(a, b, c);
}

// CHECK-LABEL: test_vmlaq_n_u32
// CHECK: vmla.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vmlaq_n_u32(uint32x4_t a, uint32x4_t b, uint32_t c) {
  return vmlaq_n_u32(a, b, c);
}

// CHECK-LABEL: test_vmlaq_n_f32
// CHECK-SWIFT: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[0]
// CHECK-SWIFT: vadd.f32
// CHECK-A57: vld1.32 {d{{[0-9]+}}[], d{{[0-9]+}}[]}, 
// CHECK-A57: vmla.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vmlaq_n_f32(float32x4_t a, float32x4_t b, float32_t c) {
  return vmlaq_n_f32(a, b, c);
}


// CHECK-LABEL: test_vmls_s8
// CHECK: vmls.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vmls_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
  return vmls_s8(a, b, c);
}

// CHECK-LABEL: test_vmls_s16
// CHECK: vmls.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmls_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
  return vmls_s16(a, b, c);
}

// CHECK-LABEL: test_vmls_s32
// CHECK: vmls.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmls_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
  return vmls_s32(a, b, c);
}

// CHECK-LABEL: test_vmls_f32
// CHECK-SWIFT: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-SWIFT: vsub.f32
// CHECK-A57: vmls.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vmls_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
  return vmls_f32(a, b, c);
}

// CHECK-LABEL: test_vmls_u8
// CHECK: vmls.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vmls_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c) {
  return vmls_u8(a, b, c);
}

// CHECK-LABEL: test_vmls_u16
// CHECK: vmls.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmls_u16(uint16x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmls_u16(a, b, c);
}

// CHECK-LABEL: test_vmls_u32
// CHECK: vmls.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmls_u32(uint32x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmls_u32(a, b, c);
}

// CHECK-LABEL: test_vmlsq_s8
// CHECK: vmls.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vmlsq_s8(int8x16_t a, int8x16_t b, int8x16_t c) {
  return vmlsq_s8(a, b, c);
}

// CHECK-LABEL: test_vmlsq_s16
// CHECK: vmls.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vmlsq_s16(int16x8_t a, int16x8_t b, int16x8_t c) {
  return vmlsq_s16(a, b, c);
}

// CHECK-LABEL: test_vmlsq_s32
// CHECK: vmls.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vmlsq_s32(int32x4_t a, int32x4_t b, int32x4_t c) {
  return vmlsq_s32(a, b, c);
}

// CHECK-LABEL: test_vmlsq_f32
// CHECK-SWIFT: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// CHECK-SWIFT: vsub.f32
// CHECK-A57: vmls.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vmlsq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return vmlsq_f32(a, b, c);
}

// CHECK-LABEL: test_vmlsq_u8
// CHECK: vmls.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vmlsq_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
  return vmlsq_u8(a, b, c);
}

// CHECK-LABEL: test_vmlsq_u16
// CHECK: vmls.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vmlsq_u16(uint16x8_t a, uint16x8_t b, uint16x8_t c) {
  return vmlsq_u16(a, b, c);
}

// CHECK-LABEL: test_vmlsq_u32
// CHECK: vmls.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vmlsq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  return vmlsq_u32(a, b, c);
}


// CHECK-LABEL: test_vmlsl_s8
// CHECK: vmlsl.s8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vmlsl_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  return vmlsl_s8(a, b, c);
}

// CHECK-LABEL: test_vmlsl_s16
// CHECK: vmlsl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vmlsl_s16(a, b, c);
}

// CHECK-LABEL: test_vmlsl_s32
// CHECK: vmlsl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vmlsl_s32(a, b, c);
}

// CHECK-LABEL: test_vmlsl_u8
// CHECK: vmlsl.u8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vmlsl_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  return vmlsl_u8(a, b, c);
}

// CHECK-LABEL: test_vmlsl_u16
// CHECK: vmlsl.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vmlsl_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmlsl_u16(a, b, c);
}

// CHECK-LABEL: test_vmlsl_u32
// CHECK: vmlsl.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vmlsl_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmlsl_u32(a, b, c);
}


// CHECK-LABEL: test_vmlsl_lane_s16
// CHECK: vmlsl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vmlsl_lane_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vmlsl_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vmlsl_lane_s32
// CHECK: vmlsl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int64x2_t test_vmlsl_lane_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vmlsl_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vmlsl_lane_u16
// CHECK: vmlsl.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x4_t test_vmlsl_lane_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmlsl_lane_u16(a, b, c, 3);
}

// CHECK-LABEL: test_vmlsl_lane_u32
// CHECK: vmlsl.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint64x2_t test_vmlsl_lane_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmlsl_lane_u32(a, b, c, 1);
}


// CHECK-LABEL: test_vmlsl_n_s16
// CHECK: vmlsl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vmlsl_n_s16(int32x4_t a, int16x4_t b, int16_t c) {
  return vmlsl_n_s16(a, b, c);
}

// CHECK-LABEL: test_vmlsl_n_s32
// CHECK: vmlsl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vmlsl_n_s32(int64x2_t a, int32x2_t b, int32_t c) {
  return vmlsl_n_s32(a, b, c);
}

// CHECK-LABEL: test_vmlsl_n_u16
// CHECK: vmlsl.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vmlsl_n_u16(uint32x4_t a, uint16x4_t b, uint16_t c) {
  return vmlsl_n_u16(a, b, c);
}

// CHECK-LABEL: test_vmlsl_n_u32
// CHECK: vmlsl.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vmlsl_n_u32(uint64x2_t a, uint32x2_t b, uint32_t c) {
  return vmlsl_n_u32(a, b, c);
}


// CHECK-LABEL: test_vmls_lane_s16
// CHECK: vmls.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x4_t test_vmls_lane_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
  return vmls_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vmls_lane_s32
// CHECK: vmls.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x2_t test_vmls_lane_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
  return vmls_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vmls_lane_u16
// CHECK: vmls.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint16x4_t test_vmls_lane_u16(uint16x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmls_lane_u16(a, b, c, 3);
}

// CHECK-LABEL: test_vmls_lane_u32
// CHECK: vmls.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x2_t test_vmls_lane_u32(uint32x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmls_lane_u32(a, b, c, 1);
}

// CHECK-LABEL: test_vmls_lane_f32
// CHECK-SWIFT: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
// CHECK-SWIFT: vsub.f32
// CHECK-A57: vmls.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
float32x2_t test_vmls_lane_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
  return vmls_lane_f32(a, b, c, 1);
}

// CHECK-LABEL: test_vmlsq_lane_s16
// CHECK: vmls.i16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x8_t test_vmlsq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t c) {
  return vmlsq_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vmlsq_lane_s32
// CHECK: vmls.i32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vmlsq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t c) {
  return vmlsq_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vmlsq_lane_u16
// CHECK: vmls.i16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint16x8_t test_vmlsq_lane_u16(uint16x8_t a, uint16x8_t b, uint16x4_t c) {
  return vmlsq_lane_u16(a, b, c, 3);
}

// CHECK-LABEL: test_vmlsq_lane_u32
// CHECK: vmls.i32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x4_t test_vmlsq_lane_u32(uint32x4_t a, uint32x4_t b, uint32x2_t c) {
  return vmlsq_lane_u32(a, b, c, 1);
}

// CHECK-LABEL: test_vmlsq_lane_f32
// CHECK-SWIFT: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
// CHECK-SWIFT: vsub.f32
// CHECK-A57: vmls.f32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
float32x4_t test_vmlsq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t c) {
  return vmlsq_lane_f32(a, b, c, 1);
}


// CHECK-LABEL: test_vmls_n_s16
// CHECK: vmls.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmls_n_s16(int16x4_t a, int16x4_t b, int16_t c) {
  return vmls_n_s16(a, b, c);
}

// CHECK-LABEL: test_vmls_n_s32
// CHECK: vmls.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmls_n_s32(int32x2_t a, int32x2_t b, int32_t c) {
  return vmls_n_s32(a, b, c);
}

// CHECK-LABEL: test_vmls_n_u16
// CHECK: vmls.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmls_n_u16(uint16x4_t a, uint16x4_t b, uint16_t c) {
  return vmls_n_u16(a, b, c);
}

// CHECK-LABEL: test_vmls_n_u32
// CHECK: vmls.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmls_n_u32(uint32x2_t a, uint32x2_t b, uint32_t c) {
  return vmls_n_u32(a, b, c);
}

// CHECK-LABEL: test_vmls_n_f32
// CHECK-SWIFT: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-SWIFT: vsub.f32
// CHECK-A57: vmls.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vmls_n_f32(float32x2_t a, float32x2_t b, float32_t c) {
  return vmls_n_f32(a, b, c);
}

// CHECK-LABEL: test_vmlsq_n_s16
// CHECK: vmls.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vmlsq_n_s16(int16x8_t a, int16x8_t b, int16_t c) {
  return vmlsq_n_s16(a, b, c);
}

// CHECK-LABEL: test_vmlsq_n_s32
// CHECK: vmls.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vmlsq_n_s32(int32x4_t a, int32x4_t b, int32_t c) {
  return vmlsq_n_s32(a, b, c);
}

// CHECK-LABEL: test_vmlsq_n_u16
// CHECK: vmls.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vmlsq_n_u16(uint16x8_t a, uint16x8_t b, uint16_t c) {
  return vmlsq_n_u16(a, b, c);
}

// CHECK-LABEL: test_vmlsq_n_u32
// CHECK: vmls.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vmlsq_n_u32(uint32x4_t a, uint32x4_t b, uint32_t c) {
  return vmlsq_n_u32(a, b, c);
}

// CHECK-LABEL: test_vmlsq_n_f32
// CHECK-SWIFT: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[0]
// CHECK-SWIFT: vsub.f32
// CHECK-A57: vmls.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vmlsq_n_f32(float32x4_t a, float32x4_t b, float32_t c) {
  return vmlsq_n_f32(a, b, c);
}


// CHECK-LABEL: test_vmovl_s8
// CHECK: vmovl.s8 q{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vmovl_s8(int8x8_t a) {
  return vmovl_s8(a);
}

// CHECK-LABEL: test_vmovl_s16
// CHECK: vmovl.s16 q{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vmovl_s16(int16x4_t a) {
  return vmovl_s16(a);
}

// CHECK-LABEL: test_vmovl_s32
// CHECK: vmovl.s32 q{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vmovl_s32(int32x2_t a) {
  return vmovl_s32(a);
}

// CHECK-LABEL: test_vmovl_u8
// CHECK: vmovl.u8 q{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vmovl_u8(uint8x8_t a) {
  return vmovl_u8(a);
}

// CHECK-LABEL: test_vmovl_u16
// CHECK: vmovl.u16 q{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vmovl_u16(uint16x4_t a) {
  return vmovl_u16(a);
}

// CHECK-LABEL: test_vmovl_u32
// CHECK: vmovl.u32 q{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vmovl_u32(uint32x2_t a) {
  return vmovl_u32(a);
}


// CHECK-LABEL: test_vmovn_s16
// CHECK: vmovn.i16 d{{[0-9]+}}, q{{[0-9]+}}
int8x8_t test_vmovn_s16(int16x8_t a) {
  return vmovn_s16(a);
}

// CHECK-LABEL: test_vmovn_s32
// CHECK: vmovn.i32 d{{[0-9]+}}, q{{[0-9]+}}
int16x4_t test_vmovn_s32(int32x4_t a) {
  return vmovn_s32(a);
}

// CHECK-LABEL: test_vmovn_s64
// CHECK: vmovn.i64 d{{[0-9]+}}, q{{[0-9]+}}
int32x2_t test_vmovn_s64(int64x2_t a) {
  return vmovn_s64(a);
}

// CHECK-LABEL: test_vmovn_u16
// CHECK: vmovn.i16 d{{[0-9]+}}, q{{[0-9]+}}
uint8x8_t test_vmovn_u16(uint16x8_t a) {
  return vmovn_u16(a);
}

// CHECK-LABEL: test_vmovn_u32
// CHECK: vmovn.i32 d{{[0-9]+}}, q{{[0-9]+}}
uint16x4_t test_vmovn_u32(uint32x4_t a) {
  return vmovn_u32(a);
}

// CHECK-LABEL: test_vmovn_u64
// CHECK: vmovn.i64 d{{[0-9]+}}, q{{[0-9]+}}
uint32x2_t test_vmovn_u64(uint64x2_t a) {
  return vmovn_u64(a);
}


// CHECK-LABEL: test_vmov_n_u8
// CHECK: vmov {{r[0-9]+}}
uint8x8_t test_vmov_n_u8(uint8_t a) {
  return vmov_n_u8(a);
}

// CHECK-LABEL: test_vmov_n_u16
// CHECK: vmov {{r[0-9]+}}
uint16x4_t test_vmov_n_u16(uint16_t a) {
  return vmov_n_u16(a);
}

// CHECK-LABEL: test_vmov_n_u32
// CHECK: mov {{r[0-9]+}}
uint32x2_t test_vmov_n_u32(uint32_t a) {
  return vmov_n_u32(a);
}

// CHECK-LABEL: test_vmov_n_s8
// CHECK: vmov {{r[0-9]+}}
int8x8_t test_vmov_n_s8(int8_t a) {
  return vmov_n_s8(a);
}

// CHECK-LABEL: test_vmov_n_s16
// CHECK: vmov {{r[0-9]+}}
int16x4_t test_vmov_n_s16(int16_t a) {
  return vmov_n_s16(a);
}

// CHECK-LABEL: test_vmov_n_s32
// CHECK: mov {{r[0-9]+}}
int32x2_t test_vmov_n_s32(int32_t a) {
  return vmov_n_s32(a);
}

// CHECK-LABEL: test_vmov_n_p8
// CHECK: vmov {{r[0-9]+}}
poly8x8_t test_vmov_n_p8(poly8_t a) {
  return vmov_n_p8(a);
}

// CHECK-LABEL: test_vmov_n_p16
// CHECK: vmov {{r[0-9]+}}
poly16x4_t test_vmov_n_p16(poly16_t a) {
  return vmov_n_p16(a);
}

// CHECK-LABEL: test_vmov_n_f16
// CHECK: vld1.16 {{{d[0-9]+\[\]}}}
float16x4_t test_vmov_n_f16(float16_t *a) {
  return vmov_n_f16(*a);
}

// CHECK-LABEL: test_vmov_n_f32
// CHECK: mov {{r[0-9]+}}
float32x2_t test_vmov_n_f32(float32_t a) {
  return vmov_n_f32(a);
}

// CHECK-LABEL: test_vmovq_n_u8
// CHECK: vmov {{r[0-9]+}}
uint8x16_t test_vmovq_n_u8(uint8_t a) {
  return vmovq_n_u8(a);
}

// CHECK-LABEL: test_vmovq_n_u16
// CHECK: vmov {{r[0-9]+}}
uint16x8_t test_vmovq_n_u16(uint16_t a) {
  return vmovq_n_u16(a);
}

// CHECK-LABEL: test_vmovq_n_u32
// CHECK: vmov {{r[0-9]+}}
uint32x4_t test_vmovq_n_u32(uint32_t a) {
  return vmovq_n_u32(a);
}

// CHECK-LABEL: test_vmovq_n_s8
// CHECK: vmov {{r[0-9]+}}
int8x16_t test_vmovq_n_s8(int8_t a) {
  return vmovq_n_s8(a);
}

// CHECK-LABEL: test_vmovq_n_s16
// CHECK: vmov {{r[0-9]+}}
int16x8_t test_vmovq_n_s16(int16_t a) {
  return vmovq_n_s16(a);
}

// CHECK-LABEL: test_vmovq_n_s32
// CHECK: vmov {{r[0-9]+}}
int32x4_t test_vmovq_n_s32(int32_t a) {
  return vmovq_n_s32(a);
}

// CHECK-LABEL: test_vmovq_n_p8
// CHECK: vmov {{r[0-9]+}}
poly8x16_t test_vmovq_n_p8(poly8_t a) {
  return vmovq_n_p8(a);
}

// CHECK-LABEL: test_vmovq_n_p16
// CHECK: vmov {{r[0-9]+}}
poly16x8_t test_vmovq_n_p16(poly16_t a) {
  return vmovq_n_p16(a);
}

// CHECK-LABEL: test_vmovq_n_f16
// CHECK: vld1.16 {{{d[0-9]+\[\], d[0-9]+\[\]}}}
float16x8_t test_vmovq_n_f16(float16_t *a) {
  return vmovq_n_f16(*a);
}

// CHECK-LABEL: test_vmovq_n_f32
// CHECK: vmov {{r[0-9]+}}
float32x4_t test_vmovq_n_f32(float32_t a) {
  return vmovq_n_f32(a);
}

// CHECK-LABEL: test_vmov_n_s64
// CHECK: vmov.32 [[REG:d[0-9]+]][0], r0
// CHECK: vmov.32 [[REG]][1], r1
int64x1_t test_vmov_n_s64(int64_t a) {
  int64x1_t tmp = vmov_n_s64(a);
  return vadd_s64(tmp, tmp);
}

// CHECK-LABEL: test_vmov_n_u64
// CHECK: vmov.32 [[REG:d[0-9]+]][0], r0
// CHECK: vmov.32 [[REG]][1], r1
uint64x1_t test_vmov_n_u64(uint64_t a) {
  uint64x1_t tmp = vmov_n_u64(a);
  return vadd_u64(tmp, tmp);
}

// CHECK-LABEL: test_vmovq_n_s64
// CHECK: vmov {{r[0-9]+}}
int64x2_t test_vmovq_n_s64(int64_t a) {
  return vmovq_n_s64(a);
}

// CHECK-LABEL: test_vmovq_n_u64
// CHECK: vmov {{r[0-9]+}}
uint64x2_t test_vmovq_n_u64(uint64_t a) {
  return vmovq_n_u64(a);
}


// CHECK-LABEL: test_vmul_s8
// CHECK: vmul.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vmul_s8(int8x8_t a, int8x8_t b) {
  return vmul_s8(a, b);
}

// CHECK-LABEL: test_vmul_s16
// CHECK: vmul.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmul_s16(int16x4_t a, int16x4_t b) {
  return vmul_s16(a, b);
}

// CHECK-LABEL: test_vmul_s32
// CHECK: vmul.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmul_s32(int32x2_t a, int32x2_t b) {
  return vmul_s32(a, b);
}

// CHECK-LABEL: test_vmul_f32
// CHECK: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vmul_f32(float32x2_t a, float32x2_t b) {
  return vmul_f32(a, b);
}

// CHECK-LABEL: test_vmul_u8
// CHECK: vmul.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vmul_u8(uint8x8_t a, uint8x8_t b) {
  return vmul_u8(a, b);
}

// CHECK-LABEL: test_vmul_u16
// CHECK: vmul.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmul_u16(uint16x4_t a, uint16x4_t b) {
  return vmul_u16(a, b);
}

// CHECK-LABEL: test_vmul_u32
// CHECK: vmul.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmul_u32(uint32x2_t a, uint32x2_t b) {
  return vmul_u32(a, b);
}

// CHECK-LABEL: test_vmulq_s8
// CHECK: vmul.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vmulq_s8(int8x16_t a, int8x16_t b) {
  return vmulq_s8(a, b);
}

// CHECK-LABEL: test_vmulq_s16
// CHECK: vmul.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vmulq_s16(int16x8_t a, int16x8_t b) {
  return vmulq_s16(a, b);
}

// CHECK-LABEL: test_vmulq_s32
// CHECK: vmul.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vmulq_s32(int32x4_t a, int32x4_t b) {
  return vmulq_s32(a, b);
}

// CHECK-LABEL: test_vmulq_f32
// CHECK: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vmulq_f32(float32x4_t a, float32x4_t b) {
  return vmulq_f32(a, b);
}

// CHECK-LABEL: test_vmulq_u8
// CHECK: vmul.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vmulq_u8(uint8x16_t a, uint8x16_t b) {
  return vmulq_u8(a, b);
}

// CHECK-LABEL: test_vmulq_u16
// CHECK: vmul.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vmulq_u16(uint16x8_t a, uint16x8_t b) {
  return vmulq_u16(a, b);
}

// CHECK-LABEL: test_vmulq_u32
// CHECK: vmul.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vmulq_u32(uint32x4_t a, uint32x4_t b) {
  return vmulq_u32(a, b);
}


// CHECK-LABEL: test_vmull_s8
// CHECK: vmull.s8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vmull_s8(int8x8_t a, int8x8_t b) {
  return vmull_s8(a, b);
}

// CHECK-LABEL: test_vmull_s16
// CHECK: vmull.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vmull_s16(int16x4_t a, int16x4_t b) {
  return vmull_s16(a, b);
}

// CHECK-LABEL: test_vmull_s32
// CHECK: vmull.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vmull_s32(int32x2_t a, int32x2_t b) {
  return vmull_s32(a, b);
}

// CHECK-LABEL: test_vmull_u8
// CHECK: vmull.u8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vmull_u8(uint8x8_t a, uint8x8_t b) {
  return vmull_u8(a, b);
}

// CHECK-LABEL: test_vmull_u16
// CHECK: vmull.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vmull_u16(uint16x4_t a, uint16x4_t b) {
  return vmull_u16(a, b);
}

// CHECK-LABEL: test_vmull_u32
// CHECK: vmull.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vmull_u32(uint32x2_t a, uint32x2_t b) {
  return vmull_u32(a, b);
}

// CHECK-LABEL: test_vmull_p8
// CHECK: vmull.p8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
poly16x8_t test_vmull_p8(poly8x8_t a, poly8x8_t b) {
  return vmull_p8(a, b);
}


// CHECK-LABEL: test_vmull_lane_s16
// CHECK: vmull.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vmull_lane_s16(int16x4_t a, int16x4_t b) {
  return vmull_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vmull_lane_s32
// CHECK: vmull.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int64x2_t test_vmull_lane_s32(int32x2_t a, int32x2_t b) {
  return vmull_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vmull_lane_u16
// CHECK: vmull.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x4_t test_vmull_lane_u16(uint16x4_t a, uint16x4_t b) {
  return vmull_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vmull_lane_u32
// CHECK: vmull.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint64x2_t test_vmull_lane_u32(uint32x2_t a, uint32x2_t b) {
  return vmull_lane_u32(a, b, 1);
}


// CHECK-LABEL: test_vmull_n_s16
// CHECK: vmull.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vmull_n_s16(int16x4_t a, int16_t b) {
  return vmull_n_s16(a, b);
}

// CHECK-LABEL: test_vmull_n_s32
// CHECK: vmull.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vmull_n_s32(int32x2_t a, int32_t b) {
  return vmull_n_s32(a, b);
}

// CHECK-LABEL: test_vmull_n_u16
// CHECK: vmull.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vmull_n_u16(uint16x4_t a, uint16_t b) {
  return vmull_n_u16(a, b);
}

// CHECK-LABEL: test_vmull_n_u32
// CHECK: vmull.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vmull_n_u32(uint32x2_t a, uint32_t b) {
  return vmull_n_u32(a, b);
}


// CHECK-LABEL: test_vmul_p8
// CHECK: vmul.p8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
poly8x8_t test_vmul_p8(poly8x8_t a, poly8x8_t b) {
  return vmul_p8(a, b);
}

// CHECK-LABEL: test_vmulq_p8
// CHECK: vmul.p8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
poly8x16_t test_vmulq_p8(poly8x16_t a, poly8x16_t b) {
  return vmulq_p8(a, b);
}


// CHECK-LABEL: test_vmul_lane_s16
// CHECK: vmul.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x4_t test_vmul_lane_s16(int16x4_t a, int16x4_t b) {
  return vmul_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vmul_lane_s32
// CHECK: vmul.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x2_t test_vmul_lane_s32(int32x2_t a, int32x2_t b) {
  return vmul_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vmul_lane_f32
// CHECK: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
float32x2_t test_vmul_lane_f32(float32x2_t a, float32x2_t b) {
  return vmul_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vmul_lane_u16
// CHECK: vmul.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint16x4_t test_vmul_lane_u16(uint16x4_t a, uint16x4_t b) {
  return vmul_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vmul_lane_u32
// CHECK: vmul.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x2_t test_vmul_lane_u32(uint32x2_t a, uint32x2_t b) {
  return vmul_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vmulq_lane_s16
// CHECK: vmul.i16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x8_t test_vmulq_lane_s16(int16x8_t a, int16x4_t b) {
  return vmulq_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vmulq_lane_s32
// CHECK: vmul.i32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vmulq_lane_s32(int32x4_t a, int32x2_t b) {
  return vmulq_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vmulq_lane_f32
// CHECK: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
float32x4_t test_vmulq_lane_f32(float32x4_t a, float32x2_t b) {
  return vmulq_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vmulq_lane_u16
// CHECK: vmul.i16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint16x8_t test_vmulq_lane_u16(uint16x8_t a, uint16x4_t b) {
  return vmulq_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vmulq_lane_u32
// CHECK: vmul.i32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
uint32x4_t test_vmulq_lane_u32(uint32x4_t a, uint32x2_t b) {
  return vmulq_lane_u32(a, b, 1);
}


// CHECK-LABEL: test_vmul_n_s16
// CHECK: vmul.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmul_n_s16(int16x4_t a, int16_t b) {
  return vmul_n_s16(a, b);
}

// CHECK-LABEL: test_vmul_n_s32
// CHECK: vmul.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmul_n_s32(int32x2_t a, int32_t b) {
  return vmul_n_s32(a, b);
}

// CHECK-LABEL: test_vmul_n_f32
// CHECK: vmul.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vmul_n_f32(float32x2_t a, float32_t b) {
  return vmul_n_f32(a, b);
}

// CHECK-LABEL: test_vmul_n_u16
// CHECK: vmul.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmul_n_u16(uint16x4_t a, uint16_t b) {
  return vmul_n_u16(a, b);
}

// CHECK-LABEL: test_vmul_n_u32
// CHECK: vmul.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmul_n_u32(uint32x2_t a, uint32_t b) {
  return vmul_n_u32(a, b);
}

// CHECK-LABEL: test_vmulq_n_s16
// CHECK: vmul.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vmulq_n_s16(int16x8_t a, int16_t b) {
  return vmulq_n_s16(a, b);
}

// CHECK-LABEL: test_vmulq_n_s32
// CHECK: vmul.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vmulq_n_s32(int32x4_t a, int32_t b) {
  return vmulq_n_s32(a, b);
}

// CHECK-LABEL: test_vmulq_n_f32
// CHECK: vmul.f32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[0]
float32x4_t test_vmulq_n_f32(float32x4_t a, float32_t b) {
  return vmulq_n_f32(a, b);
}

// CHECK-LABEL: test_vmulq_n_u16
// CHECK: vmul.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vmulq_n_u16(uint16x8_t a, uint16_t b) {
  return vmulq_n_u16(a, b);
}

// CHECK-LABEL: test_vmulq_n_u32
// CHECK: vmul.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vmulq_n_u32(uint32x4_t a, uint32_t b) {
  return vmulq_n_u32(a, b);
}


// CHECK-LABEL: test_vmvn_s8
// CHECK: vmvn d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vmvn_s8(int8x8_t a) {
  return vmvn_s8(a);
}

// CHECK-LABEL: test_vmvn_s16
// CHECK: vmvn d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vmvn_s16(int16x4_t a) {
  return vmvn_s16(a);
}

// CHECK-LABEL: test_vmvn_s32
// CHECK: vmvn d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vmvn_s32(int32x2_t a) {
  return vmvn_s32(a);
}

// CHECK-LABEL: test_vmvn_u8
// CHECK: vmvn d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vmvn_u8(uint8x8_t a) {
  return vmvn_u8(a);
}

// CHECK-LABEL: test_vmvn_u16
// CHECK: vmvn d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vmvn_u16(uint16x4_t a) {
  return vmvn_u16(a);
}

// CHECK-LABEL: test_vmvn_u32
// CHECK: vmvn d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vmvn_u32(uint32x2_t a) {
  return vmvn_u32(a);
}

// CHECK-LABEL: test_vmvn_p8
// CHECK: vmvn d{{[0-9]+}}, d{{[0-9]+}}
poly8x8_t test_vmvn_p8(poly8x8_t a) {
  return vmvn_p8(a);
}

// CHECK-LABEL: test_vmvnq_s8
// CHECK: vmvn q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vmvnq_s8(int8x16_t a) {
  return vmvnq_s8(a);
}

// CHECK-LABEL: test_vmvnq_s16
// CHECK: vmvn q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vmvnq_s16(int16x8_t a) {
  return vmvnq_s16(a);
}

// CHECK-LABEL: test_vmvnq_s32
// CHECK: vmvn q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vmvnq_s32(int32x4_t a) {
  return vmvnq_s32(a);
}

// CHECK-LABEL: test_vmvnq_u8
// CHECK: vmvn q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vmvnq_u8(uint8x16_t a) {
  return vmvnq_u8(a);
}

// CHECK-LABEL: test_vmvnq_u16
// CHECK: vmvn q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vmvnq_u16(uint16x8_t a) {
  return vmvnq_u16(a);
}

// CHECK-LABEL: test_vmvnq_u32
// CHECK: vmvn q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vmvnq_u32(uint32x4_t a) {
  return vmvnq_u32(a);
}

// CHECK-LABEL: test_vmvnq_p8
// CHECK: vmvn q{{[0-9]+}}, q{{[0-9]+}}
poly8x16_t test_vmvnq_p8(poly8x16_t a) {
  return vmvnq_p8(a);
}


// CHECK-LABEL: test_vneg_s8
// CHECK: vneg.s8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vneg_s8(int8x8_t a) {
  return vneg_s8(a);
}

// CHECK-LABEL: test_vneg_s16
// CHECK: vneg.s16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vneg_s16(int16x4_t a) {
  return vneg_s16(a);
}

// CHECK-LABEL: test_vneg_s32
// CHECK: vneg.s32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vneg_s32(int32x2_t a) {
  return vneg_s32(a);
}

// CHECK-LABEL: test_vneg_f32
// CHECK: vneg.f32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vneg_f32(float32x2_t a) {
  return vneg_f32(a);
}

// CHECK-LABEL: test_vnegq_s8
// CHECK: vneg.s8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vnegq_s8(int8x16_t a) {
  return vnegq_s8(a);
}

// CHECK-LABEL: test_vnegq_s16
// CHECK: vneg.s16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vnegq_s16(int16x8_t a) {
  return vnegq_s16(a);
}

// CHECK-LABEL: test_vnegq_s32
// CHECK: vneg.s32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vnegq_s32(int32x4_t a) {
  return vnegq_s32(a);
}

// CHECK-LABEL: test_vnegq_f32
// CHECK: vneg.f32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vnegq_f32(float32x4_t a) {
  return vnegq_f32(a);
}


// CHECK-LABEL: test_vorn_s8
// CHECK: vorn d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vorn_s8(int8x8_t a, int8x8_t b) {
  return vorn_s8(a, b);
}

// CHECK-LABEL: test_vorn_s16
// CHECK: vorn d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vorn_s16(int16x4_t a, int16x4_t b) {
  return vorn_s16(a, b);
}

// CHECK-LABEL: test_vorn_s32
// CHECK: vorn d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vorn_s32(int32x2_t a, int32x2_t b) {
  return vorn_s32(a, b);
}

// CHECK-LABEL: test_vorn_s64
// CHECK: vorn d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vorn_s64(int64x1_t a, int64x1_t b) {
  return vorn_s64(a, b);
}

// CHECK-LABEL: test_vorn_u8
// CHECK: vorn d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vorn_u8(uint8x8_t a, uint8x8_t b) {
  return vorn_u8(a, b);
}

// CHECK-LABEL: test_vorn_u16
// CHECK: vorn d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vorn_u16(uint16x4_t a, uint16x4_t b) {
  return vorn_u16(a, b);
}

// CHECK-LABEL: test_vorn_u32
// CHECK: vorn d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vorn_u32(uint32x2_t a, uint32x2_t b) {
  return vorn_u32(a, b);
}

// CHECK-LABEL: test_vorn_u64
// CHECK: vorn d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vorn_u64(uint64x1_t a, uint64x1_t b) {
  return vorn_u64(a, b);
}

// CHECK-LABEL: test_vornq_s8
// CHECK: vorn q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vornq_s8(int8x16_t a, int8x16_t b) {
  return vornq_s8(a, b);
}

// CHECK-LABEL: test_vornq_s16
// CHECK: vorn q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vornq_s16(int16x8_t a, int16x8_t b) {
  return vornq_s16(a, b);
}

// CHECK-LABEL: test_vornq_s32
// CHECK: vorn q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vornq_s32(int32x4_t a, int32x4_t b) {
  return vornq_s32(a, b);
}

// CHECK-LABEL: test_vornq_s64
// CHECK: vorn q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vornq_s64(int64x2_t a, int64x2_t b) {
  return vornq_s64(a, b);
}

// CHECK-LABEL: test_vornq_u8
// CHECK: vorn q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vornq_u8(uint8x16_t a, uint8x16_t b) {
  return vornq_u8(a, b);
}

// CHECK-LABEL: test_vornq_u16
// CHECK: vorn q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vornq_u16(uint16x8_t a, uint16x8_t b) {
  return vornq_u16(a, b);
}

// CHECK-LABEL: test_vornq_u32
// CHECK: vorn q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vornq_u32(uint32x4_t a, uint32x4_t b) {
  return vornq_u32(a, b);
}

// CHECK-LABEL: test_vornq_u64
// CHECK: vorn q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vornq_u64(uint64x2_t a, uint64x2_t b) {
  return vornq_u64(a, b);
}


// CHECK-LABEL: test_vorr_s8
// CHECK: vorr d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vorr_s8(int8x8_t a, int8x8_t b) {
  return vorr_s8(a, b);
}

// CHECK-LABEL: test_vorr_s16
// CHECK: vorr d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vorr_s16(int16x4_t a, int16x4_t b) {
  return vorr_s16(a, b);
}

// CHECK-LABEL: test_vorr_s32
// CHECK: vorr d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vorr_s32(int32x2_t a, int32x2_t b) {
  return vorr_s32(a, b);
}

// CHECK-LABEL: test_vorr_s64
// CHECK: vorr d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vorr_s64(int64x1_t a, int64x1_t b) {
  return vorr_s64(a, b);
}

// CHECK-LABEL: test_vorr_u8
// CHECK: vorr d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vorr_u8(uint8x8_t a, uint8x8_t b) {
  return vorr_u8(a, b);
}

// CHECK-LABEL: test_vorr_u16
// CHECK: vorr d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vorr_u16(uint16x4_t a, uint16x4_t b) {
  return vorr_u16(a, b);
}

// CHECK-LABEL: test_vorr_u32
// CHECK: vorr d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vorr_u32(uint32x2_t a, uint32x2_t b) {
  return vorr_u32(a, b);
}

// CHECK-LABEL: test_vorr_u64
// CHECK: vorr d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vorr_u64(uint64x1_t a, uint64x1_t b) {
  return vorr_u64(a, b);
}

// CHECK-LABEL: test_vorrq_s8
// CHECK: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vorrq_s8(int8x16_t a, int8x16_t b) {
  return vorrq_s8(a, b);
}

// CHECK-LABEL: test_vorrq_s16
// CHECK: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vorrq_s16(int16x8_t a, int16x8_t b) {
  return vorrq_s16(a, b);
}

// CHECK-LABEL: test_vorrq_s32
// CHECK: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vorrq_s32(int32x4_t a, int32x4_t b) {
  return vorrq_s32(a, b);
}

// CHECK-LABEL: test_vorrq_s64
// CHECK: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vorrq_s64(int64x2_t a, int64x2_t b) {
  return vorrq_s64(a, b);
}

// CHECK-LABEL: test_vorrq_u8
// CHECK: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vorrq_u8(uint8x16_t a, uint8x16_t b) {
  return vorrq_u8(a, b);
}

// CHECK-LABEL: test_vorrq_u16
// CHECK: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vorrq_u16(uint16x8_t a, uint16x8_t b) {
  return vorrq_u16(a, b);
}

// CHECK-LABEL: test_vorrq_u32
// CHECK: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vorrq_u32(uint32x4_t a, uint32x4_t b) {
  return vorrq_u32(a, b);
}

// CHECK-LABEL: test_vorrq_u64
// CHECK: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vorrq_u64(uint64x2_t a, uint64x2_t b) {
  return vorrq_u64(a, b);
}


// CHECK-LABEL: test_vpadal_s8
// CHECK: vpadal.s8 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vpadal_s8(int16x4_t a, int8x8_t b) {
  return vpadal_s8(a, b);
}

// CHECK-LABEL: test_vpadal_s16
// CHECK: vpadal.s16 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vpadal_s16(int32x2_t a, int16x4_t b) {
  return vpadal_s16(a, b);
}

// CHECK-LABEL: test_vpadal_s32
// CHECK: vpadal.s32 d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vpadal_s32(int64x1_t a, int32x2_t b) {
  return vpadal_s32(a, b);
}

// CHECK-LABEL: test_vpadal_u8
// CHECK: vpadal.u8 d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vpadal_u8(uint16x4_t a, uint8x8_t b) {
  return vpadal_u8(a, b);
}

// CHECK-LABEL: test_vpadal_u16
// CHECK: vpadal.u16 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vpadal_u16(uint32x2_t a, uint16x4_t b) {
  return vpadal_u16(a, b);
}

// CHECK-LABEL: test_vpadal_u32
// CHECK: vpadal.u32 d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vpadal_u32(uint64x1_t a, uint32x2_t b) {
  return vpadal_u32(a, b);
}

// CHECK-LABEL: test_vpadalq_s8
// CHECK: vpadal.s8 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vpadalq_s8(int16x8_t a, int8x16_t b) {
  return vpadalq_s8(a, b);
}

// CHECK-LABEL: test_vpadalq_s16
// CHECK: vpadal.s16 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vpadalq_s16(int32x4_t a, int16x8_t b) {
  return vpadalq_s16(a, b);
}

// CHECK-LABEL: test_vpadalq_s32
// CHECK: vpadal.s32 q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vpadalq_s32(int64x2_t a, int32x4_t b) {
  return vpadalq_s32(a, b);
}

// CHECK-LABEL: test_vpadalq_u8
// CHECK: vpadal.u8 q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vpadalq_u8(uint16x8_t a, uint8x16_t b) {
  return vpadalq_u8(a, b);
}

// CHECK-LABEL: test_vpadalq_u16
// CHECK: vpadal.u16 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vpadalq_u16(uint32x4_t a, uint16x8_t b) {
  return vpadalq_u16(a, b);
}

// CHECK-LABEL: test_vpadalq_u32
// CHECK: vpadal.u32 q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vpadalq_u32(uint64x2_t a, uint32x4_t b) {
  return vpadalq_u32(a, b);
}


// CHECK-LABEL: test_vpadd_s8
// CHECK: vpadd.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vpadd_s8(int8x8_t a, int8x8_t b) {
  return vpadd_s8(a, b);
}

// CHECK-LABEL: test_vpadd_s16
// CHECK: vpadd.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vpadd_s16(int16x4_t a, int16x4_t b) {
  return vpadd_s16(a, b);
}

// CHECK-LABEL: test_vpadd_s32
// CHECK: vpadd.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vpadd_s32(int32x2_t a, int32x2_t b) {
  return vpadd_s32(a, b);
}

// CHECK-LABEL: test_vpadd_u8
// CHECK: vpadd.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vpadd_u8(uint8x8_t a, uint8x8_t b) {
  return vpadd_u8(a, b);
}

// CHECK-LABEL: test_vpadd_u16
// CHECK: vpadd.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vpadd_u16(uint16x4_t a, uint16x4_t b) {
  return vpadd_u16(a, b);
}

// CHECK-LABEL: test_vpadd_u32
// CHECK: vpadd.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vpadd_u32(uint32x2_t a, uint32x2_t b) {
  return vpadd_u32(a, b);
}

// CHECK-LABEL: test_vpadd_f32
// CHECK: vpadd.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vpadd_f32(float32x2_t a, float32x2_t b) {
  return vpadd_f32(a, b);
}


// CHECK-LABEL: test_vpaddl_s8
// CHECK: vpaddl.s8 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vpaddl_s8(int8x8_t a) {
  return vpaddl_s8(a);
}

// CHECK-LABEL: test_vpaddl_s16
// CHECK: vpaddl.s16 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vpaddl_s16(int16x4_t a) {
  return vpaddl_s16(a);
}

// CHECK-LABEL: test_vpaddl_s32
// CHECK: vpaddl.s32 d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vpaddl_s32(int32x2_t a) {
  return vpaddl_s32(a);
}

// CHECK-LABEL: test_vpaddl_u8
// CHECK: vpaddl.u8 d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vpaddl_u8(uint8x8_t a) {
  return vpaddl_u8(a);
}

// CHECK-LABEL: test_vpaddl_u16
// CHECK: vpaddl.u16 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vpaddl_u16(uint16x4_t a) {
  return vpaddl_u16(a);
}

// CHECK-LABEL: test_vpaddl_u32
// CHECK: vpaddl.u32 d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vpaddl_u32(uint32x2_t a) {
  return vpaddl_u32(a);
}

// CHECK-LABEL: test_vpaddlq_s8
// CHECK: vpaddl.s8 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vpaddlq_s8(int8x16_t a) {
  return vpaddlq_s8(a);
}

// CHECK-LABEL: test_vpaddlq_s16
// CHECK: vpaddl.s16 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vpaddlq_s16(int16x8_t a) {
  return vpaddlq_s16(a);
}

// CHECK-LABEL: test_vpaddlq_s32
// CHECK: vpaddl.s32 q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vpaddlq_s32(int32x4_t a) {
  return vpaddlq_s32(a);
}

// CHECK-LABEL: test_vpaddlq_u8
// CHECK: vpaddl.u8 q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vpaddlq_u8(uint8x16_t a) {
  return vpaddlq_u8(a);
}

// CHECK-LABEL: test_vpaddlq_u16
// CHECK: vpaddl.u16 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vpaddlq_u16(uint16x8_t a) {
  return vpaddlq_u16(a);
}

// CHECK-LABEL: test_vpaddlq_u32
// CHECK: vpaddl.u32 q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vpaddlq_u32(uint32x4_t a) {
  return vpaddlq_u32(a);
}


// CHECK-LABEL: test_vpmax_s8
// CHECK: vpmax.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vpmax_s8(int8x8_t a, int8x8_t b) {
  return vpmax_s8(a, b);
}

// CHECK-LABEL: test_vpmax_s16
// CHECK: vpmax.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vpmax_s16(int16x4_t a, int16x4_t b) {
  return vpmax_s16(a, b);
}

// CHECK-LABEL: test_vpmax_s32
// CHECK: vpmax.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vpmax_s32(int32x2_t a, int32x2_t b) {
  return vpmax_s32(a, b);
}

// CHECK-LABEL: test_vpmax_u8
// CHECK: vpmax.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vpmax_u8(uint8x8_t a, uint8x8_t b) {
  return vpmax_u8(a, b);
}

// CHECK-LABEL: test_vpmax_u16
// CHECK: vpmax.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vpmax_u16(uint16x4_t a, uint16x4_t b) {
  return vpmax_u16(a, b);
}

// CHECK-LABEL: test_vpmax_u32
// CHECK: vpmax.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vpmax_u32(uint32x2_t a, uint32x2_t b) {
  return vpmax_u32(a, b);
}

// CHECK-LABEL: test_vpmax_f32
// CHECK: vpmax.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vpmax_f32(float32x2_t a, float32x2_t b) {
  return vpmax_f32(a, b);
}


// CHECK-LABEL: test_vpmin_s8
// CHECK: vpmin.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vpmin_s8(int8x8_t a, int8x8_t b) {
  return vpmin_s8(a, b);
}

// CHECK-LABEL: test_vpmin_s16
// CHECK: vpmin.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vpmin_s16(int16x4_t a, int16x4_t b) {
  return vpmin_s16(a, b);
}

// CHECK-LABEL: test_vpmin_s32
// CHECK: vpmin.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vpmin_s32(int32x2_t a, int32x2_t b) {
  return vpmin_s32(a, b);
}

// CHECK-LABEL: test_vpmin_u8
// CHECK: vpmin.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vpmin_u8(uint8x8_t a, uint8x8_t b) {
  return vpmin_u8(a, b);
}

// CHECK-LABEL: test_vpmin_u16
// CHECK: vpmin.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vpmin_u16(uint16x4_t a, uint16x4_t b) {
  return vpmin_u16(a, b);
}

// CHECK-LABEL: test_vpmin_u32
// CHECK: vpmin.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vpmin_u32(uint32x2_t a, uint32x2_t b) {
  return vpmin_u32(a, b);
}

// CHECK-LABEL: test_vpmin_f32
// CHECK: vpmin.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vpmin_f32(float32x2_t a, float32x2_t b) {
  return vpmin_f32(a, b);
}


// CHECK-LABEL: test_vqabs_s8
// CHECK: vqabs.s8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vqabs_s8(int8x8_t a) {
  return vqabs_s8(a);
}

// CHECK-LABEL: test_vqabs_s16
// CHECK: vqabs.s16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqabs_s16(int16x4_t a) {
  return vqabs_s16(a);
}

// CHECK-LABEL: test_vqabs_s32
// CHECK: vqabs.s32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqabs_s32(int32x2_t a) {
  return vqabs_s32(a);
}

// CHECK-LABEL: test_vqabsq_s8
// CHECK: vqabs.s8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vqabsq_s8(int8x16_t a) {
  return vqabsq_s8(a);
}

// CHECK-LABEL: test_vqabsq_s16
// CHECK: vqabs.s16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqabsq_s16(int16x8_t a) {
  return vqabsq_s16(a);
}

// CHECK-LABEL: test_vqabsq_s32
// CHECK: vqabs.s32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqabsq_s32(int32x4_t a) {
  return vqabsq_s32(a);
}


// CHECK-LABEL: test_vqadd_s8
// CHECK: vqadd.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vqadd_s8(int8x8_t a, int8x8_t b) {
  return vqadd_s8(a, b);
}

// CHECK-LABEL: test_vqadd_s16
// CHECK: vqadd.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqadd_s16(int16x4_t a, int16x4_t b) {
  return vqadd_s16(a, b);
}

// CHECK-LABEL: test_vqadd_s32
// CHECK: vqadd.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqadd_s32(int32x2_t a, int32x2_t b) {
  return vqadd_s32(a, b);
}

// CHECK-LABEL: test_vqadd_s64
// CHECK: vqadd.s64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vqadd_s64(int64x1_t a, int64x1_t b) {
  return vqadd_s64(a, b);
}

// CHECK-LABEL: test_vqadd_u8
// CHECK: vqadd.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vqadd_u8(uint8x8_t a, uint8x8_t b) {
  return vqadd_u8(a, b);
}

// CHECK-LABEL: test_vqadd_u16
// CHECK: vqadd.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vqadd_u16(uint16x4_t a, uint16x4_t b) {
  return vqadd_u16(a, b);
}

// CHECK-LABEL: test_vqadd_u32
// CHECK: vqadd.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vqadd_u32(uint32x2_t a, uint32x2_t b) {
  return vqadd_u32(a, b);
}

// CHECK-LABEL: test_vqadd_u64
// CHECK: vqadd.u64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vqadd_u64(uint64x1_t a, uint64x1_t b) {
  return vqadd_u64(a, b);
}

// CHECK-LABEL: test_vqaddq_s8
// CHECK: vqadd.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vqaddq_s8(int8x16_t a, int8x16_t b) {
  return vqaddq_s8(a, b);
}

// CHECK-LABEL: test_vqaddq_s16
// CHECK: vqadd.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqaddq_s16(int16x8_t a, int16x8_t b) {
  return vqaddq_s16(a, b);
}

// CHECK-LABEL: test_vqaddq_s32
// CHECK: vqadd.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqaddq_s32(int32x4_t a, int32x4_t b) {
  return vqaddq_s32(a, b);
}

// CHECK-LABEL: test_vqaddq_s64
// CHECK: vqadd.s64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vqaddq_s64(int64x2_t a, int64x2_t b) {
  return vqaddq_s64(a, b);
}

// CHECK-LABEL: test_vqaddq_u8
// CHECK: vqadd.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vqaddq_u8(uint8x16_t a, uint8x16_t b) {
  return vqaddq_u8(a, b);
}

// CHECK-LABEL: test_vqaddq_u16
// CHECK: vqadd.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vqaddq_u16(uint16x8_t a, uint16x8_t b) {
  return vqaddq_u16(a, b);
}

// CHECK-LABEL: test_vqaddq_u32
// CHECK: vqadd.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vqaddq_u32(uint32x4_t a, uint32x4_t b) {
  return vqaddq_u32(a, b);
}

// CHECK-LABEL: test_vqaddq_u64
// CHECK: vqadd.u64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vqaddq_u64(uint64x2_t a, uint64x2_t b) {
  return vqaddq_u64(a, b);
}


// CHECK-LABEL: test_vqdmlal_s16
// CHECK: vqdmlal.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vqdmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vqdmlal_s16(a, b, c);
}

// CHECK-LABEL: test_vqdmlal_s32
// CHECK: vqdmlal.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vqdmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vqdmlal_s32(a, b, c);
}


// CHECK-LABEL: test_vqdmlal_lane_s16
// CHECK: vqdmlal.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vqdmlal_lane_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vqdmlal_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqdmlal_lane_s32
// CHECK: vqdmlal.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int64x2_t test_vqdmlal_lane_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vqdmlal_lane_s32(a, b, c, 1);
}


// CHECK-LABEL: test_vqdmlal_n_s16
// CHECK: vqdmlal.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vqdmlal_n_s16(int32x4_t a, int16x4_t b, int16_t c) {
  return vqdmlal_n_s16(a, b, c);
}

// CHECK-LABEL: test_vqdmlal_n_s32
// CHECK: vqdmlal.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vqdmlal_n_s32(int64x2_t a, int32x2_t b, int32_t c) {
  return vqdmlal_n_s32(a, b, c);
}


// CHECK-LABEL: test_vqdmlsl_s16
// CHECK: vqdmlsl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vqdmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vqdmlsl_s16(a, b, c);
}

// CHECK-LABEL: test_vqdmlsl_s32
// CHECK: vqdmlsl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vqdmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vqdmlsl_s32(a, b, c);
}


// CHECK-LABEL: test_vqdmlsl_lane_s16
// CHECK: vqdmlsl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vqdmlsl_lane_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vqdmlsl_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqdmlsl_lane_s32
// CHECK: vqdmlsl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int64x2_t test_vqdmlsl_lane_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vqdmlsl_lane_s32(a, b, c, 1);
}


// CHECK-LABEL: test_vqdmlsl_n_s16
// CHECK: vqdmlsl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vqdmlsl_n_s16(int32x4_t a, int16x4_t b, int16_t c) {
  return vqdmlsl_n_s16(a, b, c);
}

// CHECK-LABEL: test_vqdmlsl_n_s32
// CHECK: vqdmlsl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vqdmlsl_n_s32(int64x2_t a, int32x2_t b, int32_t c) {
  return vqdmlsl_n_s32(a, b, c);
}


// CHECK-LABEL: test_vqdmulh_s16
// CHECK: vqdmulh.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqdmulh_s16(int16x4_t a, int16x4_t b) {
  return vqdmulh_s16(a, b);
}

// CHECK-LABEL: test_vqdmulh_s32
// CHECK: vqdmulh.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqdmulh_s32(int32x2_t a, int32x2_t b) {
  return vqdmulh_s32(a, b);
}

// CHECK-LABEL: test_vqdmulhq_s16
// CHECK: vqdmulh.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqdmulhq_s16(int16x8_t a, int16x8_t b) {
  return vqdmulhq_s16(a, b);
}

// CHECK-LABEL: test_vqdmulhq_s32
// CHECK: vqdmulh.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqdmulhq_s32(int32x4_t a, int32x4_t b) {
  return vqdmulhq_s32(a, b);
}


// CHECK-LABEL: test_vqdmulh_lane_s16
// CHECK: vqdmulh.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x4_t test_vqdmulh_lane_s16(int16x4_t a, int16x4_t b) {
  return vqdmulh_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vqdmulh_lane_s32
// CHECK: vqdmulh.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x2_t test_vqdmulh_lane_s32(int32x2_t a, int32x2_t b) {
  return vqdmulh_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vqdmulhq_lane_s16
// CHECK: vqdmulh.s16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x8_t test_vqdmulhq_lane_s16(int16x8_t a, int16x4_t b) {
  return vqdmulhq_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vqdmulhq_lane_s32
// CHECK: vqdmulh.s32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vqdmulhq_lane_s32(int32x4_t a, int32x2_t b) {
  return vqdmulhq_lane_s32(a, b, 1);
}


// CHECK-LABEL: test_vqdmulh_n_s16
// CHECK: vqdmulh.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqdmulh_n_s16(int16x4_t a, int16_t b) {
  return vqdmulh_n_s16(a, b);
}

// CHECK-LABEL: test_vqdmulh_n_s32
// CHECK: vqdmulh.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqdmulh_n_s32(int32x2_t a, int32_t b) {
  return vqdmulh_n_s32(a, b);
}

// CHECK-LABEL: test_vqdmulhq_n_s16
// CHECK: vqdmulh.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqdmulhq_n_s16(int16x8_t a, int16_t b) {
  return vqdmulhq_n_s16(a, b);
}

// CHECK-LABEL: test_vqdmulhq_n_s32
// CHECK: vqdmulh.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqdmulhq_n_s32(int32x4_t a, int32_t b) {
  return vqdmulhq_n_s32(a, b);
}


// CHECK-LABEL: test_vqdmull_s16
// CHECK: vqdmull.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vqdmull_s16(int16x4_t a, int16x4_t b) {
  return vqdmull_s16(a, b);
}

// CHECK-LABEL: test_vqdmull_s32
// CHECK: vqdmull.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vqdmull_s32(int32x2_t a, int32x2_t b) {
  return vqdmull_s32(a, b);
}


// CHECK-LABEL: test_vqdmull_lane_s16
// CHECK: vqdmull.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vqdmull_lane_s16(int16x4_t a, int16x4_t b) {
  return vqdmull_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vqdmull_lane_s32
// CHECK: vqdmull.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int64x2_t test_vqdmull_lane_s32(int32x2_t a, int32x2_t b) {
  return vqdmull_lane_s32(a, b, 1);
}


// CHECK-LABEL: test_vqdmull_n_s16
// CHECK: vqdmull.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vqdmull_n_s16(int16x4_t a, int16_t b) {
  return vqdmull_n_s16(a, b);
}

// CHECK-LABEL: test_vqdmull_n_s32
// CHECK: vqdmull.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vqdmull_n_s32(int32x2_t a, int32_t b) {
  return vqdmull_n_s32(a, b);
}


// CHECK-LABEL: test_vqmovn_s16
// CHECK: vqmovn.s16 d{{[0-9]+}}, q{{[0-9]+}}
int8x8_t test_vqmovn_s16(int16x8_t a) {
  return vqmovn_s16(a);
}

// CHECK-LABEL: test_vqmovn_s32
// CHECK: vqmovn.s32 d{{[0-9]+}}, q{{[0-9]+}}
int16x4_t test_vqmovn_s32(int32x4_t a) {
  return vqmovn_s32(a);
}

// CHECK-LABEL: test_vqmovn_s64
// CHECK: vqmovn.s64 d{{[0-9]+}}, q{{[0-9]+}}
int32x2_t test_vqmovn_s64(int64x2_t a) {
  return vqmovn_s64(a);
}

// CHECK-LABEL: test_vqmovn_u16
// CHECK: vqmovn.u16 d{{[0-9]+}}, q{{[0-9]+}}
uint8x8_t test_vqmovn_u16(uint16x8_t a) {
  return vqmovn_u16(a);
}

// CHECK-LABEL: test_vqmovn_u32
// CHECK: vqmovn.u32 d{{[0-9]+}}, q{{[0-9]+}}
uint16x4_t test_vqmovn_u32(uint32x4_t a) {
  return vqmovn_u32(a);
}

// CHECK-LABEL: test_vqmovn_u64
// CHECK: vqmovn.u64 d{{[0-9]+}}, q{{[0-9]+}}
uint32x2_t test_vqmovn_u64(uint64x2_t a) {
  return vqmovn_u64(a);
}


// CHECK-LABEL: test_vqmovun_s16
// CHECK: vqmovun.s16 d{{[0-9]+}}, q{{[0-9]+}}
uint8x8_t test_vqmovun_s16(int16x8_t a) {
  return vqmovun_s16(a);
}

// CHECK-LABEL: test_vqmovun_s32
// CHECK: vqmovun.s32 d{{[0-9]+}}, q{{[0-9]+}}
uint16x4_t test_vqmovun_s32(int32x4_t a) {
  return vqmovun_s32(a);
}

// CHECK-LABEL: test_vqmovun_s64
// CHECK: vqmovun.s64 d{{[0-9]+}}, q{{[0-9]+}}
uint32x2_t test_vqmovun_s64(int64x2_t a) {
  return vqmovun_s64(a);
}


// CHECK-LABEL: test_vqneg_s8
// CHECK: vqneg.s8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vqneg_s8(int8x8_t a) {
  return vqneg_s8(a);
}

// CHECK-LABEL: test_vqneg_s16
// CHECK: vqneg.s16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqneg_s16(int16x4_t a) {
  return vqneg_s16(a);
}

// CHECK-LABEL: test_vqneg_s32
// CHECK: vqneg.s32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqneg_s32(int32x2_t a) {
  return vqneg_s32(a);
}

// CHECK-LABEL: test_vqnegq_s8
// CHECK: vqneg.s8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vqnegq_s8(int8x16_t a) {
  return vqnegq_s8(a);
}

// CHECK-LABEL: test_vqnegq_s16
// CHECK: vqneg.s16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqnegq_s16(int16x8_t a) {
  return vqnegq_s16(a);
}

// CHECK-LABEL: test_vqnegq_s32
// CHECK: vqneg.s32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqnegq_s32(int32x4_t a) {
  return vqnegq_s32(a);
}


// CHECK-LABEL: test_vqrdmulh_s16
// CHECK: vqrdmulh.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqrdmulh_s16(int16x4_t a, int16x4_t b) {
  return vqrdmulh_s16(a, b);
}

// CHECK-LABEL: test_vqrdmulh_s32
// CHECK: vqrdmulh.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqrdmulh_s32(int32x2_t a, int32x2_t b) {
  return vqrdmulh_s32(a, b);
}

// CHECK-LABEL: test_vqrdmulhq_s16
// CHECK: vqrdmulh.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqrdmulhq_s16(int16x8_t a, int16x8_t b) {
  return vqrdmulhq_s16(a, b);
}

// CHECK-LABEL: test_vqrdmulhq_s32
// CHECK: vqrdmulh.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqrdmulhq_s32(int32x4_t a, int32x4_t b) {
  return vqrdmulhq_s32(a, b);
}


// CHECK-LABEL: test_vqrdmulh_lane_s16
// CHECK: vqrdmulh.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x4_t test_vqrdmulh_lane_s16(int16x4_t a, int16x4_t b) {
  return vqrdmulh_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vqrdmulh_lane_s32
// CHECK: vqrdmulh.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x2_t test_vqrdmulh_lane_s32(int32x2_t a, int32x2_t b) {
  return vqrdmulh_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vqrdmulhq_lane_s16
// CHECK: vqrdmulh.s16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int16x8_t test_vqrdmulhq_lane_s16(int16x8_t a, int16x4_t b) {
  return vqrdmulhq_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vqrdmulhq_lane_s32
// CHECK: vqrdmulh.s32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[{{[0-9]}}]
int32x4_t test_vqrdmulhq_lane_s32(int32x4_t a, int32x2_t b) {
  return vqrdmulhq_lane_s32(a, b, 1);
}


// CHECK-LABEL: test_vqrdmulh_n_s16
// CHECK: vqrdmulh.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqrdmulh_n_s16(int16x4_t a, int16_t b) {
  return vqrdmulh_n_s16(a, b);
}

// CHECK-LABEL: test_vqrdmulh_n_s32
// CHECK: vqrdmulh.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqrdmulh_n_s32(int32x2_t a, int32_t b) {
  return vqrdmulh_n_s32(a, b);
}

// CHECK-LABEL: test_vqrdmulhq_n_s16
// CHECK: vqrdmulh.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqrdmulhq_n_s16(int16x8_t a, int16_t b) {
  return vqrdmulhq_n_s16(a, b);
}

// CHECK-LABEL: test_vqrdmulhq_n_s32
// CHECK: vqrdmulh.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqrdmulhq_n_s32(int32x4_t a, int32_t b) {
  return vqrdmulhq_n_s32(a, b);
}


// CHECK-LABEL: test_vqrshl_s8
// CHECK: vqrshl.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vqrshl_s8(int8x8_t a, int8x8_t b) {
  return vqrshl_s8(a, b);
}

// CHECK-LABEL: test_vqrshl_s16
// CHECK: vqrshl.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqrshl_s16(int16x4_t a, int16x4_t b) {
  return vqrshl_s16(a, b);
}

// CHECK-LABEL: test_vqrshl_s32
// CHECK: vqrshl.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqrshl_s32(int32x2_t a, int32x2_t b) {
  return vqrshl_s32(a, b);
}

// CHECK-LABEL: test_vqrshl_s64
// CHECK: vqrshl.s64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vqrshl_s64(int64x1_t a, int64x1_t b) {
  return vqrshl_s64(a, b);
}

// CHECK-LABEL: test_vqrshl_u8
// CHECK: vqrshl.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vqrshl_u8(uint8x8_t a, int8x8_t b) {
  return vqrshl_u8(a, b);
}

// CHECK-LABEL: test_vqrshl_u16
// CHECK: vqrshl.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vqrshl_u16(uint16x4_t a, int16x4_t b) {
  return vqrshl_u16(a, b);
}

// CHECK-LABEL: test_vqrshl_u32
// CHECK: vqrshl.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vqrshl_u32(uint32x2_t a, int32x2_t b) {
  return vqrshl_u32(a, b);
}

// CHECK-LABEL: test_vqrshl_u64
// CHECK: vqrshl.u64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vqrshl_u64(uint64x1_t a, int64x1_t b) {
  return vqrshl_u64(a, b);
}

// CHECK-LABEL: test_vqrshlq_s8
// CHECK: vqrshl.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vqrshlq_s8(int8x16_t a, int8x16_t b) {
  return vqrshlq_s8(a, b);
}

// CHECK-LABEL: test_vqrshlq_s16
// CHECK: vqrshl.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqrshlq_s16(int16x8_t a, int16x8_t b) {
  return vqrshlq_s16(a, b);
}

// CHECK-LABEL: test_vqrshlq_s32
// CHECK: vqrshl.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqrshlq_s32(int32x4_t a, int32x4_t b) {
  return vqrshlq_s32(a, b);
}

// CHECK-LABEL: test_vqrshlq_s64
// CHECK: vqrshl.s64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vqrshlq_s64(int64x2_t a, int64x2_t b) {
  return vqrshlq_s64(a, b);
}

// CHECK-LABEL: test_vqrshlq_u8
// CHECK: vqrshl.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vqrshlq_u8(uint8x16_t a, int8x16_t b) {
  return vqrshlq_u8(a, b);
}

// CHECK-LABEL: test_vqrshlq_u16
// CHECK: vqrshl.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vqrshlq_u16(uint16x8_t a, int16x8_t b) {
  return vqrshlq_u16(a, b);
}

// CHECK-LABEL: test_vqrshlq_u32
// CHECK: vqrshl.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vqrshlq_u32(uint32x4_t a, int32x4_t b) {
  return vqrshlq_u32(a, b);
}

// CHECK-LABEL: test_vqrshlq_u64
// CHECK: vqrshl.u64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vqrshlq_u64(uint64x2_t a, int64x2_t b) {
  return vqrshlq_u64(a, b);
}


// CHECK-LABEL: test_vqrshrn_n_s16
// CHECK: vqrshrn.s16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vqrshrn_n_s16(int16x8_t a) {
  return vqrshrn_n_s16(a, 1);
}

// CHECK-LABEL: test_vqrshrn_n_s32
// CHECK: vqrshrn.s32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vqrshrn_n_s32(int32x4_t a) {
  return vqrshrn_n_s32(a, 1);
}

// CHECK-LABEL: test_vqrshrn_n_s64
// CHECK: vqrshrn.s64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vqrshrn_n_s64(int64x2_t a) {
  return vqrshrn_n_s64(a, 1);
}

// CHECK-LABEL: test_vqrshrn_n_u16
// CHECK: vqrshrn.u16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vqrshrn_n_u16(uint16x8_t a) {
  return vqrshrn_n_u16(a, 1);
}

// CHECK-LABEL: test_vqrshrn_n_u32
// CHECK: vqrshrn.u32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vqrshrn_n_u32(uint32x4_t a) {
  return vqrshrn_n_u32(a, 1);
}

// CHECK-LABEL: test_vqrshrn_n_u64
// CHECK: vqrshrn.u64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vqrshrn_n_u64(uint64x2_t a) {
  return vqrshrn_n_u64(a, 1);
}


// CHECK-LABEL: test_vqrshrun_n_s16
// CHECK: vqrshrun.s16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vqrshrun_n_s16(int16x8_t a) {
  return vqrshrun_n_s16(a, 1);
}

// CHECK-LABEL: test_vqrshrun_n_s32
// CHECK: vqrshrun.s32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vqrshrun_n_s32(int32x4_t a) {
  return vqrshrun_n_s32(a, 1);
}

// CHECK-LABEL: test_vqrshrun_n_s64
// CHECK: vqrshrun.s64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vqrshrun_n_s64(int64x2_t a) {
  return vqrshrun_n_s64(a, 1);
}


// CHECK-LABEL: test_vqshl_s8
// CHECK: vqshl.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vqshl_s8(int8x8_t a, int8x8_t b) {
  return vqshl_s8(a, b);
}

// CHECK-LABEL: test_vqshl_s16
// CHECK: vqshl.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqshl_s16(int16x4_t a, int16x4_t b) {
  return vqshl_s16(a, b);
}

// CHECK-LABEL: test_vqshl_s32
// CHECK: vqshl.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqshl_s32(int32x2_t a, int32x2_t b) {
  return vqshl_s32(a, b);
}

// CHECK-LABEL: test_vqshl_s64
// CHECK: vqshl.s64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vqshl_s64(int64x1_t a, int64x1_t b) {
  return vqshl_s64(a, b);
}

// CHECK-LABEL: test_vqshl_u8
// CHECK: vqshl.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vqshl_u8(uint8x8_t a, int8x8_t b) {
  return vqshl_u8(a, b);
}

// CHECK-LABEL: test_vqshl_u16
// CHECK: vqshl.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vqshl_u16(uint16x4_t a, int16x4_t b) {
  return vqshl_u16(a, b);
}

// CHECK-LABEL: test_vqshl_u32
// CHECK: vqshl.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vqshl_u32(uint32x2_t a, int32x2_t b) {
  return vqshl_u32(a, b);
}

// CHECK-LABEL: test_vqshl_u64
// CHECK: vqshl.u64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vqshl_u64(uint64x1_t a, int64x1_t b) {
  return vqshl_u64(a, b);
}

// CHECK-LABEL: test_vqshlq_s8
// CHECK: vqshl.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vqshlq_s8(int8x16_t a, int8x16_t b) {
  return vqshlq_s8(a, b);
}

// CHECK-LABEL: test_vqshlq_s16
// CHECK: vqshl.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqshlq_s16(int16x8_t a, int16x8_t b) {
  return vqshlq_s16(a, b);
}

// CHECK-LABEL: test_vqshlq_s32
// CHECK: vqshl.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqshlq_s32(int32x4_t a, int32x4_t b) {
  return vqshlq_s32(a, b);
}

// CHECK-LABEL: test_vqshlq_s64
// CHECK: vqshl.s64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vqshlq_s64(int64x2_t a, int64x2_t b) {
  return vqshlq_s64(a, b);
}

// CHECK-LABEL: test_vqshlq_u8
// CHECK: vqshl.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vqshlq_u8(uint8x16_t a, int8x16_t b) {
  return vqshlq_u8(a, b);
}

// CHECK-LABEL: test_vqshlq_u16
// CHECK: vqshl.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vqshlq_u16(uint16x8_t a, int16x8_t b) {
  return vqshlq_u16(a, b);
}

// CHECK-LABEL: test_vqshlq_u32
// CHECK: vqshl.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vqshlq_u32(uint32x4_t a, int32x4_t b) {
  return vqshlq_u32(a, b);
}

// CHECK-LABEL: test_vqshlq_u64
// CHECK: vqshl.u64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vqshlq_u64(uint64x2_t a, int64x2_t b) {
  return vqshlq_u64(a, b);
}


// CHECK-LABEL: test_vqshlu_n_s8
// CHECK: vqshlu.s8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vqshlu_n_s8(int8x8_t a) {
  return vqshlu_n_s8(a, 1);
}

// CHECK-LABEL: test_vqshlu_n_s16
// CHECK: vqshlu.s16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vqshlu_n_s16(int16x4_t a) {
  return vqshlu_n_s16(a, 1);
}

// CHECK-LABEL: test_vqshlu_n_s32
// CHECK: vqshlu.s32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vqshlu_n_s32(int32x2_t a) {
  return vqshlu_n_s32(a, 1);
}

// CHECK-LABEL: test_vqshlu_n_s64
// CHECK: vqshlu.s64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vqshlu_n_s64(int64x1_t a) {
  return vqshlu_n_s64(a, 1);
}

// CHECK-LABEL: test_vqshluq_n_s8
// CHECK: vqshlu.s8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vqshluq_n_s8(int8x16_t a) {
  return vqshluq_n_s8(a, 1);
}

// CHECK-LABEL: test_vqshluq_n_s16
// CHECK: vqshlu.s16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vqshluq_n_s16(int16x8_t a) {
  return vqshluq_n_s16(a, 1);
}

// CHECK-LABEL: test_vqshluq_n_s32
// CHECK: vqshlu.s32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vqshluq_n_s32(int32x4_t a) {
  return vqshluq_n_s32(a, 1);
}

// CHECK-LABEL: test_vqshluq_n_s64
// CHECK: vqshlu.s64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vqshluq_n_s64(int64x2_t a) {
  return vqshluq_n_s64(a, 1);
}


// CHECK-LABEL: test_vqshl_n_s8
// CHECK: vqshl.s8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vqshl_n_s8(int8x8_t a) {
  return vqshl_n_s8(a, 1);
}

// CHECK-LABEL: test_vqshl_n_s16
// CHECK: vqshl.s16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vqshl_n_s16(int16x4_t a) {
  return vqshl_n_s16(a, 1);
}

// CHECK-LABEL: test_vqshl_n_s32
// CHECK: vqshl.s32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vqshl_n_s32(int32x2_t a) {
  return vqshl_n_s32(a, 1);
}

// CHECK-LABEL: test_vqshl_n_s64
// CHECK: vqshl.s64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x1_t test_vqshl_n_s64(int64x1_t a) {
  return vqshl_n_s64(a, 1);
}

// CHECK-LABEL: test_vqshl_n_u8
// CHECK: vqshl.u8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vqshl_n_u8(uint8x8_t a) {
  return vqshl_n_u8(a, 1);
}

// CHECK-LABEL: test_vqshl_n_u16
// CHECK: vqshl.u16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vqshl_n_u16(uint16x4_t a) {
  return vqshl_n_u16(a, 1);
}

// CHECK-LABEL: test_vqshl_n_u32
// CHECK: vqshl.u32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vqshl_n_u32(uint32x2_t a) {
  return vqshl_n_u32(a, 1);
}

// CHECK-LABEL: test_vqshl_n_u64
// CHECK: vqshl.u64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vqshl_n_u64(uint64x1_t a) {
  return vqshl_n_u64(a, 1);
}

// CHECK-LABEL: test_vqshlq_n_s8
// CHECK: vqshl.s8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vqshlq_n_s8(int8x16_t a) {
  return vqshlq_n_s8(a, 1);
}

// CHECK-LABEL: test_vqshlq_n_s16
// CHECK: vqshl.s16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vqshlq_n_s16(int16x8_t a) {
  return vqshlq_n_s16(a, 1);
}

// CHECK-LABEL: test_vqshlq_n_s32
// CHECK: vqshl.s32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vqshlq_n_s32(int32x4_t a) {
  return vqshlq_n_s32(a, 1);
}

// CHECK-LABEL: test_vqshlq_n_s64
// CHECK: vqshl.s64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vqshlq_n_s64(int64x2_t a) {
  return vqshlq_n_s64(a, 1);
}

// CHECK-LABEL: test_vqshlq_n_u8
// CHECK: vqshl.u8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vqshlq_n_u8(uint8x16_t a) {
  return vqshlq_n_u8(a, 1);
}

// CHECK-LABEL: test_vqshlq_n_u16
// CHECK: vqshl.u16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vqshlq_n_u16(uint16x8_t a) {
  return vqshlq_n_u16(a, 1);
}

// CHECK-LABEL: test_vqshlq_n_u32
// CHECK: vqshl.u32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vqshlq_n_u32(uint32x4_t a) {
  return vqshlq_n_u32(a, 1);
}

// CHECK-LABEL: test_vqshlq_n_u64
// CHECK: vqshl.u64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vqshlq_n_u64(uint64x2_t a) {
  return vqshlq_n_u64(a, 1);
}


// CHECK-LABEL: test_vqshrn_n_s16
// CHECK: vqshrn.s16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vqshrn_n_s16(int16x8_t a) {
  return vqshrn_n_s16(a, 1);
}

// CHECK-LABEL: test_vqshrn_n_s32
// CHECK: vqshrn.s32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vqshrn_n_s32(int32x4_t a) {
  return vqshrn_n_s32(a, 1);
}

// CHECK-LABEL: test_vqshrn_n_s64
// CHECK: vqshrn.s64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vqshrn_n_s64(int64x2_t a) {
  return vqshrn_n_s64(a, 1);
}

// CHECK-LABEL: test_vqshrn_n_u16
// CHECK: vqshrn.u16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vqshrn_n_u16(uint16x8_t a) {
  return vqshrn_n_u16(a, 1);
}

// CHECK-LABEL: test_vqshrn_n_u32
// CHECK: vqshrn.u32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vqshrn_n_u32(uint32x4_t a) {
  return vqshrn_n_u32(a, 1);
}

// CHECK-LABEL: test_vqshrn_n_u64
// CHECK: vqshrn.u64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vqshrn_n_u64(uint64x2_t a) {
  return vqshrn_n_u64(a, 1);
}


// CHECK-LABEL: test_vqshrun_n_s16
// CHECK: vqshrun.s16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vqshrun_n_s16(int16x8_t a) {
  return vqshrun_n_s16(a, 1);
}

// CHECK-LABEL: test_vqshrun_n_s32
// CHECK: vqshrun.s32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vqshrun_n_s32(int32x4_t a) {
  return vqshrun_n_s32(a, 1);
}

// CHECK-LABEL: test_vqshrun_n_s64
// CHECK: vqshrun.s64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vqshrun_n_s64(int64x2_t a) {
  return vqshrun_n_s64(a, 1);
}


// CHECK-LABEL: test_vqsub_s8
// CHECK: vqsub.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vqsub_s8(int8x8_t a, int8x8_t b) {
  return vqsub_s8(a, b);
}

// CHECK-LABEL: test_vqsub_s16
// CHECK: vqsub.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vqsub_s16(int16x4_t a, int16x4_t b) {
  return vqsub_s16(a, b);
}

// CHECK-LABEL: test_vqsub_s32
// CHECK: vqsub.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vqsub_s32(int32x2_t a, int32x2_t b) {
  return vqsub_s32(a, b);
}

// CHECK-LABEL: test_vqsub_s64
// CHECK: vqsub.s64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vqsub_s64(int64x1_t a, int64x1_t b) {
  return vqsub_s64(a, b);
}

// CHECK-LABEL: test_vqsub_u8
// CHECK: vqsub.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vqsub_u8(uint8x8_t a, uint8x8_t b) {
  return vqsub_u8(a, b);
}

// CHECK-LABEL: test_vqsub_u16
// CHECK: vqsub.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vqsub_u16(uint16x4_t a, uint16x4_t b) {
  return vqsub_u16(a, b);
}

// CHECK-LABEL: test_vqsub_u32
// CHECK: vqsub.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vqsub_u32(uint32x2_t a, uint32x2_t b) {
  return vqsub_u32(a, b);
}

// CHECK-LABEL: test_vqsub_u64
// CHECK: vqsub.u64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vqsub_u64(uint64x1_t a, uint64x1_t b) {
  return vqsub_u64(a, b);
}

// CHECK-LABEL: test_vqsubq_s8
// CHECK: vqsub.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vqsubq_s8(int8x16_t a, int8x16_t b) {
  return vqsubq_s8(a, b);
}

// CHECK-LABEL: test_vqsubq_s16
// CHECK: vqsub.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vqsubq_s16(int16x8_t a, int16x8_t b) {
  return vqsubq_s16(a, b);
}

// CHECK-LABEL: test_vqsubq_s32
// CHECK: vqsub.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vqsubq_s32(int32x4_t a, int32x4_t b) {
  return vqsubq_s32(a, b);
}

// CHECK-LABEL: test_vqsubq_s64
// CHECK: vqsub.s64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vqsubq_s64(int64x2_t a, int64x2_t b) {
  return vqsubq_s64(a, b);
}

// CHECK-LABEL: test_vqsubq_u8
// CHECK: vqsub.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vqsubq_u8(uint8x16_t a, uint8x16_t b) {
  return vqsubq_u8(a, b);
}

// CHECK-LABEL: test_vqsubq_u16
// CHECK: vqsub.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vqsubq_u16(uint16x8_t a, uint16x8_t b) {
  return vqsubq_u16(a, b);
}

// CHECK-LABEL: test_vqsubq_u32
// CHECK: vqsub.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vqsubq_u32(uint32x4_t a, uint32x4_t b) {
  return vqsubq_u32(a, b);
}

// CHECK-LABEL: test_vqsubq_u64
// CHECK: vqsub.u64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vqsubq_u64(uint64x2_t a, uint64x2_t b) {
  return vqsubq_u64(a, b);
}


// CHECK-LABEL: test_vraddhn_s16
// CHECK: vraddhn.i16 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x8_t test_vraddhn_s16(int16x8_t a, int16x8_t b) {
  return vraddhn_s16(a, b);
}

// CHECK-LABEL: test_vraddhn_s32
// CHECK: vraddhn.i32 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x4_t test_vraddhn_s32(int32x4_t a, int32x4_t b) {
  return vraddhn_s32(a, b);
}

// CHECK-LABEL: test_vraddhn_s64
// CHECK: vraddhn.i64 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x2_t test_vraddhn_s64(int64x2_t a, int64x2_t b) {
  return vraddhn_s64(a, b);
}

// CHECK-LABEL: test_vraddhn_u16
// CHECK: vraddhn.i16 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x8_t test_vraddhn_u16(uint16x8_t a, uint16x8_t b) {
  return vraddhn_u16(a, b);
}

// CHECK-LABEL: test_vraddhn_u32
// CHECK: vraddhn.i32 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x4_t test_vraddhn_u32(uint32x4_t a, uint32x4_t b) {
  return vraddhn_u32(a, b);
}

// CHECK-LABEL: test_vraddhn_u64
// CHECK: vraddhn.i64 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x2_t test_vraddhn_u64(uint64x2_t a, uint64x2_t b) {
  return vraddhn_u64(a, b);
}


// CHECK-LABEL: test_vrecpe_f32
// CHECK: vrecpe.f32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vrecpe_f32(float32x2_t a) {
  return vrecpe_f32(a);
}

// CHECK-LABEL: test_vrecpe_u32
// CHECK: vrecpe.u32 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vrecpe_u32(uint32x2_t a) {
  return vrecpe_u32(a);
}

// CHECK-LABEL: test_vrecpeq_f32
// CHECK: vrecpe.f32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vrecpeq_f32(float32x4_t a) {
  return vrecpeq_f32(a);
}

// CHECK-LABEL: test_vrecpeq_u32
// CHECK: vrecpe.u32 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vrecpeq_u32(uint32x4_t a) {
  return vrecpeq_u32(a);
}


// CHECK-LABEL: test_vrecps_f32
// CHECK: vrecps.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vrecps_f32(float32x2_t a, float32x2_t b) {
  return vrecps_f32(a, b);
}

// CHECK-LABEL: test_vrecpsq_f32
// CHECK: vrecps.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vrecpsq_f32(float32x4_t a, float32x4_t b) {
  return vrecpsq_f32(a, b);
}


// CHECK-LABEL: test_vreinterpret_s8_s16
int8x8_t test_vreinterpret_s8_s16(int16x4_t a) {
  return vreinterpret_s8_s16(a);
}

// CHECK-LABEL: test_vreinterpret_s8_s32
int8x8_t test_vreinterpret_s8_s32(int32x2_t a) {
  return vreinterpret_s8_s32(a);
}

// CHECK-LABEL: test_vreinterpret_s8_s64
int8x8_t test_vreinterpret_s8_s64(int64x1_t a) {
  return vreinterpret_s8_s64(a);
}

// CHECK-LABEL: test_vreinterpret_s8_u8
int8x8_t test_vreinterpret_s8_u8(uint8x8_t a) {
  return vreinterpret_s8_u8(a);
}

// CHECK-LABEL: test_vreinterpret_s8_u16
int8x8_t test_vreinterpret_s8_u16(uint16x4_t a) {
  return vreinterpret_s8_u16(a);
}

// CHECK-LABEL: test_vreinterpret_s8_u32
int8x8_t test_vreinterpret_s8_u32(uint32x2_t a) {
  return vreinterpret_s8_u32(a);
}

// CHECK-LABEL: test_vreinterpret_s8_u64
int8x8_t test_vreinterpret_s8_u64(uint64x1_t a) {
  return vreinterpret_s8_u64(a);
}

// CHECK-LABEL: test_vreinterpret_s8_f16
int8x8_t test_vreinterpret_s8_f16(float16x4_t a) {
  return vreinterpret_s8_f16(a);
}

// CHECK-LABEL: test_vreinterpret_s8_f32
int8x8_t test_vreinterpret_s8_f32(float32x2_t a) {
  return vreinterpret_s8_f32(a);
}

// CHECK-LABEL: test_vreinterpret_s8_p8
int8x8_t test_vreinterpret_s8_p8(poly8x8_t a) {
  return vreinterpret_s8_p8(a);
}

// CHECK-LABEL: test_vreinterpret_s8_p16
int8x8_t test_vreinterpret_s8_p16(poly16x4_t a) {
  return vreinterpret_s8_p16(a);
}

// CHECK-LABEL: test_vreinterpret_s16_s8
int16x4_t test_vreinterpret_s16_s8(int8x8_t a) {
  return vreinterpret_s16_s8(a);
}

// CHECK-LABEL: test_vreinterpret_s16_s32
int16x4_t test_vreinterpret_s16_s32(int32x2_t a) {
  return vreinterpret_s16_s32(a);
}

// CHECK-LABEL: test_vreinterpret_s16_s64
int16x4_t test_vreinterpret_s16_s64(int64x1_t a) {
  return vreinterpret_s16_s64(a);
}

// CHECK-LABEL: test_vreinterpret_s16_u8
int16x4_t test_vreinterpret_s16_u8(uint8x8_t a) {
  return vreinterpret_s16_u8(a);
}

// CHECK-LABEL: test_vreinterpret_s16_u16
int16x4_t test_vreinterpret_s16_u16(uint16x4_t a) {
  return vreinterpret_s16_u16(a);
}

// CHECK-LABEL: test_vreinterpret_s16_u32
int16x4_t test_vreinterpret_s16_u32(uint32x2_t a) {
  return vreinterpret_s16_u32(a);
}

// CHECK-LABEL: test_vreinterpret_s16_u64
int16x4_t test_vreinterpret_s16_u64(uint64x1_t a) {
  return vreinterpret_s16_u64(a);
}

// CHECK-LABEL: test_vreinterpret_s16_f16
int16x4_t test_vreinterpret_s16_f16(float16x4_t a) {
  return vreinterpret_s16_f16(a);
}

// CHECK-LABEL: test_vreinterpret_s16_f32
int16x4_t test_vreinterpret_s16_f32(float32x2_t a) {
  return vreinterpret_s16_f32(a);
}

// CHECK-LABEL: test_vreinterpret_s16_p8
int16x4_t test_vreinterpret_s16_p8(poly8x8_t a) {
  return vreinterpret_s16_p8(a);
}

// CHECK-LABEL: test_vreinterpret_s16_p16
int16x4_t test_vreinterpret_s16_p16(poly16x4_t a) {
  return vreinterpret_s16_p16(a);
}

// CHECK-LABEL: test_vreinterpret_s32_s8
int32x2_t test_vreinterpret_s32_s8(int8x8_t a) {
  return vreinterpret_s32_s8(a);
}

// CHECK-LABEL: test_vreinterpret_s32_s16
int32x2_t test_vreinterpret_s32_s16(int16x4_t a) {
  return vreinterpret_s32_s16(a);
}

// CHECK-LABEL: test_vreinterpret_s32_s64
int32x2_t test_vreinterpret_s32_s64(int64x1_t a) {
  return vreinterpret_s32_s64(a);
}

// CHECK-LABEL: test_vreinterpret_s32_u8
int32x2_t test_vreinterpret_s32_u8(uint8x8_t a) {
  return vreinterpret_s32_u8(a);
}

// CHECK-LABEL: test_vreinterpret_s32_u16
int32x2_t test_vreinterpret_s32_u16(uint16x4_t a) {
  return vreinterpret_s32_u16(a);
}

// CHECK-LABEL: test_vreinterpret_s32_u32
int32x2_t test_vreinterpret_s32_u32(uint32x2_t a) {
  return vreinterpret_s32_u32(a);
}

// CHECK-LABEL: test_vreinterpret_s32_u64
int32x2_t test_vreinterpret_s32_u64(uint64x1_t a) {
  return vreinterpret_s32_u64(a);
}

// CHECK-LABEL: test_vreinterpret_s32_f16
int32x2_t test_vreinterpret_s32_f16(float16x4_t a) {
  return vreinterpret_s32_f16(a);
}

// CHECK-LABEL: test_vreinterpret_s32_f32
int32x2_t test_vreinterpret_s32_f32(float32x2_t a) {
  return vreinterpret_s32_f32(a);
}

// CHECK-LABEL: test_vreinterpret_s32_p8
int32x2_t test_vreinterpret_s32_p8(poly8x8_t a) {
  return vreinterpret_s32_p8(a);
}

// CHECK-LABEL: test_vreinterpret_s32_p16
int32x2_t test_vreinterpret_s32_p16(poly16x4_t a) {
  return vreinterpret_s32_p16(a);
}

// CHECK-LABEL: test_vreinterpret_s64_s8
int64x1_t test_vreinterpret_s64_s8(int8x8_t a) {
  return vreinterpret_s64_s8(a);
}

// CHECK-LABEL: test_vreinterpret_s64_s16
int64x1_t test_vreinterpret_s64_s16(int16x4_t a) {
  return vreinterpret_s64_s16(a);
}

// CHECK-LABEL: test_vreinterpret_s64_s32
int64x1_t test_vreinterpret_s64_s32(int32x2_t a) {
  return vreinterpret_s64_s32(a);
}

// CHECK-LABEL: test_vreinterpret_s64_u8
int64x1_t test_vreinterpret_s64_u8(uint8x8_t a) {
  return vreinterpret_s64_u8(a);
}

// CHECK-LABEL: test_vreinterpret_s64_u16
int64x1_t test_vreinterpret_s64_u16(uint16x4_t a) {
  return vreinterpret_s64_u16(a);
}

// CHECK-LABEL: test_vreinterpret_s64_u32
int64x1_t test_vreinterpret_s64_u32(uint32x2_t a) {
  return vreinterpret_s64_u32(a);
}

// CHECK-LABEL: test_vreinterpret_s64_u64
int64x1_t test_vreinterpret_s64_u64(uint64x1_t a) {
  return vreinterpret_s64_u64(a);
}

// CHECK-LABEL: test_vreinterpret_s64_f16
int64x1_t test_vreinterpret_s64_f16(float16x4_t a) {
  return vreinterpret_s64_f16(a);
}

// CHECK-LABEL: test_vreinterpret_s64_f32
int64x1_t test_vreinterpret_s64_f32(float32x2_t a) {
  return vreinterpret_s64_f32(a);
}

// CHECK-LABEL: test_vreinterpret_s64_p8
int64x1_t test_vreinterpret_s64_p8(poly8x8_t a) {
  return vreinterpret_s64_p8(a);
}

// CHECK-LABEL: test_vreinterpret_s64_p16
int64x1_t test_vreinterpret_s64_p16(poly16x4_t a) {
  return vreinterpret_s64_p16(a);
}

// CHECK-LABEL: test_vreinterpret_u8_s8
uint8x8_t test_vreinterpret_u8_s8(int8x8_t a) {
  return vreinterpret_u8_s8(a);
}

// CHECK-LABEL: test_vreinterpret_u8_s16
uint8x8_t test_vreinterpret_u8_s16(int16x4_t a) {
  return vreinterpret_u8_s16(a);
}

// CHECK-LABEL: test_vreinterpret_u8_s32
uint8x8_t test_vreinterpret_u8_s32(int32x2_t a) {
  return vreinterpret_u8_s32(a);
}

// CHECK-LABEL: test_vreinterpret_u8_s64
uint8x8_t test_vreinterpret_u8_s64(int64x1_t a) {
  return vreinterpret_u8_s64(a);
}

// CHECK-LABEL: test_vreinterpret_u8_u16
uint8x8_t test_vreinterpret_u8_u16(uint16x4_t a) {
  return vreinterpret_u8_u16(a);
}

// CHECK-LABEL: test_vreinterpret_u8_u32
uint8x8_t test_vreinterpret_u8_u32(uint32x2_t a) {
  return vreinterpret_u8_u32(a);
}

// CHECK-LABEL: test_vreinterpret_u8_u64
uint8x8_t test_vreinterpret_u8_u64(uint64x1_t a) {
  return vreinterpret_u8_u64(a);
}

// CHECK-LABEL: test_vreinterpret_u8_f16
uint8x8_t test_vreinterpret_u8_f16(float16x4_t a) {
  return vreinterpret_u8_f16(a);
}

// CHECK-LABEL: test_vreinterpret_u8_f32
uint8x8_t test_vreinterpret_u8_f32(float32x2_t a) {
  return vreinterpret_u8_f32(a);
}

// CHECK-LABEL: test_vreinterpret_u8_p8
uint8x8_t test_vreinterpret_u8_p8(poly8x8_t a) {
  return vreinterpret_u8_p8(a);
}

// CHECK-LABEL: test_vreinterpret_u8_p16
uint8x8_t test_vreinterpret_u8_p16(poly16x4_t a) {
  return vreinterpret_u8_p16(a);
}

// CHECK-LABEL: test_vreinterpret_u16_s8
uint16x4_t test_vreinterpret_u16_s8(int8x8_t a) {
  return vreinterpret_u16_s8(a);
}

// CHECK-LABEL: test_vreinterpret_u16_s16
uint16x4_t test_vreinterpret_u16_s16(int16x4_t a) {
  return vreinterpret_u16_s16(a);
}

// CHECK-LABEL: test_vreinterpret_u16_s32
uint16x4_t test_vreinterpret_u16_s32(int32x2_t a) {
  return vreinterpret_u16_s32(a);
}

// CHECK-LABEL: test_vreinterpret_u16_s64
uint16x4_t test_vreinterpret_u16_s64(int64x1_t a) {
  return vreinterpret_u16_s64(a);
}

// CHECK-LABEL: test_vreinterpret_u16_u8
uint16x4_t test_vreinterpret_u16_u8(uint8x8_t a) {
  return vreinterpret_u16_u8(a);
}

// CHECK-LABEL: test_vreinterpret_u16_u32
uint16x4_t test_vreinterpret_u16_u32(uint32x2_t a) {
  return vreinterpret_u16_u32(a);
}

// CHECK-LABEL: test_vreinterpret_u16_u64
uint16x4_t test_vreinterpret_u16_u64(uint64x1_t a) {
  return vreinterpret_u16_u64(a);
}

// CHECK-LABEL: test_vreinterpret_u16_f16
uint16x4_t test_vreinterpret_u16_f16(float16x4_t a) {
  return vreinterpret_u16_f16(a);
}

// CHECK-LABEL: test_vreinterpret_u16_f32
uint16x4_t test_vreinterpret_u16_f32(float32x2_t a) {
  return vreinterpret_u16_f32(a);
}

// CHECK-LABEL: test_vreinterpret_u16_p8
uint16x4_t test_vreinterpret_u16_p8(poly8x8_t a) {
  return vreinterpret_u16_p8(a);
}

// CHECK-LABEL: test_vreinterpret_u16_p16
uint16x4_t test_vreinterpret_u16_p16(poly16x4_t a) {
  return vreinterpret_u16_p16(a);
}

// CHECK-LABEL: test_vreinterpret_u32_s8
uint32x2_t test_vreinterpret_u32_s8(int8x8_t a) {
  return vreinterpret_u32_s8(a);
}

// CHECK-LABEL: test_vreinterpret_u32_s16
uint32x2_t test_vreinterpret_u32_s16(int16x4_t a) {
  return vreinterpret_u32_s16(a);
}

// CHECK-LABEL: test_vreinterpret_u32_s32
uint32x2_t test_vreinterpret_u32_s32(int32x2_t a) {
  return vreinterpret_u32_s32(a);
}

// CHECK-LABEL: test_vreinterpret_u32_s64
uint32x2_t test_vreinterpret_u32_s64(int64x1_t a) {
  return vreinterpret_u32_s64(a);
}

// CHECK-LABEL: test_vreinterpret_u32_u8
uint32x2_t test_vreinterpret_u32_u8(uint8x8_t a) {
  return vreinterpret_u32_u8(a);
}

// CHECK-LABEL: test_vreinterpret_u32_u16
uint32x2_t test_vreinterpret_u32_u16(uint16x4_t a) {
  return vreinterpret_u32_u16(a);
}

// CHECK-LABEL: test_vreinterpret_u32_u64
uint32x2_t test_vreinterpret_u32_u64(uint64x1_t a) {
  return vreinterpret_u32_u64(a);
}

// CHECK-LABEL: test_vreinterpret_u32_f16
uint32x2_t test_vreinterpret_u32_f16(float16x4_t a) {
  return vreinterpret_u32_f16(a);
}

// CHECK-LABEL: test_vreinterpret_u32_f32
uint32x2_t test_vreinterpret_u32_f32(float32x2_t a) {
  return vreinterpret_u32_f32(a);
}

// CHECK-LABEL: test_vreinterpret_u32_p8
uint32x2_t test_vreinterpret_u32_p8(poly8x8_t a) {
  return vreinterpret_u32_p8(a);
}

// CHECK-LABEL: test_vreinterpret_u32_p16
uint32x2_t test_vreinterpret_u32_p16(poly16x4_t a) {
  return vreinterpret_u32_p16(a);
}

// CHECK-LABEL: test_vreinterpret_u64_s8
uint64x1_t test_vreinterpret_u64_s8(int8x8_t a) {
  return vreinterpret_u64_s8(a);
}

// CHECK-LABEL: test_vreinterpret_u64_s16
uint64x1_t test_vreinterpret_u64_s16(int16x4_t a) {
  return vreinterpret_u64_s16(a);
}

// CHECK-LABEL: test_vreinterpret_u64_s32
uint64x1_t test_vreinterpret_u64_s32(int32x2_t a) {
  return vreinterpret_u64_s32(a);
}

// CHECK-LABEL: test_vreinterpret_u64_s64
uint64x1_t test_vreinterpret_u64_s64(int64x1_t a) {
  return vreinterpret_u64_s64(a);
}

// CHECK-LABEL: test_vreinterpret_u64_u8
uint64x1_t test_vreinterpret_u64_u8(uint8x8_t a) {
  return vreinterpret_u64_u8(a);
}

// CHECK-LABEL: test_vreinterpret_u64_u16
uint64x1_t test_vreinterpret_u64_u16(uint16x4_t a) {
  return vreinterpret_u64_u16(a);
}

// CHECK-LABEL: test_vreinterpret_u64_u32
uint64x1_t test_vreinterpret_u64_u32(uint32x2_t a) {
  return vreinterpret_u64_u32(a);
}

// CHECK-LABEL: test_vreinterpret_u64_f16
uint64x1_t test_vreinterpret_u64_f16(float16x4_t a) {
  return vreinterpret_u64_f16(a);
}

// CHECK-LABEL: test_vreinterpret_u64_f32
uint64x1_t test_vreinterpret_u64_f32(float32x2_t a) {
  return vreinterpret_u64_f32(a);
}

// CHECK-LABEL: test_vreinterpret_u64_p8
uint64x1_t test_vreinterpret_u64_p8(poly8x8_t a) {
  return vreinterpret_u64_p8(a);
}

// CHECK-LABEL: test_vreinterpret_u64_p16
uint64x1_t test_vreinterpret_u64_p16(poly16x4_t a) {
  return vreinterpret_u64_p16(a);
}

// CHECK-LABEL: test_vreinterpret_f16_s8
float16x4_t test_vreinterpret_f16_s8(int8x8_t a) {
  return vreinterpret_f16_s8(a);
}

// CHECK-LABEL: test_vreinterpret_f16_s16
float16x4_t test_vreinterpret_f16_s16(int16x4_t a) {
  return vreinterpret_f16_s16(a);
}

// CHECK-LABEL: test_vreinterpret_f16_s32
float16x4_t test_vreinterpret_f16_s32(int32x2_t a) {
  return vreinterpret_f16_s32(a);
}

// CHECK-LABEL: test_vreinterpret_f16_s64
float16x4_t test_vreinterpret_f16_s64(int64x1_t a) {
  return vreinterpret_f16_s64(a);
}

// CHECK-LABEL: test_vreinterpret_f16_u8
float16x4_t test_vreinterpret_f16_u8(uint8x8_t a) {
  return vreinterpret_f16_u8(a);
}

// CHECK-LABEL: test_vreinterpret_f16_u16
float16x4_t test_vreinterpret_f16_u16(uint16x4_t a) {
  return vreinterpret_f16_u16(a);
}

// CHECK-LABEL: test_vreinterpret_f16_u32
float16x4_t test_vreinterpret_f16_u32(uint32x2_t a) {
  return vreinterpret_f16_u32(a);
}

// CHECK-LABEL: test_vreinterpret_f16_u64
float16x4_t test_vreinterpret_f16_u64(uint64x1_t a) {
  return vreinterpret_f16_u64(a);
}

// CHECK-LABEL: test_vreinterpret_f16_f32
float16x4_t test_vreinterpret_f16_f32(float32x2_t a) {
  return vreinterpret_f16_f32(a);
}

// CHECK-LABEL: test_vreinterpret_f16_p8
float16x4_t test_vreinterpret_f16_p8(poly8x8_t a) {
  return vreinterpret_f16_p8(a);
}

// CHECK-LABEL: test_vreinterpret_f16_p16
float16x4_t test_vreinterpret_f16_p16(poly16x4_t a) {
  return vreinterpret_f16_p16(a);
}

// CHECK-LABEL: test_vreinterpret_f32_s8
float32x2_t test_vreinterpret_f32_s8(int8x8_t a) {
  return vreinterpret_f32_s8(a);
}

// CHECK-LABEL: test_vreinterpret_f32_s16
float32x2_t test_vreinterpret_f32_s16(int16x4_t a) {
  return vreinterpret_f32_s16(a);
}

// CHECK-LABEL: test_vreinterpret_f32_s32
float32x2_t test_vreinterpret_f32_s32(int32x2_t a) {
  return vreinterpret_f32_s32(a);
}

// CHECK-LABEL: test_vreinterpret_f32_s64
float32x2_t test_vreinterpret_f32_s64(int64x1_t a) {
  return vreinterpret_f32_s64(a);
}

// CHECK-LABEL: test_vreinterpret_f32_u8
float32x2_t test_vreinterpret_f32_u8(uint8x8_t a) {
  return vreinterpret_f32_u8(a);
}

// CHECK-LABEL: test_vreinterpret_f32_u16
float32x2_t test_vreinterpret_f32_u16(uint16x4_t a) {
  return vreinterpret_f32_u16(a);
}

// CHECK-LABEL: test_vreinterpret_f32_u32
float32x2_t test_vreinterpret_f32_u32(uint32x2_t a) {
  return vreinterpret_f32_u32(a);
}

// CHECK-LABEL: test_vreinterpret_f32_u64
float32x2_t test_vreinterpret_f32_u64(uint64x1_t a) {
  return vreinterpret_f32_u64(a);
}

// CHECK-LABEL: test_vreinterpret_f32_f16
float32x2_t test_vreinterpret_f32_f16(float16x4_t a) {
  return vreinterpret_f32_f16(a);
}

// CHECK-LABEL: test_vreinterpret_f32_p8
float32x2_t test_vreinterpret_f32_p8(poly8x8_t a) {
  return vreinterpret_f32_p8(a);
}

// CHECK-LABEL: test_vreinterpret_f32_p16
float32x2_t test_vreinterpret_f32_p16(poly16x4_t a) {
  return vreinterpret_f32_p16(a);
}

// CHECK-LABEL: test_vreinterpret_p8_s8
poly8x8_t test_vreinterpret_p8_s8(int8x8_t a) {
  return vreinterpret_p8_s8(a);
}

// CHECK-LABEL: test_vreinterpret_p8_s16
poly8x8_t test_vreinterpret_p8_s16(int16x4_t a) {
  return vreinterpret_p8_s16(a);
}

// CHECK-LABEL: test_vreinterpret_p8_s32
poly8x8_t test_vreinterpret_p8_s32(int32x2_t a) {
  return vreinterpret_p8_s32(a);
}

// CHECK-LABEL: test_vreinterpret_p8_s64
poly8x8_t test_vreinterpret_p8_s64(int64x1_t a) {
  return vreinterpret_p8_s64(a);
}

// CHECK-LABEL: test_vreinterpret_p8_u8
poly8x8_t test_vreinterpret_p8_u8(uint8x8_t a) {
  return vreinterpret_p8_u8(a);
}

// CHECK-LABEL: test_vreinterpret_p8_u16
poly8x8_t test_vreinterpret_p8_u16(uint16x4_t a) {
  return vreinterpret_p8_u16(a);
}

// CHECK-LABEL: test_vreinterpret_p8_u32
poly8x8_t test_vreinterpret_p8_u32(uint32x2_t a) {
  return vreinterpret_p8_u32(a);
}

// CHECK-LABEL: test_vreinterpret_p8_u64
poly8x8_t test_vreinterpret_p8_u64(uint64x1_t a) {
  return vreinterpret_p8_u64(a);
}

// CHECK-LABEL: test_vreinterpret_p8_f16
poly8x8_t test_vreinterpret_p8_f16(float16x4_t a) {
  return vreinterpret_p8_f16(a);
}

// CHECK-LABEL: test_vreinterpret_p8_f32
poly8x8_t test_vreinterpret_p8_f32(float32x2_t a) {
  return vreinterpret_p8_f32(a);
}

// CHECK-LABEL: test_vreinterpret_p8_p16
poly8x8_t test_vreinterpret_p8_p16(poly16x4_t a) {
  return vreinterpret_p8_p16(a);
}

// CHECK-LABEL: test_vreinterpret_p16_s8
poly16x4_t test_vreinterpret_p16_s8(int8x8_t a) {
  return vreinterpret_p16_s8(a);
}

// CHECK-LABEL: test_vreinterpret_p16_s16
poly16x4_t test_vreinterpret_p16_s16(int16x4_t a) {
  return vreinterpret_p16_s16(a);
}

// CHECK-LABEL: test_vreinterpret_p16_s32
poly16x4_t test_vreinterpret_p16_s32(int32x2_t a) {
  return vreinterpret_p16_s32(a);
}

// CHECK-LABEL: test_vreinterpret_p16_s64
poly16x4_t test_vreinterpret_p16_s64(int64x1_t a) {
  return vreinterpret_p16_s64(a);
}

// CHECK-LABEL: test_vreinterpret_p16_u8
poly16x4_t test_vreinterpret_p16_u8(uint8x8_t a) {
  return vreinterpret_p16_u8(a);
}

// CHECK-LABEL: test_vreinterpret_p16_u16
poly16x4_t test_vreinterpret_p16_u16(uint16x4_t a) {
  return vreinterpret_p16_u16(a);
}

// CHECK-LABEL: test_vreinterpret_p16_u32
poly16x4_t test_vreinterpret_p16_u32(uint32x2_t a) {
  return vreinterpret_p16_u32(a);
}

// CHECK-LABEL: test_vreinterpret_p16_u64
poly16x4_t test_vreinterpret_p16_u64(uint64x1_t a) {
  return vreinterpret_p16_u64(a);
}

// CHECK-LABEL: test_vreinterpret_p16_f16
poly16x4_t test_vreinterpret_p16_f16(float16x4_t a) {
  return vreinterpret_p16_f16(a);
}

// CHECK-LABEL: test_vreinterpret_p16_f32
poly16x4_t test_vreinterpret_p16_f32(float32x2_t a) {
  return vreinterpret_p16_f32(a);
}

// CHECK-LABEL: test_vreinterpret_p16_p8
poly16x4_t test_vreinterpret_p16_p8(poly8x8_t a) {
  return vreinterpret_p16_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_s16
int8x16_t test_vreinterpretq_s8_s16(int16x8_t a) {
  return vreinterpretq_s8_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_s32
int8x16_t test_vreinterpretq_s8_s32(int32x4_t a) {
  return vreinterpretq_s8_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_s64
int8x16_t test_vreinterpretq_s8_s64(int64x2_t a) {
  return vreinterpretq_s8_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_u8
int8x16_t test_vreinterpretq_s8_u8(uint8x16_t a) {
  return vreinterpretq_s8_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_u16
int8x16_t test_vreinterpretq_s8_u16(uint16x8_t a) {
  return vreinterpretq_s8_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_u32
int8x16_t test_vreinterpretq_s8_u32(uint32x4_t a) {
  return vreinterpretq_s8_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_u64
int8x16_t test_vreinterpretq_s8_u64(uint64x2_t a) {
  return vreinterpretq_s8_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_f16
int8x16_t test_vreinterpretq_s8_f16(float16x8_t a) {
  return vreinterpretq_s8_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_f32
int8x16_t test_vreinterpretq_s8_f32(float32x4_t a) {
  return vreinterpretq_s8_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_p8
int8x16_t test_vreinterpretq_s8_p8(poly8x16_t a) {
  return vreinterpretq_s8_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_p16
int8x16_t test_vreinterpretq_s8_p16(poly16x8_t a) {
  return vreinterpretq_s8_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_s8
int16x8_t test_vreinterpretq_s16_s8(int8x16_t a) {
  return vreinterpretq_s16_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_s32
int16x8_t test_vreinterpretq_s16_s32(int32x4_t a) {
  return vreinterpretq_s16_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_s64
int16x8_t test_vreinterpretq_s16_s64(int64x2_t a) {
  return vreinterpretq_s16_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_u8
int16x8_t test_vreinterpretq_s16_u8(uint8x16_t a) {
  return vreinterpretq_s16_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_u16
int16x8_t test_vreinterpretq_s16_u16(uint16x8_t a) {
  return vreinterpretq_s16_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_u32
int16x8_t test_vreinterpretq_s16_u32(uint32x4_t a) {
  return vreinterpretq_s16_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_u64
int16x8_t test_vreinterpretq_s16_u64(uint64x2_t a) {
  return vreinterpretq_s16_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_f16
int16x8_t test_vreinterpretq_s16_f16(float16x8_t a) {
  return vreinterpretq_s16_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_f32
int16x8_t test_vreinterpretq_s16_f32(float32x4_t a) {
  return vreinterpretq_s16_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_p8
int16x8_t test_vreinterpretq_s16_p8(poly8x16_t a) {
  return vreinterpretq_s16_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_p16
int16x8_t test_vreinterpretq_s16_p16(poly16x8_t a) {
  return vreinterpretq_s16_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_s8
int32x4_t test_vreinterpretq_s32_s8(int8x16_t a) {
  return vreinterpretq_s32_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_s16
int32x4_t test_vreinterpretq_s32_s16(int16x8_t a) {
  return vreinterpretq_s32_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_s64
int32x4_t test_vreinterpretq_s32_s64(int64x2_t a) {
  return vreinterpretq_s32_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_u8
int32x4_t test_vreinterpretq_s32_u8(uint8x16_t a) {
  return vreinterpretq_s32_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_u16
int32x4_t test_vreinterpretq_s32_u16(uint16x8_t a) {
  return vreinterpretq_s32_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_u32
int32x4_t test_vreinterpretq_s32_u32(uint32x4_t a) {
  return vreinterpretq_s32_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_u64
int32x4_t test_vreinterpretq_s32_u64(uint64x2_t a) {
  return vreinterpretq_s32_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_f16
int32x4_t test_vreinterpretq_s32_f16(float16x8_t a) {
  return vreinterpretq_s32_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_f32
int32x4_t test_vreinterpretq_s32_f32(float32x4_t a) {
  return vreinterpretq_s32_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_p8
int32x4_t test_vreinterpretq_s32_p8(poly8x16_t a) {
  return vreinterpretq_s32_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_p16
int32x4_t test_vreinterpretq_s32_p16(poly16x8_t a) {
  return vreinterpretq_s32_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_s8
int64x2_t test_vreinterpretq_s64_s8(int8x16_t a) {
  return vreinterpretq_s64_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_s16
int64x2_t test_vreinterpretq_s64_s16(int16x8_t a) {
  return vreinterpretq_s64_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_s32
int64x2_t test_vreinterpretq_s64_s32(int32x4_t a) {
  return vreinterpretq_s64_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_u8
int64x2_t test_vreinterpretq_s64_u8(uint8x16_t a) {
  return vreinterpretq_s64_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_u16
int64x2_t test_vreinterpretq_s64_u16(uint16x8_t a) {
  return vreinterpretq_s64_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_u32
int64x2_t test_vreinterpretq_s64_u32(uint32x4_t a) {
  return vreinterpretq_s64_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_u64
int64x2_t test_vreinterpretq_s64_u64(uint64x2_t a) {
  return vreinterpretq_s64_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_f16
int64x2_t test_vreinterpretq_s64_f16(float16x8_t a) {
  return vreinterpretq_s64_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_f32
int64x2_t test_vreinterpretq_s64_f32(float32x4_t a) {
  return vreinterpretq_s64_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_p8
int64x2_t test_vreinterpretq_s64_p8(poly8x16_t a) {
  return vreinterpretq_s64_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_p16
int64x2_t test_vreinterpretq_s64_p16(poly16x8_t a) {
  return vreinterpretq_s64_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_s8
uint8x16_t test_vreinterpretq_u8_s8(int8x16_t a) {
  return vreinterpretq_u8_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_s16
uint8x16_t test_vreinterpretq_u8_s16(int16x8_t a) {
  return vreinterpretq_u8_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_s32
uint8x16_t test_vreinterpretq_u8_s32(int32x4_t a) {
  return vreinterpretq_u8_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_s64
uint8x16_t test_vreinterpretq_u8_s64(int64x2_t a) {
  return vreinterpretq_u8_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_u16
uint8x16_t test_vreinterpretq_u8_u16(uint16x8_t a) {
  return vreinterpretq_u8_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_u32
uint8x16_t test_vreinterpretq_u8_u32(uint32x4_t a) {
  return vreinterpretq_u8_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_u64
uint8x16_t test_vreinterpretq_u8_u64(uint64x2_t a) {
  return vreinterpretq_u8_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_f16
uint8x16_t test_vreinterpretq_u8_f16(float16x8_t a) {
  return vreinterpretq_u8_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_f32
uint8x16_t test_vreinterpretq_u8_f32(float32x4_t a) {
  return vreinterpretq_u8_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_p8
uint8x16_t test_vreinterpretq_u8_p8(poly8x16_t a) {
  return vreinterpretq_u8_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_p16
uint8x16_t test_vreinterpretq_u8_p16(poly16x8_t a) {
  return vreinterpretq_u8_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_s8
uint16x8_t test_vreinterpretq_u16_s8(int8x16_t a) {
  return vreinterpretq_u16_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_s16
uint16x8_t test_vreinterpretq_u16_s16(int16x8_t a) {
  return vreinterpretq_u16_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_s32
uint16x8_t test_vreinterpretq_u16_s32(int32x4_t a) {
  return vreinterpretq_u16_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_s64
uint16x8_t test_vreinterpretq_u16_s64(int64x2_t a) {
  return vreinterpretq_u16_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_u8
uint16x8_t test_vreinterpretq_u16_u8(uint8x16_t a) {
  return vreinterpretq_u16_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_u32
uint16x8_t test_vreinterpretq_u16_u32(uint32x4_t a) {
  return vreinterpretq_u16_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_u64
uint16x8_t test_vreinterpretq_u16_u64(uint64x2_t a) {
  return vreinterpretq_u16_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_f16
uint16x8_t test_vreinterpretq_u16_f16(float16x8_t a) {
  return vreinterpretq_u16_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_f32
uint16x8_t test_vreinterpretq_u16_f32(float32x4_t a) {
  return vreinterpretq_u16_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_p8
uint16x8_t test_vreinterpretq_u16_p8(poly8x16_t a) {
  return vreinterpretq_u16_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_p16
uint16x8_t test_vreinterpretq_u16_p16(poly16x8_t a) {
  return vreinterpretq_u16_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_s8
uint32x4_t test_vreinterpretq_u32_s8(int8x16_t a) {
  return vreinterpretq_u32_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_s16
uint32x4_t test_vreinterpretq_u32_s16(int16x8_t a) {
  return vreinterpretq_u32_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_s32
uint32x4_t test_vreinterpretq_u32_s32(int32x4_t a) {
  return vreinterpretq_u32_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_s64
uint32x4_t test_vreinterpretq_u32_s64(int64x2_t a) {
  return vreinterpretq_u32_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_u8
uint32x4_t test_vreinterpretq_u32_u8(uint8x16_t a) {
  return vreinterpretq_u32_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_u16
uint32x4_t test_vreinterpretq_u32_u16(uint16x8_t a) {
  return vreinterpretq_u32_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_u64
uint32x4_t test_vreinterpretq_u32_u64(uint64x2_t a) {
  return vreinterpretq_u32_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_f16
uint32x4_t test_vreinterpretq_u32_f16(float16x8_t a) {
  return vreinterpretq_u32_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_f32
uint32x4_t test_vreinterpretq_u32_f32(float32x4_t a) {
  return vreinterpretq_u32_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_p8
uint32x4_t test_vreinterpretq_u32_p8(poly8x16_t a) {
  return vreinterpretq_u32_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_p16
uint32x4_t test_vreinterpretq_u32_p16(poly16x8_t a) {
  return vreinterpretq_u32_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_s8
uint64x2_t test_vreinterpretq_u64_s8(int8x16_t a) {
  return vreinterpretq_u64_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_s16
uint64x2_t test_vreinterpretq_u64_s16(int16x8_t a) {
  return vreinterpretq_u64_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_s32
uint64x2_t test_vreinterpretq_u64_s32(int32x4_t a) {
  return vreinterpretq_u64_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_s64
uint64x2_t test_vreinterpretq_u64_s64(int64x2_t a) {
  return vreinterpretq_u64_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_u8
uint64x2_t test_vreinterpretq_u64_u8(uint8x16_t a) {
  return vreinterpretq_u64_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_u16
uint64x2_t test_vreinterpretq_u64_u16(uint16x8_t a) {
  return vreinterpretq_u64_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_u32
uint64x2_t test_vreinterpretq_u64_u32(uint32x4_t a) {
  return vreinterpretq_u64_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_f16
uint64x2_t test_vreinterpretq_u64_f16(float16x8_t a) {
  return vreinterpretq_u64_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_f32
uint64x2_t test_vreinterpretq_u64_f32(float32x4_t a) {
  return vreinterpretq_u64_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_p8
uint64x2_t test_vreinterpretq_u64_p8(poly8x16_t a) {
  return vreinterpretq_u64_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_p16
uint64x2_t test_vreinterpretq_u64_p16(poly16x8_t a) {
  return vreinterpretq_u64_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_s8
float16x8_t test_vreinterpretq_f16_s8(int8x16_t a) {
  return vreinterpretq_f16_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_s16
float16x8_t test_vreinterpretq_f16_s16(int16x8_t a) {
  return vreinterpretq_f16_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_s32
float16x8_t test_vreinterpretq_f16_s32(int32x4_t a) {
  return vreinterpretq_f16_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_s64
float16x8_t test_vreinterpretq_f16_s64(int64x2_t a) {
  return vreinterpretq_f16_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_u8
float16x8_t test_vreinterpretq_f16_u8(uint8x16_t a) {
  return vreinterpretq_f16_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_u16
float16x8_t test_vreinterpretq_f16_u16(uint16x8_t a) {
  return vreinterpretq_f16_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_u32
float16x8_t test_vreinterpretq_f16_u32(uint32x4_t a) {
  return vreinterpretq_f16_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_u64
float16x8_t test_vreinterpretq_f16_u64(uint64x2_t a) {
  return vreinterpretq_f16_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_f32
float16x8_t test_vreinterpretq_f16_f32(float32x4_t a) {
  return vreinterpretq_f16_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_p8
float16x8_t test_vreinterpretq_f16_p8(poly8x16_t a) {
  return vreinterpretq_f16_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_f16_p16
float16x8_t test_vreinterpretq_f16_p16(poly16x8_t a) {
  return vreinterpretq_f16_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_s8
float32x4_t test_vreinterpretq_f32_s8(int8x16_t a) {
  return vreinterpretq_f32_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_s16
float32x4_t test_vreinterpretq_f32_s16(int16x8_t a) {
  return vreinterpretq_f32_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_s32
float32x4_t test_vreinterpretq_f32_s32(int32x4_t a) {
  return vreinterpretq_f32_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_s64
float32x4_t test_vreinterpretq_f32_s64(int64x2_t a) {
  return vreinterpretq_f32_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_u8
float32x4_t test_vreinterpretq_f32_u8(uint8x16_t a) {
  return vreinterpretq_f32_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_u16
float32x4_t test_vreinterpretq_f32_u16(uint16x8_t a) {
  return vreinterpretq_f32_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_u32
float32x4_t test_vreinterpretq_f32_u32(uint32x4_t a) {
  return vreinterpretq_f32_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_u64
float32x4_t test_vreinterpretq_f32_u64(uint64x2_t a) {
  return vreinterpretq_f32_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_f16
float32x4_t test_vreinterpretq_f32_f16(float16x8_t a) {
  return vreinterpretq_f32_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_p8
float32x4_t test_vreinterpretq_f32_p8(poly8x16_t a) {
  return vreinterpretq_f32_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_p16
float32x4_t test_vreinterpretq_f32_p16(poly16x8_t a) {
  return vreinterpretq_f32_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_s8
poly8x16_t test_vreinterpretq_p8_s8(int8x16_t a) {
  return vreinterpretq_p8_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_s16
poly8x16_t test_vreinterpretq_p8_s16(int16x8_t a) {
  return vreinterpretq_p8_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_s32
poly8x16_t test_vreinterpretq_p8_s32(int32x4_t a) {
  return vreinterpretq_p8_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_s64
poly8x16_t test_vreinterpretq_p8_s64(int64x2_t a) {
  return vreinterpretq_p8_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_u8
poly8x16_t test_vreinterpretq_p8_u8(uint8x16_t a) {
  return vreinterpretq_p8_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_u16
poly8x16_t test_vreinterpretq_p8_u16(uint16x8_t a) {
  return vreinterpretq_p8_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_u32
poly8x16_t test_vreinterpretq_p8_u32(uint32x4_t a) {
  return vreinterpretq_p8_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_u64
poly8x16_t test_vreinterpretq_p8_u64(uint64x2_t a) {
  return vreinterpretq_p8_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_f16
poly8x16_t test_vreinterpretq_p8_f16(float16x8_t a) {
  return vreinterpretq_p8_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_f32
poly8x16_t test_vreinterpretq_p8_f32(float32x4_t a) {
  return vreinterpretq_p8_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_p16
poly8x16_t test_vreinterpretq_p8_p16(poly16x8_t a) {
  return vreinterpretq_p8_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_s8
poly16x8_t test_vreinterpretq_p16_s8(int8x16_t a) {
  return vreinterpretq_p16_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_s16
poly16x8_t test_vreinterpretq_p16_s16(int16x8_t a) {
  return vreinterpretq_p16_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_s32
poly16x8_t test_vreinterpretq_p16_s32(int32x4_t a) {
  return vreinterpretq_p16_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_s64
poly16x8_t test_vreinterpretq_p16_s64(int64x2_t a) {
  return vreinterpretq_p16_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_u8
poly16x8_t test_vreinterpretq_p16_u8(uint8x16_t a) {
  return vreinterpretq_p16_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_u16
poly16x8_t test_vreinterpretq_p16_u16(uint16x8_t a) {
  return vreinterpretq_p16_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_u32
poly16x8_t test_vreinterpretq_p16_u32(uint32x4_t a) {
  return vreinterpretq_p16_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_u64
poly16x8_t test_vreinterpretq_p16_u64(uint64x2_t a) {
  return vreinterpretq_p16_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_f16
poly16x8_t test_vreinterpretq_p16_f16(float16x8_t a) {
  return vreinterpretq_p16_f16(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_f32
poly16x8_t test_vreinterpretq_p16_f32(float32x4_t a) {
  return vreinterpretq_p16_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_p8
poly16x8_t test_vreinterpretq_p16_p8(poly8x16_t a) {
  return vreinterpretq_p16_p8(a);
}


// CHECK-LABEL: test_vrev16_s8
// CHECK: vrev16.8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vrev16_s8(int8x8_t a) {
  return vrev16_s8(a);
}

// CHECK-LABEL: test_vrev16_u8
// CHECK: vrev16.8 d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vrev16_u8(uint8x8_t a) {
  return vrev16_u8(a);
}

// CHECK-LABEL: test_vrev16_p8
// CHECK: vrev16.8 d{{[0-9]+}}, d{{[0-9]+}}
poly8x8_t test_vrev16_p8(poly8x8_t a) {
  return vrev16_p8(a);
}

// CHECK-LABEL: test_vrev16q_s8
// CHECK: vrev16.8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vrev16q_s8(int8x16_t a) {
  return vrev16q_s8(a);
}

// CHECK-LABEL: test_vrev16q_u8
// CHECK: vrev16.8 q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vrev16q_u8(uint8x16_t a) {
  return vrev16q_u8(a);
}

// CHECK-LABEL: test_vrev16q_p8
// CHECK: vrev16.8 q{{[0-9]+}}, q{{[0-9]+}}
poly8x16_t test_vrev16q_p8(poly8x16_t a) {
  return vrev16q_p8(a);
}


// CHECK-LABEL: test_vrev32_s8
// CHECK: vrev32.8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vrev32_s8(int8x8_t a) {
  return vrev32_s8(a);
}

// CHECK-LABEL: test_vrev32_s16
// CHECK: vrev32.16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vrev32_s16(int16x4_t a) {
  return vrev32_s16(a);
}

// CHECK-LABEL: test_vrev32_u8
// CHECK: vrev32.8 d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vrev32_u8(uint8x8_t a) {
  return vrev32_u8(a);
}

// CHECK-LABEL: test_vrev32_u16
// CHECK: vrev32.16 d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vrev32_u16(uint16x4_t a) {
  return vrev32_u16(a);
}

// CHECK-LABEL: test_vrev32_p8
// CHECK: vrev32.8 d{{[0-9]+}}, d{{[0-9]+}}
poly8x8_t test_vrev32_p8(poly8x8_t a) {
  return vrev32_p8(a);
}

// CHECK-LABEL: test_vrev32_p16
// CHECK: vrev32.16 d{{[0-9]+}}, d{{[0-9]+}}
poly16x4_t test_vrev32_p16(poly16x4_t a) {
  return vrev32_p16(a);
}

// CHECK-LABEL: test_vrev32q_s8
// CHECK: vrev32.8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vrev32q_s8(int8x16_t a) {
  return vrev32q_s8(a);
}

// CHECK-LABEL: test_vrev32q_s16
// CHECK: vrev32.16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vrev32q_s16(int16x8_t a) {
  return vrev32q_s16(a);
}

// CHECK-LABEL: test_vrev32q_u8
// CHECK: vrev32.8 q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vrev32q_u8(uint8x16_t a) {
  return vrev32q_u8(a);
}

// CHECK-LABEL: test_vrev32q_u16
// CHECK: vrev32.16 q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vrev32q_u16(uint16x8_t a) {
  return vrev32q_u16(a);
}

// CHECK-LABEL: test_vrev32q_p8
// CHECK: vrev32.8 q{{[0-9]+}}, q{{[0-9]+}}
poly8x16_t test_vrev32q_p8(poly8x16_t a) {
  return vrev32q_p8(a);
}

// CHECK-LABEL: test_vrev32q_p16
// CHECK: vrev32.16 q{{[0-9]+}}, q{{[0-9]+}}
poly16x8_t test_vrev32q_p16(poly16x8_t a) {
  return vrev32q_p16(a);
}


// CHECK-LABEL: test_vrev64_s8
// CHECK: vrev64.8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vrev64_s8(int8x8_t a) {
  return vrev64_s8(a);
}

// CHECK-LABEL: test_vrev64_s16
// CHECK: vrev64.16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vrev64_s16(int16x4_t a) {
  return vrev64_s16(a);
}

// CHECK-LABEL: test_vrev64_s32
// CHECK: vrev64.32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vrev64_s32(int32x2_t a) {
  return vrev64_s32(a);
}

// CHECK-LABEL: test_vrev64_u8
// CHECK: vrev64.8 d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vrev64_u8(uint8x8_t a) {
  return vrev64_u8(a);
}

// CHECK-LABEL: test_vrev64_u16
// CHECK: vrev64.16 d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vrev64_u16(uint16x4_t a) {
  return vrev64_u16(a);
}

// CHECK-LABEL: test_vrev64_u32
// CHECK: vrev64.32 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vrev64_u32(uint32x2_t a) {
  return vrev64_u32(a);
}

// CHECK-LABEL: test_vrev64_p8
// CHECK: vrev64.8 d{{[0-9]+}}, d{{[0-9]+}}
poly8x8_t test_vrev64_p8(poly8x8_t a) {
  return vrev64_p8(a);
}

// CHECK-LABEL: test_vrev64_p16
// CHECK: vrev64.16 d{{[0-9]+}}, d{{[0-9]+}}
poly16x4_t test_vrev64_p16(poly16x4_t a) {
  return vrev64_p16(a);
}

// CHECK-LABEL: test_vrev64_f32
// CHECK: vrev64.32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vrev64_f32(float32x2_t a) {
  return vrev64_f32(a);
}

// CHECK-LABEL: test_vrev64q_s8
// CHECK: vrev64.8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vrev64q_s8(int8x16_t a) {
  return vrev64q_s8(a);
}

// CHECK-LABEL: test_vrev64q_s16
// CHECK: vrev64.16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vrev64q_s16(int16x8_t a) {
  return vrev64q_s16(a);
}

// CHECK-LABEL: test_vrev64q_s32
// CHECK: vrev64.32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vrev64q_s32(int32x4_t a) {
  return vrev64q_s32(a);
}

// CHECK-LABEL: test_vrev64q_u8
// CHECK: vrev64.8 q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vrev64q_u8(uint8x16_t a) {
  return vrev64q_u8(a);
}

// CHECK-LABEL: test_vrev64q_u16
// CHECK: vrev64.16 q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vrev64q_u16(uint16x8_t a) {
  return vrev64q_u16(a);
}

// CHECK-LABEL: test_vrev64q_u32
// CHECK: vrev64.32 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vrev64q_u32(uint32x4_t a) {
  return vrev64q_u32(a);
}

// CHECK-LABEL: test_vrev64q_p8
// CHECK: vrev64.8 q{{[0-9]+}}, q{{[0-9]+}}
poly8x16_t test_vrev64q_p8(poly8x16_t a) {
  return vrev64q_p8(a);
}

// CHECK-LABEL: test_vrev64q_p16
// CHECK: vrev64.16 q{{[0-9]+}}, q{{[0-9]+}}
poly16x8_t test_vrev64q_p16(poly16x8_t a) {
  return vrev64q_p16(a);
}

// CHECK-LABEL: test_vrev64q_f32
// CHECK: vrev64.32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vrev64q_f32(float32x4_t a) {
  return vrev64q_f32(a);
}


// CHECK-LABEL: test_vrhadd_s8
// CHECK: vrhadd.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vrhadd_s8(int8x8_t a, int8x8_t b) {
  return vrhadd_s8(a, b);
}

// CHECK-LABEL: test_vrhadd_s16
// CHECK: vrhadd.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vrhadd_s16(int16x4_t a, int16x4_t b) {
  return vrhadd_s16(a, b);
}

// CHECK-LABEL: test_vrhadd_s32
// CHECK: vrhadd.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vrhadd_s32(int32x2_t a, int32x2_t b) {
  return vrhadd_s32(a, b);
}

// CHECK-LABEL: test_vrhadd_u8
// CHECK: vrhadd.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vrhadd_u8(uint8x8_t a, uint8x8_t b) {
  return vrhadd_u8(a, b);
}

// CHECK-LABEL: test_vrhadd_u16
// CHECK: vrhadd.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vrhadd_u16(uint16x4_t a, uint16x4_t b) {
  return vrhadd_u16(a, b);
}

// CHECK-LABEL: test_vrhadd_u32
// CHECK: vrhadd.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vrhadd_u32(uint32x2_t a, uint32x2_t b) {
  return vrhadd_u32(a, b);
}

// CHECK-LABEL: test_vrhaddq_s8
// CHECK: vrhadd.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vrhaddq_s8(int8x16_t a, int8x16_t b) {
  return vrhaddq_s8(a, b);
}

// CHECK-LABEL: test_vrhaddq_s16
// CHECK: vrhadd.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vrhaddq_s16(int16x8_t a, int16x8_t b) {
  return vrhaddq_s16(a, b);
}

// CHECK-LABEL: test_vrhaddq_s32
// CHECK: vrhadd.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vrhaddq_s32(int32x4_t a, int32x4_t b) {
  return vrhaddq_s32(a, b);
}

// CHECK-LABEL: test_vrhaddq_u8
// CHECK: vrhadd.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vrhaddq_u8(uint8x16_t a, uint8x16_t b) {
  return vrhaddq_u8(a, b);
}

// CHECK-LABEL: test_vrhaddq_u16
// CHECK: vrhadd.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vrhaddq_u16(uint16x8_t a, uint16x8_t b) {
  return vrhaddq_u16(a, b);
}

// CHECK-LABEL: test_vrhaddq_u32
// CHECK: vrhadd.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vrhaddq_u32(uint32x4_t a, uint32x4_t b) {
  return vrhaddq_u32(a, b);
}


// CHECK-LABEL: test_vrshl_s8
// CHECK: vrshl.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vrshl_s8(int8x8_t a, int8x8_t b) {
  return vrshl_s8(a, b);
}

// CHECK-LABEL: test_vrshl_s16
// CHECK: vrshl.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vrshl_s16(int16x4_t a, int16x4_t b) {
  return vrshl_s16(a, b);
}

// CHECK-LABEL: test_vrshl_s32
// CHECK: vrshl.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vrshl_s32(int32x2_t a, int32x2_t b) {
  return vrshl_s32(a, b);
}

// CHECK-LABEL: test_vrshl_s64
// CHECK: vrshl.s64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vrshl_s64(int64x1_t a, int64x1_t b) {
  return vrshl_s64(a, b);
}

// CHECK-LABEL: test_vrshl_u8
// CHECK: vrshl.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vrshl_u8(uint8x8_t a, int8x8_t b) {
  return vrshl_u8(a, b);
}

// CHECK-LABEL: test_vrshl_u16
// CHECK: vrshl.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vrshl_u16(uint16x4_t a, int16x4_t b) {
  return vrshl_u16(a, b);
}

// CHECK-LABEL: test_vrshl_u32
// CHECK: vrshl.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vrshl_u32(uint32x2_t a, int32x2_t b) {
  return vrshl_u32(a, b);
}

// CHECK-LABEL: test_vrshl_u64
// CHECK: vrshl.u64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vrshl_u64(uint64x1_t a, int64x1_t b) {
  return vrshl_u64(a, b);
}

// CHECK-LABEL: test_vrshlq_s8
// CHECK: vrshl.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vrshlq_s8(int8x16_t a, int8x16_t b) {
  return vrshlq_s8(a, b);
}

// CHECK-LABEL: test_vrshlq_s16
// CHECK: vrshl.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vrshlq_s16(int16x8_t a, int16x8_t b) {
  return vrshlq_s16(a, b);
}

// CHECK-LABEL: test_vrshlq_s32
// CHECK: vrshl.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vrshlq_s32(int32x4_t a, int32x4_t b) {
  return vrshlq_s32(a, b);
}

// CHECK-LABEL: test_vrshlq_s64
// CHECK: vrshl.s64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vrshlq_s64(int64x2_t a, int64x2_t b) {
  return vrshlq_s64(a, b);
}

// CHECK-LABEL: test_vrshlq_u8
// CHECK: vrshl.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vrshlq_u8(uint8x16_t a, int8x16_t b) {
  return vrshlq_u8(a, b);
}

// CHECK-LABEL: test_vrshlq_u16
// CHECK: vrshl.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vrshlq_u16(uint16x8_t a, int16x8_t b) {
  return vrshlq_u16(a, b);
}

// CHECK-LABEL: test_vrshlq_u32
// CHECK: vrshl.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vrshlq_u32(uint32x4_t a, int32x4_t b) {
  return vrshlq_u32(a, b);
}

// CHECK-LABEL: test_vrshlq_u64
// CHECK: vrshl.u64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vrshlq_u64(uint64x2_t a, int64x2_t b) {
  return vrshlq_u64(a, b);
}


// CHECK-LABEL: test_vrshrn_n_s16
// CHECK: vrshrn.i16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vrshrn_n_s16(int16x8_t a) {
  return vrshrn_n_s16(a, 1);
}

// CHECK-LABEL: test_vrshrn_n_s32
// CHECK: vrshrn.i32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vrshrn_n_s32(int32x4_t a) {
  return vrshrn_n_s32(a, 1);
}

// CHECK-LABEL: test_vrshrn_n_s64
// CHECK: vrshrn.i64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vrshrn_n_s64(int64x2_t a) {
  return vrshrn_n_s64(a, 1);
}

// CHECK-LABEL: test_vrshrn_n_u16
// CHECK: vrshrn.i16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vrshrn_n_u16(uint16x8_t a) {
  return vrshrn_n_u16(a, 1);
}

// CHECK-LABEL: test_vrshrn_n_u32
// CHECK: vrshrn.i32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vrshrn_n_u32(uint32x4_t a) {
  return vrshrn_n_u32(a, 1);
}

// CHECK-LABEL: test_vrshrn_n_u64
// CHECK: vrshrn.i64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vrshrn_n_u64(uint64x2_t a) {
  return vrshrn_n_u64(a, 1);
}


// CHECK-LABEL: test_vrshr_n_s8
// CHECK: vrshr.s8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vrshr_n_s8(int8x8_t a) {
  return vrshr_n_s8(a, 1);
}

// CHECK-LABEL: test_vrshr_n_s16
// CHECK: vrshr.s16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vrshr_n_s16(int16x4_t a) {
  return vrshr_n_s16(a, 1);
}

// CHECK-LABEL: test_vrshr_n_s32
// CHECK: vrshr.s32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vrshr_n_s32(int32x2_t a) {
  return vrshr_n_s32(a, 1);
}

// CHECK-LABEL: test_vrshr_n_s64
// CHECK: vrshr.s64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x1_t test_vrshr_n_s64(int64x1_t a) {
  return vrshr_n_s64(a, 1);
}

// CHECK-LABEL: test_vrshr_n_u8
// CHECK: vrshr.u8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vrshr_n_u8(uint8x8_t a) {
  return vrshr_n_u8(a, 1);
}

// CHECK-LABEL: test_vrshr_n_u16
// CHECK: vrshr.u16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vrshr_n_u16(uint16x4_t a) {
  return vrshr_n_u16(a, 1);
}

// CHECK-LABEL: test_vrshr_n_u32
// CHECK: vrshr.u32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vrshr_n_u32(uint32x2_t a) {
  return vrshr_n_u32(a, 1);
}

// CHECK-LABEL: test_vrshr_n_u64
// CHECK: vrshr.u64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vrshr_n_u64(uint64x1_t a) {
  return vrshr_n_u64(a, 1);
}

// CHECK-LABEL: test_vrshrq_n_s8
// CHECK: vrshr.s8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vrshrq_n_s8(int8x16_t a) {
  return vrshrq_n_s8(a, 1);
}

// CHECK-LABEL: test_vrshrq_n_s16
// CHECK: vrshr.s16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vrshrq_n_s16(int16x8_t a) {
  return vrshrq_n_s16(a, 1);
}

// CHECK-LABEL: test_vrshrq_n_s32
// CHECK: vrshr.s32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vrshrq_n_s32(int32x4_t a) {
  return vrshrq_n_s32(a, 1);
}

// CHECK-LABEL: test_vrshrq_n_s64
// CHECK: vrshr.s64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vrshrq_n_s64(int64x2_t a) {
  return vrshrq_n_s64(a, 1);
}

// CHECK-LABEL: test_vrshrq_n_u8
// CHECK: vrshr.u8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vrshrq_n_u8(uint8x16_t a) {
  return vrshrq_n_u8(a, 1);
}

// CHECK-LABEL: test_vrshrq_n_u16
// CHECK: vrshr.u16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vrshrq_n_u16(uint16x8_t a) {
  return vrshrq_n_u16(a, 1);
}

// CHECK-LABEL: test_vrshrq_n_u32
// CHECK: vrshr.u32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vrshrq_n_u32(uint32x4_t a) {
  return vrshrq_n_u32(a, 1);
}

// CHECK-LABEL: test_vrshrq_n_u64
// CHECK: vrshr.u64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vrshrq_n_u64(uint64x2_t a) {
  return vrshrq_n_u64(a, 1);
}


// CHECK-LABEL: test_vrsqrte_f32
// CHECK: vrsqrte.f32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vrsqrte_f32(float32x2_t a) {
  return vrsqrte_f32(a);
}

// CHECK-LABEL: test_vrsqrte_u32
// CHECK: vrsqrte.u32 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vrsqrte_u32(uint32x2_t a) {
  return vrsqrte_u32(a);
}

// CHECK-LABEL: test_vrsqrteq_f32
// CHECK: vrsqrte.f32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vrsqrteq_f32(float32x4_t a) {
  return vrsqrteq_f32(a);
}

// CHECK-LABEL: test_vrsqrteq_u32
// CHECK: vrsqrte.u32 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vrsqrteq_u32(uint32x4_t a) {
  return vrsqrteq_u32(a);
}


// CHECK-LABEL: test_vrsqrts_f32
// CHECK: vrsqrts.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vrsqrts_f32(float32x2_t a, float32x2_t b) {
  return vrsqrts_f32(a, b);
}

// CHECK-LABEL: test_vrsqrtsq_f32
// CHECK: vrsqrts.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vrsqrtsq_f32(float32x4_t a, float32x4_t b) {
  return vrsqrtsq_f32(a, b);
}


// CHECK-LABEL: test_vrsra_n_s8
// CHECK: vrsra.s8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vrsra_n_s8(int8x8_t a, int8x8_t b) {
  return vrsra_n_s8(a, b, 1);
}

// CHECK-LABEL: test_vrsra_n_s16
// CHECK: vrsra.s16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vrsra_n_s16(int16x4_t a, int16x4_t b) {
  return vrsra_n_s16(a, b, 1);
}

// CHECK-LABEL: test_vrsra_n_s32
// CHECK: vrsra.s32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vrsra_n_s32(int32x2_t a, int32x2_t b) {
  return vrsra_n_s32(a, b, 1);
}

// CHECK-LABEL: test_vrsra_n_s64
// CHECK: vrsra.s64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x1_t test_vrsra_n_s64(int64x1_t a, int64x1_t b) {
  return vrsra_n_s64(a, b, 1);
}

// CHECK-LABEL: test_vrsra_n_u8
// CHECK: vrsra.u8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vrsra_n_u8(uint8x8_t a, uint8x8_t b) {
  return vrsra_n_u8(a, b, 1);
}

// CHECK-LABEL: test_vrsra_n_u16
// CHECK: vrsra.u16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vrsra_n_u16(uint16x4_t a, uint16x4_t b) {
  return vrsra_n_u16(a, b, 1);
}

// CHECK-LABEL: test_vrsra_n_u32
// CHECK: vrsra.u32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vrsra_n_u32(uint32x2_t a, uint32x2_t b) {
  return vrsra_n_u32(a, b, 1);
}

// CHECK-LABEL: test_vrsra_n_u64
// CHECK: vrsra.u64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vrsra_n_u64(uint64x1_t a, uint64x1_t b) {
  return vrsra_n_u64(a, b, 1);
}

// CHECK-LABEL: test_vrsraq_n_s8
// CHECK: vrsra.s8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vrsraq_n_s8(int8x16_t a, int8x16_t b) {
  return vrsraq_n_s8(a, b, 1);
}

// CHECK-LABEL: test_vrsraq_n_s16
// CHECK: vrsra.s16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vrsraq_n_s16(int16x8_t a, int16x8_t b) {
  return vrsraq_n_s16(a, b, 1);
}

// CHECK-LABEL: test_vrsraq_n_s32
// CHECK: vrsra.s32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vrsraq_n_s32(int32x4_t a, int32x4_t b) {
  return vrsraq_n_s32(a, b, 1);
}

// CHECK-LABEL: test_vrsraq_n_s64
// CHECK: vrsra.s64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vrsraq_n_s64(int64x2_t a, int64x2_t b) {
  return vrsraq_n_s64(a, b, 1);
}

// CHECK-LABEL: test_vrsraq_n_u8
// CHECK: vrsra.u8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vrsraq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vrsraq_n_u8(a, b, 1);
}

// CHECK-LABEL: test_vrsraq_n_u16
// CHECK: vrsra.u16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vrsraq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vrsraq_n_u16(a, b, 1);
}

// CHECK-LABEL: test_vrsraq_n_u32
// CHECK: vrsra.u32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vrsraq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vrsraq_n_u32(a, b, 1);
}

// CHECK-LABEL: test_vrsraq_n_u64
// CHECK: vrsra.u64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vrsraq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vrsraq_n_u64(a, b, 1);
}


// CHECK-LABEL: test_vrsubhn_s16
// CHECK: vrsubhn.i16 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x8_t test_vrsubhn_s16(int16x8_t a, int16x8_t b) {
  return vrsubhn_s16(a, b);
}

// CHECK-LABEL: test_vrsubhn_s32
// CHECK: vrsubhn.i32 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x4_t test_vrsubhn_s32(int32x4_t a, int32x4_t b) {
  return vrsubhn_s32(a, b);
}

// CHECK-LABEL: test_vrsubhn_s64
// CHECK: vrsubhn.i64 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x2_t test_vrsubhn_s64(int64x2_t a, int64x2_t b) {
  return vrsubhn_s64(a, b);
}

// CHECK-LABEL: test_vrsubhn_u16
// CHECK: vrsubhn.i16 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x8_t test_vrsubhn_u16(uint16x8_t a, uint16x8_t b) {
  return vrsubhn_u16(a, b);
}

// CHECK-LABEL: test_vrsubhn_u32
// CHECK: vrsubhn.i32 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x4_t test_vrsubhn_u32(uint32x4_t a, uint32x4_t b) {
  return vrsubhn_u32(a, b);
}

// CHECK-LABEL: test_vrsubhn_u64
// CHECK: vrsubhn.i64 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x2_t test_vrsubhn_u64(uint64x2_t a, uint64x2_t b) {
  return vrsubhn_u64(a, b);
}


// CHECK-LABEL: test_vset_lane_u8
// CHECK: vmov 
uint8x8_t test_vset_lane_u8(uint8_t a, uint8x8_t b) {
  return vset_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vset_lane_u16
// CHECK: vmov 
uint16x4_t test_vset_lane_u16(uint16_t a, uint16x4_t b) {
  return vset_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vset_lane_u32
// CHECK: mov 
uint32x2_t test_vset_lane_u32(uint32_t a, uint32x2_t b) {
  return vset_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vset_lane_s8
// CHECK: vmov 
int8x8_t test_vset_lane_s8(int8_t a, int8x8_t b) {
  return vset_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vset_lane_s16
// CHECK: vmov 
int16x4_t test_vset_lane_s16(int16_t a, int16x4_t b) {
  return vset_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vset_lane_s32
// CHECK: mov 
int32x2_t test_vset_lane_s32(int32_t a, int32x2_t b) {
  return vset_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vset_lane_p8
// CHECK: vmov 
poly8x8_t test_vset_lane_p8(poly8_t a, poly8x8_t b) {
  return vset_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vset_lane_p16
// CHECK: vmov 
poly16x4_t test_vset_lane_p16(poly16_t a, poly16x4_t b) {
  return vset_lane_p16(a, b, 3);
}

// CHECK-LABEL: test_vset_lane_f32
// CHECK: mov 
float32x2_t test_vset_lane_f32(float32_t a, float32x2_t b) {
  return vset_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vset_lane_f16
// CHECK: mov 
float16x4_t test_vset_lane_f16(float16_t *a, float16x4_t b) {
  return vset_lane_f16(*a, b, 1);
}

// CHECK-LABEL: test_vsetq_lane_u8
// CHECK: vmov 
uint8x16_t test_vsetq_lane_u8(uint8_t a, uint8x16_t b) {
  return vsetq_lane_u8(a, b, 15);
}

// CHECK-LABEL: test_vsetq_lane_u16
// CHECK: vmov 
uint16x8_t test_vsetq_lane_u16(uint16_t a, uint16x8_t b) {
  return vsetq_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vsetq_lane_u32
// CHECK: vmov 
uint32x4_t test_vsetq_lane_u32(uint32_t a, uint32x4_t b) {
  return vsetq_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vsetq_lane_s8
// CHECK: vmov 
int8x16_t test_vsetq_lane_s8(int8_t a, int8x16_t b) {
  return vsetq_lane_s8(a, b, 15);
}

// CHECK-LABEL: test_vsetq_lane_s16
// CHECK: vmov 
int16x8_t test_vsetq_lane_s16(int16_t a, int16x8_t b) {
  return vsetq_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vsetq_lane_s32
// CHECK: vmov 
int32x4_t test_vsetq_lane_s32(int32_t a, int32x4_t b) {
  return vsetq_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vsetq_lane_p8
// CHECK: vmov 
poly8x16_t test_vsetq_lane_p8(poly8_t a, poly8x16_t b) {
  return vsetq_lane_p8(a, b, 15);
}

// CHECK-LABEL: test_vsetq_lane_p16
// CHECK: vmov 
poly16x8_t test_vsetq_lane_p16(poly16_t a, poly16x8_t b) {
  return vsetq_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vsetq_lane_f32
// CHECK: vmov 
float32x4_t test_vsetq_lane_f32(float32_t a, float32x4_t b) {
  return vsetq_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vsetq_lane_f16
// CHECK: vmov 
float16x8_t test_vsetq_lane_f16(float16_t *a, float16x8_t b) {
  return vsetq_lane_f16(*a, b, 3);
}

// CHECK-LABEL: test_vset_lane_s64
// The optimizer is able to get rid of all moves now.
int64x1_t test_vset_lane_s64(int64_t a, int64x1_t b) {
  return vset_lane_s64(a, b, 0);
}

// CHECK-LABEL: test_vset_lane_u64
// The optimizer is able to get rid of all moves now.
uint64x1_t test_vset_lane_u64(uint64_t a, uint64x1_t b) {
  return vset_lane_u64(a, b, 0);
}

// CHECK-LABEL: test_vsetq_lane_s64
// CHECK: vmov 
int64x2_t test_vsetq_lane_s64(int64_t a, int64x2_t b) {
  return vsetq_lane_s64(a, b, 1);
}

// CHECK-LABEL: test_vsetq_lane_u64
// CHECK: vmov 
uint64x2_t test_vsetq_lane_u64(uint64_t a, uint64x2_t b) {
  return vsetq_lane_u64(a, b, 1);
}


// CHECK-LABEL: test_vshl_s8
// CHECK: vshl.s8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vshl_s8(int8x8_t a, int8x8_t b) {
  return vshl_s8(a, b);
}

// CHECK-LABEL: test_vshl_s16
// CHECK: vshl.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vshl_s16(int16x4_t a, int16x4_t b) {
  return vshl_s16(a, b);
}

// CHECK-LABEL: test_vshl_s32
// CHECK: vshl.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vshl_s32(int32x2_t a, int32x2_t b) {
  return vshl_s32(a, b);
}

// CHECK-LABEL: test_vshl_s64
// CHECK: vshl.s64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vshl_s64(int64x1_t a, int64x1_t b) {
  return vshl_s64(a, b);
}

// CHECK-LABEL: test_vshl_u8
// CHECK: vshl.u8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vshl_u8(uint8x8_t a, int8x8_t b) {
  return vshl_u8(a, b);
}

// CHECK-LABEL: test_vshl_u16
// CHECK: vshl.u16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vshl_u16(uint16x4_t a, int16x4_t b) {
  return vshl_u16(a, b);
}

// CHECK-LABEL: test_vshl_u32
// CHECK: vshl.u32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vshl_u32(uint32x2_t a, int32x2_t b) {
  return vshl_u32(a, b);
}

// CHECK-LABEL: test_vshl_u64
// CHECK: vshl.u64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vshl_u64(uint64x1_t a, int64x1_t b) {
  return vshl_u64(a, b);
}

// CHECK-LABEL: test_vshlq_s8
// CHECK: vshl.s8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vshlq_s8(int8x16_t a, int8x16_t b) {
  return vshlq_s8(a, b);
}

// CHECK-LABEL: test_vshlq_s16
// CHECK: vshl.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vshlq_s16(int16x8_t a, int16x8_t b) {
  return vshlq_s16(a, b);
}

// CHECK-LABEL: test_vshlq_s32
// CHECK: vshl.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vshlq_s32(int32x4_t a, int32x4_t b) {
  return vshlq_s32(a, b);
}

// CHECK-LABEL: test_vshlq_s64
// CHECK: vshl.s64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vshlq_s64(int64x2_t a, int64x2_t b) {
  return vshlq_s64(a, b);
}

// CHECK-LABEL: test_vshlq_u8
// CHECK: vshl.u8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vshlq_u8(uint8x16_t a, int8x16_t b) {
  return vshlq_u8(a, b);
}

// CHECK-LABEL: test_vshlq_u16
// CHECK: vshl.u16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vshlq_u16(uint16x8_t a, int16x8_t b) {
  return vshlq_u16(a, b);
}

// CHECK-LABEL: test_vshlq_u32
// CHECK: vshl.u32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vshlq_u32(uint32x4_t a, int32x4_t b) {
  return vshlq_u32(a, b);
}

// CHECK-LABEL: test_vshlq_u64
// CHECK: vshl.u64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vshlq_u64(uint64x2_t a, int64x2_t b) {
  return vshlq_u64(a, b);
}


// CHECK-LABEL: test_vshll_n_s8
// CHECK: vshll.s8 q{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vshll_n_s8(int8x8_t a) {
  return vshll_n_s8(a, 1);
}

// CHECK-LABEL: test_vshll_n_s16
// CHECK: vshll.s16 q{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vshll_n_s16(int16x4_t a) {
  return vshll_n_s16(a, 1);
}

// CHECK-LABEL: test_vshll_n_s32
// CHECK: vshll.s32 q{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vshll_n_s32(int32x2_t a) {
  return vshll_n_s32(a, 1);
}

// CHECK-LABEL: test_vshll_n_u8
// CHECK: vshll.u8 q{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vshll_n_u8(uint8x8_t a) {
  return vshll_n_u8(a, 1);
}

// CHECK-LABEL: test_vshll_n_u16
// CHECK: vshll.u16 q{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vshll_n_u16(uint16x4_t a) {
  return vshll_n_u16(a, 1);
}

// CHECK-LABEL: test_vshll_n_u32
// CHECK: vshll.u32 q{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vshll_n_u32(uint32x2_t a) {
  return vshll_n_u32(a, 1);
}


// CHECK-LABEL: test_vshl_n_s8
// CHECK: vshl.i8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vshl_n_s8(int8x8_t a) {
  return vshl_n_s8(a, 1);
}

// CHECK-LABEL: test_vshl_n_s16
// CHECK: vshl.i16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vshl_n_s16(int16x4_t a) {
  return vshl_n_s16(a, 1);
}

// CHECK-LABEL: test_vshl_n_s32
// CHECK: vshl.i32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vshl_n_s32(int32x2_t a) {
  return vshl_n_s32(a, 1);
}

// CHECK-LABEL: test_vshl_n_s64
// CHECK: vshl.i64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x1_t test_vshl_n_s64(int64x1_t a) {
  return vshl_n_s64(a, 1);
}

// CHECK-LABEL: test_vshl_n_u8
// CHECK: vshl.i8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vshl_n_u8(uint8x8_t a) {
  return vshl_n_u8(a, 1);
}

// CHECK-LABEL: test_vshl_n_u16
// CHECK: vshl.i16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vshl_n_u16(uint16x4_t a) {
  return vshl_n_u16(a, 1);
}

// CHECK-LABEL: test_vshl_n_u32
// CHECK: vshl.i32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vshl_n_u32(uint32x2_t a) {
  return vshl_n_u32(a, 1);
}

// CHECK-LABEL: test_vshl_n_u64
// CHECK: vshl.i64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vshl_n_u64(uint64x1_t a) {
  return vshl_n_u64(a, 1);
}

// CHECK-LABEL: test_vshlq_n_s8
// CHECK: vshl.i8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vshlq_n_s8(int8x16_t a) {
  return vshlq_n_s8(a, 1);
}

// CHECK-LABEL: test_vshlq_n_s16
// CHECK: vshl.i16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vshlq_n_s16(int16x8_t a) {
  return vshlq_n_s16(a, 1);
}

// CHECK-LABEL: test_vshlq_n_s32
// CHECK: vshl.i32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vshlq_n_s32(int32x4_t a) {
  return vshlq_n_s32(a, 1);
}

// CHECK-LABEL: test_vshlq_n_s64
// CHECK: vshl.i64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vshlq_n_s64(int64x2_t a) {
  return vshlq_n_s64(a, 1);
}

// CHECK-LABEL: test_vshlq_n_u8
// CHECK: vshl.i8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vshlq_n_u8(uint8x16_t a) {
  return vshlq_n_u8(a, 1);
}

// CHECK-LABEL: test_vshlq_n_u16
// CHECK: vshl.i16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vshlq_n_u16(uint16x8_t a) {
  return vshlq_n_u16(a, 1);
}

// CHECK-LABEL: test_vshlq_n_u32
// CHECK: vshl.i32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vshlq_n_u32(uint32x4_t a) {
  return vshlq_n_u32(a, 1);
}

// CHECK-LABEL: test_vshlq_n_u64
// CHECK: vshl.i64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vshlq_n_u64(uint64x2_t a) {
  return vshlq_n_u64(a, 1);
}


// CHECK-LABEL: test_vshrn_n_s16
// CHECK: vshrn.i16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vshrn_n_s16(int16x8_t a) {
  return vshrn_n_s16(a, 1);
}

// CHECK-LABEL: test_vshrn_n_s32
// CHECK: vshrn.i32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vshrn_n_s32(int32x4_t a) {
  return vshrn_n_s32(a, 1);
}

// CHECK-LABEL: test_vshrn_n_s64
// CHECK: vshrn.i64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vshrn_n_s64(int64x2_t a) {
  return vshrn_n_s64(a, 1);
}

// CHECK-LABEL: test_vshrn_n_u16
// CHECK: vshrn.i16 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vshrn_n_u16(uint16x8_t a) {
  return vshrn_n_u16(a, 1);
}

// CHECK-LABEL: test_vshrn_n_u32
// CHECK: vshrn.i32 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vshrn_n_u32(uint32x4_t a) {
  return vshrn_n_u32(a, 1);
}

// CHECK-LABEL: test_vshrn_n_u64
// CHECK: vshrn.i64 d{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vshrn_n_u64(uint64x2_t a) {
  return vshrn_n_u64(a, 1);
}


// CHECK-LABEL: test_vshr_n_s8
// CHECK: vshr.s8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vshr_n_s8(int8x8_t a) {
  return vshr_n_s8(a, 1);
}

// CHECK-LABEL: test_vshr_n_s16
// CHECK: vshr.s16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vshr_n_s16(int16x4_t a) {
  return vshr_n_s16(a, 1);
}

// CHECK-LABEL: test_vshr_n_s32
// CHECK: vshr.s32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vshr_n_s32(int32x2_t a) {
  return vshr_n_s32(a, 1);
}

// CHECK-LABEL: test_vshr_n_s64
// CHECK: vshr.s64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x1_t test_vshr_n_s64(int64x1_t a) {
  return vshr_n_s64(a, 1);
}

// CHECK-LABEL: test_vshr_n_u8
// CHECK: vshr.u8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vshr_n_u8(uint8x8_t a) {
  return vshr_n_u8(a, 1);
}

// CHECK-LABEL: test_vshr_n_u16
// CHECK: vshr.u16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vshr_n_u16(uint16x4_t a) {
  return vshr_n_u16(a, 1);
}

// CHECK-LABEL: test_vshr_n_u32
// CHECK: vshr.u32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vshr_n_u32(uint32x2_t a) {
  return vshr_n_u32(a, 1);
}

// CHECK-LABEL: test_vshr_n_u64
// CHECK: vshr.u64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vshr_n_u64(uint64x1_t a) {
  return vshr_n_u64(a, 1);
}

// CHECK-LABEL: test_vshrq_n_s8
// CHECK: vshr.s8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vshrq_n_s8(int8x16_t a) {
  return vshrq_n_s8(a, 1);
}

// CHECK-LABEL: test_vshrq_n_s16
// CHECK: vshr.s16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vshrq_n_s16(int16x8_t a) {
  return vshrq_n_s16(a, 1);
}

// CHECK-LABEL: test_vshrq_n_s32
// CHECK: vshr.s32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vshrq_n_s32(int32x4_t a) {
  return vshrq_n_s32(a, 1);
}

// CHECK-LABEL: test_vshrq_n_s64
// CHECK: vshr.s64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vshrq_n_s64(int64x2_t a) {
  return vshrq_n_s64(a, 1);
}

// CHECK-LABEL: test_vshrq_n_u8
// CHECK: vshr.u8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vshrq_n_u8(uint8x16_t a) {
  return vshrq_n_u8(a, 1);
}

// CHECK-LABEL: test_vshrq_n_u16
// CHECK: vshr.u16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vshrq_n_u16(uint16x8_t a) {
  return vshrq_n_u16(a, 1);
}

// CHECK-LABEL: test_vshrq_n_u32
// CHECK: vshr.u32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vshrq_n_u32(uint32x4_t a) {
  return vshrq_n_u32(a, 1);
}

// CHECK-LABEL: test_vshrq_n_u64
// CHECK: vshr.u64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vshrq_n_u64(uint64x2_t a) {
  return vshrq_n_u64(a, 1);
}


// CHECK-LABEL: test_vsli_n_s8
// CHECK: vsli.8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vsli_n_s8(int8x8_t a, int8x8_t b) {
  return vsli_n_s8(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_s16
// CHECK: vsli.16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vsli_n_s16(int16x4_t a, int16x4_t b) {
  return vsli_n_s16(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_s32
// CHECK: vsli.32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vsli_n_s32(int32x2_t a, int32x2_t b) {
  return vsli_n_s32(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_s64
// CHECK: vsli.64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x1_t test_vsli_n_s64(int64x1_t a, int64x1_t b) {
  return vsli_n_s64(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_u8
// CHECK: vsli.8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vsli_n_u8(uint8x8_t a, uint8x8_t b) {
  return vsli_n_u8(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_u16
// CHECK: vsli.16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vsli_n_u16(uint16x4_t a, uint16x4_t b) {
  return vsli_n_u16(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_u32
// CHECK: vsli.32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vsli_n_u32(uint32x2_t a, uint32x2_t b) {
  return vsli_n_u32(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_u64
// CHECK: vsli.64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vsli_n_u64(uint64x1_t a, uint64x1_t b) {
  return vsli_n_u64(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_p8
// CHECK: vsli.8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
poly8x8_t test_vsli_n_p8(poly8x8_t a, poly8x8_t b) {
  return vsli_n_p8(a, b, 1);
}

// CHECK-LABEL: test_vsli_n_p16
// CHECK: vsli.16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
poly16x4_t test_vsli_n_p16(poly16x4_t a, poly16x4_t b) {
  return vsli_n_p16(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_s8
// CHECK: vsli.8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vsliq_n_s8(int8x16_t a, int8x16_t b) {
  return vsliq_n_s8(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_s16
// CHECK: vsli.16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vsliq_n_s16(int16x8_t a, int16x8_t b) {
  return vsliq_n_s16(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_s32
// CHECK: vsli.32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vsliq_n_s32(int32x4_t a, int32x4_t b) {
  return vsliq_n_s32(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_s64
// CHECK: vsli.64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vsliq_n_s64(int64x2_t a, int64x2_t b) {
  return vsliq_n_s64(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_u8
// CHECK: vsli.8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vsliq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vsliq_n_u8(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_u16
// CHECK: vsli.16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vsliq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vsliq_n_u16(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_u32
// CHECK: vsli.32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vsliq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vsliq_n_u32(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_u64
// CHECK: vsli.64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vsliq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vsliq_n_u64(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_p8
// CHECK: vsli.8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
poly8x16_t test_vsliq_n_p8(poly8x16_t a, poly8x16_t b) {
  return vsliq_n_p8(a, b, 1);
}

// CHECK-LABEL: test_vsliq_n_p16
// CHECK: vsli.16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
poly16x8_t test_vsliq_n_p16(poly16x8_t a, poly16x8_t b) {
  return vsliq_n_p16(a, b, 1);
}


// CHECK-LABEL: test_vsra_n_s8
// CHECK: vsra.s8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vsra_n_s8(int8x8_t a, int8x8_t b) {
  return vsra_n_s8(a, b, 1);
}

// CHECK-LABEL: test_vsra_n_s16
// CHECK: vsra.s16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vsra_n_s16(int16x4_t a, int16x4_t b) {
  return vsra_n_s16(a, b, 1);
}

// CHECK-LABEL: test_vsra_n_s32
// CHECK: vsra.s32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vsra_n_s32(int32x2_t a, int32x2_t b) {
  return vsra_n_s32(a, b, 1);
}

// CHECK-LABEL: test_vsra_n_s64
// CHECK: vsra.s64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x1_t test_vsra_n_s64(int64x1_t a, int64x1_t b) {
  return vsra_n_s64(a, b, 1);
}

// CHECK-LABEL: test_vsra_n_u8
// CHECK: vsra.u8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vsra_n_u8(uint8x8_t a, uint8x8_t b) {
  return vsra_n_u8(a, b, 1);
}

// CHECK-LABEL: test_vsra_n_u16
// CHECK: vsra.u16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vsra_n_u16(uint16x4_t a, uint16x4_t b) {
  return vsra_n_u16(a, b, 1);
}

// CHECK-LABEL: test_vsra_n_u32
// CHECK: vsra.u32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vsra_n_u32(uint32x2_t a, uint32x2_t b) {
  return vsra_n_u32(a, b, 1);
}

// CHECK-LABEL: test_vsra_n_u64
// CHECK: vsra.u64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vsra_n_u64(uint64x1_t a, uint64x1_t b) {
  return vsra_n_u64(a, b, 1);
}

// CHECK-LABEL: test_vsraq_n_s8
// CHECK: vsra.s8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vsraq_n_s8(int8x16_t a, int8x16_t b) {
  return vsraq_n_s8(a, b, 1);
}

// CHECK-LABEL: test_vsraq_n_s16
// CHECK: vsra.s16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vsraq_n_s16(int16x8_t a, int16x8_t b) {
  return vsraq_n_s16(a, b, 1);
}

// CHECK-LABEL: test_vsraq_n_s32
// CHECK: vsra.s32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vsraq_n_s32(int32x4_t a, int32x4_t b) {
  return vsraq_n_s32(a, b, 1);
}

// CHECK-LABEL: test_vsraq_n_s64
// CHECK: vsra.s64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vsraq_n_s64(int64x2_t a, int64x2_t b) {
  return vsraq_n_s64(a, b, 1);
}

// CHECK-LABEL: test_vsraq_n_u8
// CHECK: vsra.u8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vsraq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vsraq_n_u8(a, b, 1);
}

// CHECK-LABEL: test_vsraq_n_u16
// CHECK: vsra.u16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vsraq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vsraq_n_u16(a, b, 1);
}

// CHECK-LABEL: test_vsraq_n_u32
// CHECK: vsra.u32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vsraq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vsraq_n_u32(a, b, 1);
}

// CHECK-LABEL: test_vsraq_n_u64
// CHECK: vsra.u64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vsraq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vsraq_n_u64(a, b, 1);
}


// CHECK-LABEL: test_vsri_n_s8
// CHECK: vsri.8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int8x8_t test_vsri_n_s8(int8x8_t a, int8x8_t b) {
  return vsri_n_s8(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_s16
// CHECK: vsri.16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int16x4_t test_vsri_n_s16(int16x4_t a, int16x4_t b) {
  return vsri_n_s16(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_s32
// CHECK: vsri.32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int32x2_t test_vsri_n_s32(int32x2_t a, int32x2_t b) {
  return vsri_n_s32(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_s64
// CHECK: vsri.64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
int64x1_t test_vsri_n_s64(int64x1_t a, int64x1_t b) {
  return vsri_n_s64(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_u8
// CHECK: vsri.8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint8x8_t test_vsri_n_u8(uint8x8_t a, uint8x8_t b) {
  return vsri_n_u8(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_u16
// CHECK: vsri.16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint16x4_t test_vsri_n_u16(uint16x4_t a, uint16x4_t b) {
  return vsri_n_u16(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_u32
// CHECK: vsri.32 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint32x2_t test_vsri_n_u32(uint32x2_t a, uint32x2_t b) {
  return vsri_n_u32(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_u64
// CHECK: vsri.64 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
uint64x1_t test_vsri_n_u64(uint64x1_t a, uint64x1_t b) {
  return vsri_n_u64(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_p8
// CHECK: vsri.8 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
poly8x8_t test_vsri_n_p8(poly8x8_t a, poly8x8_t b) {
  return vsri_n_p8(a, b, 1);
}

// CHECK-LABEL: test_vsri_n_p16
// CHECK: vsri.16 d{{[0-9]+}}, d{{[0-9]+}}, #{{[0-9]+}}
poly16x4_t test_vsri_n_p16(poly16x4_t a, poly16x4_t b) {
  return vsri_n_p16(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_s8
// CHECK: vsri.8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int8x16_t test_vsriq_n_s8(int8x16_t a, int8x16_t b) {
  return vsriq_n_s8(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_s16
// CHECK: vsri.16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int16x8_t test_vsriq_n_s16(int16x8_t a, int16x8_t b) {
  return vsriq_n_s16(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_s32
// CHECK: vsri.32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int32x4_t test_vsriq_n_s32(int32x4_t a, int32x4_t b) {
  return vsriq_n_s32(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_s64
// CHECK: vsri.64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
int64x2_t test_vsriq_n_s64(int64x2_t a, int64x2_t b) {
  return vsriq_n_s64(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_u8
// CHECK: vsri.8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint8x16_t test_vsriq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vsriq_n_u8(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_u16
// CHECK: vsri.16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint16x8_t test_vsriq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vsriq_n_u16(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_u32
// CHECK: vsri.32 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint32x4_t test_vsriq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vsriq_n_u32(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_u64
// CHECK: vsri.64 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
uint64x2_t test_vsriq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vsriq_n_u64(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_p8
// CHECK: vsri.8 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
poly8x16_t test_vsriq_n_p8(poly8x16_t a, poly8x16_t b) {
  return vsriq_n_p8(a, b, 1);
}

// CHECK-LABEL: test_vsriq_n_p16
// CHECK: vsri.16 q{{[0-9]+}}, q{{[0-9]+}}, #{{[0-9]+}}
poly16x8_t test_vsriq_n_p16(poly16x8_t a, poly16x8_t b) {
  return vsriq_n_p16(a, b, 1);
}


// CHECK-LABEL: test_vst1q_u8
// CHECK: vst1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_u8(uint8_t * a, uint8x16_t b) {
  vst1q_u8(a, b);
}

// CHECK-LABEL: test_vst1q_u16
// CHECK: vst1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_u16(uint16_t * a, uint16x8_t b) {
  vst1q_u16(a, b);
}

// CHECK-LABEL: test_vst1q_u32
// CHECK: vst1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_u32(uint32_t * a, uint32x4_t b) {
  vst1q_u32(a, b);
}

// CHECK-LABEL: test_vst1q_u64
// CHECK: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}{{(:64)?}}]
void test_vst1q_u64(uint64_t * a, uint64x2_t b) {
  vst1q_u64(a, b);
}

// CHECK-LABEL: test_vst1q_s8
// CHECK: vst1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_s8(int8_t * a, int8x16_t b) {
  vst1q_s8(a, b);
}

// CHECK-LABEL: test_vst1q_s16
// CHECK: vst1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_s16(int16_t * a, int16x8_t b) {
  vst1q_s16(a, b);
}

// CHECK-LABEL: test_vst1q_s32
// CHECK: vst1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_s32(int32_t * a, int32x4_t b) {
  vst1q_s32(a, b);
}

// CHECK-LABEL: test_vst1q_s64
// CHECK: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}{{(:64)?}}]
void test_vst1q_s64(int64_t * a, int64x2_t b) {
  vst1q_s64(a, b);
}

// CHECK-LABEL: test_vst1q_f16
// CHECK: vst1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_f16(float16_t * a, float16x8_t b) {
  vst1q_f16(a, b);
}

// CHECK-LABEL: test_vst1q_f32
// CHECK: vst1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_f32(float32_t * a, float32x4_t b) {
  vst1q_f32(a, b);
}

// CHECK-LABEL: test_vst1q_p8
// CHECK: vst1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_p8(poly8_t * a, poly8x16_t b) {
  vst1q_p8(a, b);
}

// CHECK-LABEL: test_vst1q_p16
// CHECK: vst1.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1q_p16(poly16_t * a, poly16x8_t b) {
  vst1q_p16(a, b);
}

// CHECK-LABEL: test_vst1_u8
// CHECK: vst1.8 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_u8(uint8_t * a, uint8x8_t b) {
  vst1_u8(a, b);
}

// CHECK-LABEL: test_vst1_u16
// CHECK: vst1.16 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_u16(uint16_t * a, uint16x4_t b) {
  vst1_u16(a, b);
}

// CHECK-LABEL: test_vst1_u32
// CHECK: vst1.32 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_u32(uint32_t * a, uint32x2_t b) {
  vst1_u32(a, b);
}

// CHECK-LABEL: test_vst1_u64
// CHECK: vst1.64 {d{{[0-9]+}}}, [r{{[0-9]+}}{{(:64)?}}]
void test_vst1_u64(uint64_t * a, uint64x1_t b) {
  vst1_u64(a, b);
}

// CHECK-LABEL: test_vst1_s8
// CHECK: vst1.8 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_s8(int8_t * a, int8x8_t b) {
  vst1_s8(a, b);
}

// CHECK-LABEL: test_vst1_s16
// CHECK: vst1.16 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_s16(int16_t * a, int16x4_t b) {
  vst1_s16(a, b);
}

// CHECK-LABEL: test_vst1_s32
// CHECK: vst1.32 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_s32(int32_t * a, int32x2_t b) {
  vst1_s32(a, b);
}

// CHECK-LABEL: test_vst1_s64
// CHECK: vst1.64 {d{{[0-9]+}}}, [r{{[0-9]+}}{{(:64)?}}]
void test_vst1_s64(int64_t * a, int64x1_t b) {
  vst1_s64(a, b);
}

// CHECK-LABEL: test_vst1_f16
// CHECK: vst1.16 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_f16(float16_t * a, float16x4_t b) {
  vst1_f16(a, b);
}

// CHECK-LABEL: test_vst1_f32
// CHECK: vst1.32 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_f32(float32_t * a, float32x2_t b) {
  vst1_f32(a, b);
}

// CHECK-LABEL: test_vst1_p8
// CHECK: vst1.8 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_p8(poly8_t * a, poly8x8_t b) {
  vst1_p8(a, b);
}

// CHECK-LABEL: test_vst1_p16
// CHECK: vst1.16 {d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst1_p16(poly16_t * a, poly16x4_t b) {
  vst1_p16(a, b);
}


// CHECK-LABEL: test_vst1q_lane_u8
// CHECK: vst1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst1q_lane_u8(uint8_t * a, uint8x16_t b) {
  vst1q_lane_u8(a, b, 15);
}

// CHECK-LABEL: test_vst1q_lane_u16
// CHECK: vst1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
void test_vst1q_lane_u16(uint16_t * a, uint16x8_t b) {
  vst1q_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vst1q_lane_u32
// CHECK: vst1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
void test_vst1q_lane_u32(uint32_t * a, uint32x4_t b) {
  vst1q_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vst1q_lane_u64
// CHECK: {{str|vstr|vmov}}
void test_vst1q_lane_u64(uint64_t * a, uint64x2_t b) {
  vst1q_lane_u64(a, b, 1);
}

// CHECK-LABEL: test_vst1q_lane_s8
// CHECK: vst1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst1q_lane_s8(int8_t * a, int8x16_t b) {
  vst1q_lane_s8(a, b, 15);
}

// CHECK-LABEL: test_vst1q_lane_s16
// CHECK: vst1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
void test_vst1q_lane_s16(int16_t * a, int16x8_t b) {
  vst1q_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vst1q_lane_s32
// CHECK: vst1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
void test_vst1q_lane_s32(int32_t * a, int32x4_t b) {
  vst1q_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vst1q_lane_s64
// CHECK: {{str|vstr|vmov}}
void test_vst1q_lane_s64(int64_t * a, int64x2_t b) {
  vst1q_lane_s64(a, b, 1);
}

// CHECK-LABEL: test_vst1q_lane_f16
// CHECK: vst1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
void test_vst1q_lane_f16(float16_t * a, float16x8_t b) {
  vst1q_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vst1q_lane_f32
// CHECK: vst1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
void test_vst1q_lane_f32(float32_t * a, float32x4_t b) {
  vst1q_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vst1q_lane_p8
// CHECK: vst1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst1q_lane_p8(poly8_t * a, poly8x16_t b) {
  vst1q_lane_p8(a, b, 15);
}

// CHECK-LABEL: test_vst1q_lane_p16
// CHECK: vst1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
void test_vst1q_lane_p16(poly16_t * a, poly16x8_t b) {
  vst1q_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vst1_lane_u8
// CHECK: vst1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst1_lane_u8(uint8_t * a, uint8x8_t b) {
  vst1_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vst1_lane_u16
// CHECK: vst1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
void test_vst1_lane_u16(uint16_t * a, uint16x4_t b) {
  vst1_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vst1_lane_u32
// CHECK: vst1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
void test_vst1_lane_u32(uint32_t * a, uint32x2_t b) {
  vst1_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vst1_lane_u64
// CHECK: {{str|vstr|vmov}}
void test_vst1_lane_u64(uint64_t * a, uint64x1_t b) {
  vst1_lane_u64(a, b, 0);
}

// CHECK-LABEL: test_vst1_lane_s8
// CHECK: vst1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst1_lane_s8(int8_t * a, int8x8_t b) {
  vst1_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vst1_lane_s16
// CHECK: vst1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
void test_vst1_lane_s16(int16_t * a, int16x4_t b) {
  vst1_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vst1_lane_s32
// CHECK: vst1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
void test_vst1_lane_s32(int32_t * a, int32x2_t b) {
  vst1_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vst1_lane_s64
// CHECK: {{str|vstr|vmov}}
void test_vst1_lane_s64(int64_t * a, int64x1_t b) {
  vst1_lane_s64(a, b, 0);
}

// CHECK-LABEL: test_vst1_lane_f16
// CHECK: vst1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
void test_vst1_lane_f16(float16_t * a, float16x4_t b) {
  vst1_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vst1_lane_f32
// CHECK: vst1.32 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:32]
void test_vst1_lane_f32(float32_t * a, float32x2_t b) {
  vst1_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vst1_lane_p8
// CHECK: vst1.8 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst1_lane_p8(poly8_t * a, poly8x8_t b) {
  vst1_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vst1_lane_p16
// CHECK: vst1.16 {d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}:16]
void test_vst1_lane_p16(poly16_t * a, poly16x4_t b) {
  vst1_lane_p16(a, b, 3);
}


// CHECK-LABEL: test_vst2q_u8
// CHECK: vst2.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_u8(uint8_t * a, uint8x16x2_t b) {
  vst2q_u8(a, b);
}

// CHECK-LABEL: test_vst2q_u16
// CHECK: vst2.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_u16(uint16_t * a, uint16x8x2_t b) {
  vst2q_u16(a, b);
}

// CHECK-LABEL: test_vst2q_u32
// CHECK: vst2.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_u32(uint32_t * a, uint32x4x2_t b) {
  vst2q_u32(a, b);
}

// CHECK-LABEL: test_vst2q_s8
// CHECK: vst2.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_s8(int8_t * a, int8x16x2_t b) {
  vst2q_s8(a, b);
}

// CHECK-LABEL: test_vst2q_s16
// CHECK: vst2.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_s16(int16_t * a, int16x8x2_t b) {
  vst2q_s16(a, b);
}

// CHECK-LABEL: test_vst2q_s32
// CHECK: vst2.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_s32(int32_t * a, int32x4x2_t b) {
  vst2q_s32(a, b);
}

// CHECK-LABEL: test_vst2q_f16
// CHECK: vst2.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_f16(float16_t * a, float16x8x2_t b) {
  vst2q_f16(a, b);
}

// CHECK-LABEL: test_vst2q_f32
// CHECK: vst2.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_f32(float32_t * a, float32x4x2_t b) {
  vst2q_f32(a, b);
}

// CHECK-LABEL: test_vst2q_p8
// CHECK: vst2.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_p8(poly8_t * a, poly8x16x2_t b) {
  vst2q_p8(a, b);
}

// CHECK-LABEL: test_vst2q_p16
// CHECK: vst2.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2q_p16(poly16_t * a, poly16x8x2_t b) {
  vst2q_p16(a, b);
}

// CHECK-LABEL: test_vst2_u8
// CHECK: vst2.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_u8(uint8_t * a, uint8x8x2_t b) {
  vst2_u8(a, b);
}

// CHECK-LABEL: test_vst2_u16
// CHECK: vst2.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_u16(uint16_t * a, uint16x4x2_t b) {
  vst2_u16(a, b);
}

// CHECK-LABEL: test_vst2_u32
// CHECK: vst2.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_u32(uint32_t * a, uint32x2x2_t b) {
  vst2_u32(a, b);
}

// CHECK-LABEL: test_vst2_u64
// CHECK: vst1.64
void test_vst2_u64(uint64_t * a, uint64x1x2_t b) {
  vst2_u64(a, b);
}

// CHECK-LABEL: test_vst2_s8
// CHECK: vst2.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_s8(int8_t * a, int8x8x2_t b) {
  vst2_s8(a, b);
}

// CHECK-LABEL: test_vst2_s16
// CHECK: vst2.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_s16(int16_t * a, int16x4x2_t b) {
  vst2_s16(a, b);
}

// CHECK-LABEL: test_vst2_s32
// CHECK: vst2.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_s32(int32_t * a, int32x2x2_t b) {
  vst2_s32(a, b);
}

// CHECK-LABEL: test_vst2_s64
// CHECK: vst1.64
void test_vst2_s64(int64_t * a, int64x1x2_t b) {
  vst2_s64(a, b);
}

// CHECK-LABEL: test_vst2_f16
// CHECK: vst2.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_f16(float16_t * a, float16x4x2_t b) {
  vst2_f16(a, b);
}

// CHECK-LABEL: test_vst2_f32
// CHECK: vst2.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_f32(float32_t * a, float32x2x2_t b) {
  vst2_f32(a, b);
}

// CHECK-LABEL: test_vst2_p8
// CHECK: vst2.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_p8(poly8_t * a, poly8x8x2_t b) {
  vst2_p8(a, b);
}

// CHECK-LABEL: test_vst2_p16
// CHECK: vst2.16 {d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst2_p16(poly16_t * a, poly16x4x2_t b) {
  vst2_p16(a, b);
}


// CHECK-LABEL: test_vst2q_lane_u16
// CHECK: vst2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2q_lane_u16(uint16_t * a, uint16x8x2_t b) {
  vst2q_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vst2q_lane_u32
// CHECK: vst2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2q_lane_u32(uint32_t * a, uint32x4x2_t b) {
  vst2q_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vst2q_lane_s16
// CHECK: vst2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2q_lane_s16(int16_t * a, int16x8x2_t b) {
  vst2q_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vst2q_lane_s32
// CHECK: vst2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2q_lane_s32(int32_t * a, int32x4x2_t b) {
  vst2q_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vst2q_lane_f16
// CHECK: vst2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2q_lane_f16(float16_t * a, float16x8x2_t b) {
  vst2q_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vst2q_lane_f32
// CHECK: vst2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2q_lane_f32(float32_t * a, float32x4x2_t b) {
  vst2q_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vst2q_lane_p16
// CHECK: vst2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2q_lane_p16(poly16_t * a, poly16x8x2_t b) {
  vst2q_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vst2_lane_u8
// CHECK: vst2.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_u8(uint8_t * a, uint8x8x2_t b) {
  vst2_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vst2_lane_u16
// CHECK: vst2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_u16(uint16_t * a, uint16x4x2_t b) {
  vst2_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vst2_lane_u32
// CHECK: vst2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_u32(uint32_t * a, uint32x2x2_t b) {
  vst2_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vst2_lane_s8
// CHECK: vst2.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_s8(int8_t * a, int8x8x2_t b) {
  vst2_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vst2_lane_s16
// CHECK: vst2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_s16(int16_t * a, int16x4x2_t b) {
  vst2_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vst2_lane_s32
// CHECK: vst2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_s32(int32_t * a, int32x2x2_t b) {
  vst2_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vst2_lane_f16
// CHECK: vst2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_f16(float16_t * a, float16x4x2_t b) {
  vst2_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vst2_lane_f32
// CHECK: vst2.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_f32(float32_t * a, float32x2x2_t b) {
  vst2_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vst2_lane_p8
// CHECK: vst2.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_p8(poly8_t * a, poly8x8x2_t b) {
  vst2_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vst2_lane_p16
// CHECK: vst2.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst2_lane_p16(poly16_t * a, poly16x4x2_t b) {
  vst2_lane_p16(a, b, 3);
}


// CHECK-LABEL: test_vst3q_u8
// CHECK: vst3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_u8(uint8_t * a, uint8x16x3_t b) {
  vst3q_u8(a, b);
}

// CHECK-LABEL: test_vst3q_u16
// CHECK: vst3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_u16(uint16_t * a, uint16x8x3_t b) {
  vst3q_u16(a, b);
}

// CHECK-LABEL: test_vst3q_u32
// CHECK: vst3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_u32(uint32_t * a, uint32x4x3_t b) {
  vst3q_u32(a, b);
}

// CHECK-LABEL: test_vst3q_s8
// CHECK: vst3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_s8(int8_t * a, int8x16x3_t b) {
  vst3q_s8(a, b);
}

// CHECK-LABEL: test_vst3q_s16
// CHECK: vst3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_s16(int16_t * a, int16x8x3_t b) {
  vst3q_s16(a, b);
}

// CHECK-LABEL: test_vst3q_s32
// CHECK: vst3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_s32(int32_t * a, int32x4x3_t b) {
  vst3q_s32(a, b);
}

// CHECK-LABEL: test_vst3q_f16
// CHECK: vst3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_f16(float16_t * a, float16x8x3_t b) {
  vst3q_f16(a, b);
}

// CHECK-LABEL: test_vst3q_f32
// CHECK: vst3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_f32(float32_t * a, float32x4x3_t b) {
  vst3q_f32(a, b);
}

// CHECK-LABEL: test_vst3q_p8
// CHECK: vst3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_p8(poly8_t * a, poly8x16x3_t b) {
  vst3q_p8(a, b);
}

// CHECK-LABEL: test_vst3q_p16
// CHECK: vst3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst3q_p16(poly16_t * a, poly16x8x3_t b) {
  vst3q_p16(a, b);
}

// CHECK-LABEL: test_vst3_u8
// CHECK: vst3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_u8(uint8_t * a, uint8x8x3_t b) {
  vst3_u8(a, b);
}

// CHECK-LABEL: test_vst3_u16
// CHECK: vst3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_u16(uint16_t * a, uint16x4x3_t b) {
  vst3_u16(a, b);
}

// CHECK-LABEL: test_vst3_u32
// CHECK: vst3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_u32(uint32_t * a, uint32x2x3_t b) {
  vst3_u32(a, b);
}

// CHECK-LABEL: test_vst3_u64
// CHECK: vst1.64
void test_vst3_u64(uint64_t * a, uint64x1x3_t b) {
  vst3_u64(a, b);
}

// CHECK-LABEL: test_vst3_s8
// CHECK: vst3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_s8(int8_t * a, int8x8x3_t b) {
  vst3_s8(a, b);
}

// CHECK-LABEL: test_vst3_s16
// CHECK: vst3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_s16(int16_t * a, int16x4x3_t b) {
  vst3_s16(a, b);
}

// CHECK-LABEL: test_vst3_s32
// CHECK: vst3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_s32(int32_t * a, int32x2x3_t b) {
  vst3_s32(a, b);
}

// CHECK-LABEL: test_vst3_s64
// CHECK: vst1.64
void test_vst3_s64(int64_t * a, int64x1x3_t b) {
  vst3_s64(a, b);
}

// CHECK-LABEL: test_vst3_f16
// CHECK: vst3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_f16(float16_t * a, float16x4x3_t b) {
  vst3_f16(a, b);
}

// CHECK-LABEL: test_vst3_f32
// CHECK: vst3.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_f32(float32_t * a, float32x2x3_t b) {
  vst3_f32(a, b);
}

// CHECK-LABEL: test_vst3_p8
// CHECK: vst3.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_p8(poly8_t * a, poly8x8x3_t b) {
  vst3_p8(a, b);
}

// CHECK-LABEL: test_vst3_p16
// CHECK: vst3.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst3_p16(poly16_t * a, poly16x4x3_t b) {
  vst3_p16(a, b);
}


// CHECK-LABEL: test_vst3q_lane_u16
// CHECK: vst3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst3q_lane_u16(uint16_t * a, uint16x8x3_t b) {
  vst3q_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vst3q_lane_u32
// CHECK: vst3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst3q_lane_u32(uint32_t * a, uint32x4x3_t b) {
  vst3q_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vst3q_lane_s16
// CHECK: vst3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst3q_lane_s16(int16_t * a, int16x8x3_t b) {
  vst3q_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vst3q_lane_s32
// CHECK: vst3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst3q_lane_s32(int32_t * a, int32x4x3_t b) {
  vst3q_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vst3q_lane_f16
// CHECK: vst3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst3q_lane_f16(float16_t * a, float16x8x3_t b) {
  vst3q_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vst3q_lane_f32
// CHECK: vst3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst3q_lane_f32(float32_t * a, float32x4x3_t b) {
  vst3q_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vst3q_lane_p16
// CHECK: vst3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst3q_lane_p16(poly16_t * a, poly16x8x3_t b) {
  vst3q_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vst3_lane_u8
// CHECK: vst3.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_u8(uint8_t * a, uint8x8x3_t b) {
  vst3_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vst3_lane_u16
// CHECK: vst3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_u16(uint16_t * a, uint16x4x3_t b) {
  vst3_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vst3_lane_u32
// CHECK: vst3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_u32(uint32_t * a, uint32x2x3_t b) {
  vst3_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vst3_lane_s8
// CHECK: vst3.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_s8(int8_t * a, int8x8x3_t b) {
  vst3_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vst3_lane_s16
// CHECK: vst3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_s16(int16_t * a, int16x4x3_t b) {
  vst3_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vst3_lane_s32
// CHECK: vst3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_s32(int32_t * a, int32x2x3_t b) {
  vst3_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vst3_lane_f16
// CHECK: vst3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_f16(float16_t * a, float16x4x3_t b) {
  vst3_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vst3_lane_f32
// CHECK: vst3.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_f32(float32_t * a, float32x2x3_t b) {
  vst3_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vst3_lane_p8
// CHECK: vst3.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_p8(poly8_t * a, poly8x8x3_t b) {
  vst3_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vst3_lane_p16
// CHECK: vst3.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst3_lane_p16(poly16_t * a, poly16x4x3_t b) {
  vst3_lane_p16(a, b, 3);
}


// CHECK-LABEL: test_vst4q_u8
// CHECK: vst4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_u8(uint8_t * a, uint8x16x4_t b) {
  vst4q_u8(a, b);
}

// CHECK-LABEL: test_vst4q_u16
// CHECK: vst4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_u16(uint16_t * a, uint16x8x4_t b) {
  vst4q_u16(a, b);
}

// CHECK-LABEL: test_vst4q_u32
// CHECK: vst4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_u32(uint32_t * a, uint32x4x4_t b) {
  vst4q_u32(a, b);
}

// CHECK-LABEL: test_vst4q_s8
// CHECK: vst4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_s8(int8_t * a, int8x16x4_t b) {
  vst4q_s8(a, b);
}

// CHECK-LABEL: test_vst4q_s16
// CHECK: vst4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_s16(int16_t * a, int16x8x4_t b) {
  vst4q_s16(a, b);
}

// CHECK-LABEL: test_vst4q_s32
// CHECK: vst4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_s32(int32_t * a, int32x4x4_t b) {
  vst4q_s32(a, b);
}

// CHECK-LABEL: test_vst4q_f16
// CHECK: vst4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_f16(float16_t * a, float16x8x4_t b) {
  vst4q_f16(a, b);
}

// CHECK-LABEL: test_vst4q_f32
// CHECK: vst4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_f32(float32_t * a, float32x4x4_t b) {
  vst4q_f32(a, b);
}

// CHECK-LABEL: test_vst4q_p8
// CHECK: vst4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_p8(poly8_t * a, poly8x16x4_t b) {
  vst4q_p8(a, b);
}

// CHECK-LABEL: test_vst4q_p16
// CHECK: vst4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}
void test_vst4q_p16(poly16_t * a, poly16x8x4_t b) {
  vst4q_p16(a, b);
}

// CHECK-LABEL: test_vst4_u8
// CHECK: vst4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_u8(uint8_t * a, uint8x8x4_t b) {
  vst4_u8(a, b);
}

// CHECK-LABEL: test_vst4_u16
// CHECK: vst4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_u16(uint16_t * a, uint16x4x4_t b) {
  vst4_u16(a, b);
}

// CHECK-LABEL: test_vst4_u32
// CHECK: vst4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_u32(uint32_t * a, uint32x2x4_t b) {
  vst4_u32(a, b);
}

// CHECK-LABEL: test_vst4_u64
// CHECK: vst1.64
void test_vst4_u64(uint64_t * a, uint64x1x4_t b) {
  vst4_u64(a, b);
}

// CHECK-LABEL: test_vst4_s8
// CHECK: vst4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_s8(int8_t * a, int8x8x4_t b) {
  vst4_s8(a, b);
}

// CHECK-LABEL: test_vst4_s16
// CHECK: vst4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_s16(int16_t * a, int16x4x4_t b) {
  vst4_s16(a, b);
}

// CHECK-LABEL: test_vst4_s32
// CHECK: vst4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_s32(int32_t * a, int32x2x4_t b) {
  vst4_s32(a, b);
}

// CHECK-LABEL: test_vst4_s64
// CHECK: vst1.64
void test_vst4_s64(int64_t * a, int64x1x4_t b) {
  vst4_s64(a, b);
}

// CHECK-LABEL: test_vst4_f16
// CHECK: vst4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_f16(float16_t * a, float16x4x4_t b) {
  vst4_f16(a, b);
}

// CHECK-LABEL: test_vst4_f32
// CHECK: vst4.32 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_f32(float32_t * a, float32x2x4_t b) {
  vst4_f32(a, b);
}

// CHECK-LABEL: test_vst4_p8
// CHECK: vst4.8 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_p8(poly8_t * a, poly8x8x4_t b) {
  vst4_p8(a, b);
}

// CHECK-LABEL: test_vst4_p16
// CHECK: vst4.16 {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, [r{{[0-9]+}}]
void test_vst4_p16(poly16_t * a, poly16x4x4_t b) {
  vst4_p16(a, b);
}


// CHECK-LABEL: test_vst4q_lane_u16
// CHECK: vst4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst4q_lane_u16(uint16_t * a, uint16x8x4_t b) {
  vst4q_lane_u16(a, b, 7);
}

// CHECK-LABEL: test_vst4q_lane_u32
// CHECK: vst4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst4q_lane_u32(uint32_t * a, uint32x4x4_t b) {
  vst4q_lane_u32(a, b, 3);
}

// CHECK-LABEL: test_vst4q_lane_s16
// CHECK: vst4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst4q_lane_s16(int16_t * a, int16x8x4_t b) {
  vst4q_lane_s16(a, b, 7);
}

// CHECK-LABEL: test_vst4q_lane_s32
// CHECK: vst4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst4q_lane_s32(int32_t * a, int32x4x4_t b) {
  vst4q_lane_s32(a, b, 3);
}

// CHECK-LABEL: test_vst4q_lane_f16
// CHECK: vst4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst4q_lane_f16(float16_t * a, float16x8x4_t b) {
  vst4q_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vst4q_lane_f32
// CHECK: vst4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst4q_lane_f32(float32_t * a, float32x4x4_t b) {
  vst4q_lane_f32(a, b, 3);
}

// CHECK-LABEL: test_vst4q_lane_p16
// CHECK: vst4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}
void test_vst4q_lane_p16(poly16_t * a, poly16x8x4_t b) {
  vst4q_lane_p16(a, b, 7);
}

// CHECK-LABEL: test_vst4_lane_u8
// CHECK: vst4.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_u8(uint8_t * a, uint8x8x4_t b) {
  vst4_lane_u8(a, b, 7);
}

// CHECK-LABEL: test_vst4_lane_u16
// CHECK: vst4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_u16(uint16_t * a, uint16x4x4_t b) {
  vst4_lane_u16(a, b, 3);
}

// CHECK-LABEL: test_vst4_lane_u32
// CHECK: vst4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_u32(uint32_t * a, uint32x2x4_t b) {
  vst4_lane_u32(a, b, 1);
}

// CHECK-LABEL: test_vst4_lane_s8
// CHECK: vst4.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_s8(int8_t * a, int8x8x4_t b) {
  vst4_lane_s8(a, b, 7);
}

// CHECK-LABEL: test_vst4_lane_s16
// CHECK: vst4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_s16(int16_t * a, int16x4x4_t b) {
  vst4_lane_s16(a, b, 3);
}

// CHECK-LABEL: test_vst4_lane_s32
// CHECK: vst4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_s32(int32_t * a, int32x2x4_t b) {
  vst4_lane_s32(a, b, 1);
}

// CHECK-LABEL: test_vst4_lane_f16
// CHECK: vst4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_f16(float16_t * a, float16x4x4_t b) {
  vst4_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vst4_lane_f32
// CHECK: vst4.32 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_f32(float32_t * a, float32x2x4_t b) {
  vst4_lane_f32(a, b, 1);
}

// CHECK-LABEL: test_vst4_lane_p8
// CHECK: vst4.8 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_p8(poly8_t * a, poly8x8x4_t b) {
  vst4_lane_p8(a, b, 7);
}

// CHECK-LABEL: test_vst4_lane_p16
// CHECK: vst4.16 {d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}], d{{[0-9]+}}[{{[0-9]+}}]}, [r{{[0-9]+}}]
void test_vst4_lane_p16(poly16_t * a, poly16x4x4_t b) {
  vst4_lane_p16(a, b, 3);
}


// CHECK-LABEL: test_vsub_s8
// CHECK: vsub.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int8x8_t test_vsub_s8(int8x8_t a, int8x8_t b) {
  return vsub_s8(a, b);
}

// CHECK-LABEL: test_vsub_s16
// CHECK: vsub.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x4_t test_vsub_s16(int16x4_t a, int16x4_t b) {
  return vsub_s16(a, b);
}

// CHECK-LABEL: test_vsub_s32
// CHECK: vsub.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x2_t test_vsub_s32(int32x2_t a, int32x2_t b) {
  return vsub_s32(a, b);
}

// CHECK-LABEL: test_vsub_s64
// CHECK: vsub.i64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x1_t test_vsub_s64(int64x1_t a, int64x1_t b) {
  return vsub_s64(a, b);
}

// CHECK-LABEL: test_vsub_f32
// CHECK: vsub.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
float32x2_t test_vsub_f32(float32x2_t a, float32x2_t b) {
  return vsub_f32(a, b);
}

// CHECK-LABEL: test_vsub_u8
// CHECK: vsub.i8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vsub_u8(uint8x8_t a, uint8x8_t b) {
  return vsub_u8(a, b);
}

// CHECK-LABEL: test_vsub_u16
// CHECK: vsub.i16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vsub_u16(uint16x4_t a, uint16x4_t b) {
  return vsub_u16(a, b);
}

// CHECK-LABEL: test_vsub_u32
// CHECK: vsub.i32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vsub_u32(uint32x2_t a, uint32x2_t b) {
  return vsub_u32(a, b);
}

// CHECK-LABEL: test_vsub_u64
// CHECK: vsub.i64 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x1_t test_vsub_u64(uint64x1_t a, uint64x1_t b) {
  return vsub_u64(a, b);
}

// CHECK-LABEL: test_vsubq_s8
// CHECK: vsub.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x16_t test_vsubq_s8(int8x16_t a, int8x16_t b) {
  return vsubq_s8(a, b);
}

// CHECK-LABEL: test_vsubq_s16
// CHECK: vsub.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x8_t test_vsubq_s16(int16x8_t a, int16x8_t b) {
  return vsubq_s16(a, b);
}

// CHECK-LABEL: test_vsubq_s32
// CHECK: vsub.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x4_t test_vsubq_s32(int32x4_t a, int32x4_t b) {
  return vsubq_s32(a, b);
}

// CHECK-LABEL: test_vsubq_s64
// CHECK: vsub.i64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int64x2_t test_vsubq_s64(int64x2_t a, int64x2_t b) {
  return vsubq_s64(a, b);
}

// CHECK-LABEL: test_vsubq_f32
// CHECK: vsub.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
float32x4_t test_vsubq_f32(float32x4_t a, float32x4_t b) {
  return vsubq_f32(a, b);
}

// CHECK-LABEL: test_vsubq_u8
// CHECK: vsub.i8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vsubq_u8(uint8x16_t a, uint8x16_t b) {
  return vsubq_u8(a, b);
}

// CHECK-LABEL: test_vsubq_u16
// CHECK: vsub.i16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vsubq_u16(uint16x8_t a, uint16x8_t b) {
  return vsubq_u16(a, b);
}

// CHECK-LABEL: test_vsubq_u32
// CHECK: vsub.i32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vsubq_u32(uint32x4_t a, uint32x4_t b) {
  return vsubq_u32(a, b);
}

// CHECK-LABEL: test_vsubq_u64
// CHECK: vsub.i64 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint64x2_t test_vsubq_u64(uint64x2_t a, uint64x2_t b) {
  return vsubq_u64(a, b);
}


// CHECK-LABEL: test_vsubhn_s16
// CHECK: vsubhn.i16 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int8x8_t test_vsubhn_s16(int16x8_t a, int16x8_t b) {
  return vsubhn_s16(a, b);
}

// CHECK-LABEL: test_vsubhn_s32
// CHECK: vsubhn.i32 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int16x4_t test_vsubhn_s32(int32x4_t a, int32x4_t b) {
  return vsubhn_s32(a, b);
}

// CHECK-LABEL: test_vsubhn_s64
// CHECK: vsubhn.i64 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
int32x2_t test_vsubhn_s64(int64x2_t a, int64x2_t b) {
  return vsubhn_s64(a, b);
}

// CHECK-LABEL: test_vsubhn_u16
// CHECK: vsubhn.i16 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x8_t test_vsubhn_u16(uint16x8_t a, uint16x8_t b) {
  return vsubhn_u16(a, b);
}

// CHECK-LABEL: test_vsubhn_u32
// CHECK: vsubhn.i32 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x4_t test_vsubhn_u32(uint32x4_t a, uint32x4_t b) {
  return vsubhn_u32(a, b);
}

// CHECK-LABEL: test_vsubhn_u64
// CHECK: vsubhn.i64 d{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x2_t test_vsubhn_u64(uint64x2_t a, uint64x2_t b) {
  return vsubhn_u64(a, b);
}


// CHECK-LABEL: test_vsubl_s8
// CHECK: vsubl.s8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vsubl_s8(int8x8_t a, int8x8_t b) {
  return vsubl_s8(a, b);
}

// CHECK-LABEL: test_vsubl_s16
// CHECK: vsubl.s16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vsubl_s16(int16x4_t a, int16x4_t b) {
  return vsubl_s16(a, b);
}

// CHECK-LABEL: test_vsubl_s32
// CHECK: vsubl.s32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vsubl_s32(int32x2_t a, int32x2_t b) {
  return vsubl_s32(a, b);
}

// CHECK-LABEL: test_vsubl_u8
// CHECK: vsubl.u8 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vsubl_u8(uint8x8_t a, uint8x8_t b) {
  return vsubl_u8(a, b);
}

// CHECK-LABEL: test_vsubl_u16
// CHECK: vsubl.u16 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vsubl_u16(uint16x4_t a, uint16x4_t b) {
  return vsubl_u16(a, b);
}

// CHECK-LABEL: test_vsubl_u32
// CHECK: vsubl.u32 q{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vsubl_u32(uint32x2_t a, uint32x2_t b) {
  return vsubl_u32(a, b);
}


// CHECK-LABEL: test_vsubw_s8
// CHECK: vsubw.s8 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
int16x8_t test_vsubw_s8(int16x8_t a, int8x8_t b) {
  return vsubw_s8(a, b);
}

// CHECK-LABEL: test_vsubw_s16
// CHECK: vsubw.s16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
int32x4_t test_vsubw_s16(int32x4_t a, int16x4_t b) {
  return vsubw_s16(a, b);
}

// CHECK-LABEL: test_vsubw_s32
// CHECK: vsubw.s32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
int64x2_t test_vsubw_s32(int64x2_t a, int32x2_t b) {
  return vsubw_s32(a, b);
}

// CHECK-LABEL: test_vsubw_u8
// CHECK: vsubw.u8 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
uint16x8_t test_vsubw_u8(uint16x8_t a, uint8x8_t b) {
  return vsubw_u8(a, b);
}

// CHECK-LABEL: test_vsubw_u16
// CHECK: vsubw.u16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
uint32x4_t test_vsubw_u16(uint32x4_t a, uint16x4_t b) {
  return vsubw_u16(a, b);
}

// CHECK-LABEL: test_vsubw_u32
// CHECK: vsubw.u32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}
uint64x2_t test_vsubw_u32(uint64x2_t a, uint32x2_t b) {
  return vsubw_u32(a, b);
}


// CHECK-LABEL: test_vtbl1_u8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}}, d{{[0-9]+}}
uint8x8_t test_vtbl1_u8(uint8x8_t a, uint8x8_t b) {
  return vtbl1_u8(a, b);
}

// CHECK-LABEL: test_vtbl1_s8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}}, d{{[0-9]+}}
int8x8_t test_vtbl1_s8(int8x8_t a, int8x8_t b) {
  return vtbl1_s8(a, b);
}

// CHECK-LABEL: test_vtbl1_p8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}}, d{{[0-9]+}}
poly8x8_t test_vtbl1_p8(poly8x8_t a, uint8x8_t b) {
  return vtbl1_p8(a, b);
}


// CHECK-LABEL: test_vtbl2_u8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
uint8x8_t test_vtbl2_u8(uint8x8x2_t a, uint8x8_t b) {
  return vtbl2_u8(a, b);
}

// CHECK-LABEL: test_vtbl2_s8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
int8x8_t test_vtbl2_s8(int8x8x2_t a, int8x8_t b) {
  return vtbl2_s8(a, b);
}

// CHECK-LABEL: test_vtbl2_p8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
poly8x8_t test_vtbl2_p8(poly8x8x2_t a, uint8x8_t b) {
  return vtbl2_p8(a, b);
}


// CHECK-LABEL: test_vtbl3_u8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
uint8x8_t test_vtbl3_u8(uint8x8x3_t a, uint8x8_t b) {
  return vtbl3_u8(a, b);
}

// CHECK-LABEL: test_vtbl3_s8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
int8x8_t test_vtbl3_s8(int8x8x3_t a, int8x8_t b) {
  return vtbl3_s8(a, b);
}

// CHECK-LABEL: test_vtbl3_p8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
poly8x8_t test_vtbl3_p8(poly8x8x3_t a, uint8x8_t b) {
  return vtbl3_p8(a, b);
}


// CHECK-LABEL: test_vtbl4_u8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
uint8x8_t test_vtbl4_u8(uint8x8x4_t a, uint8x8_t b) {
  return vtbl4_u8(a, b);
}

// CHECK-LABEL: test_vtbl4_s8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
int8x8_t test_vtbl4_s8(int8x8x4_t a, int8x8_t b) {
  return vtbl4_s8(a, b);
}

// CHECK-LABEL: test_vtbl4_p8
// CHECK: vtbl.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
poly8x8_t test_vtbl4_p8(poly8x8x4_t a, uint8x8_t b) {
  return vtbl4_p8(a, b);
}


// CHECK-LABEL: test_vtbx1_u8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}}, d{{[0-9]+}}
uint8x8_t test_vtbx1_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c) {
  return vtbx1_u8(a, b, c);
}

// CHECK-LABEL: test_vtbx1_s8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}}, d{{[0-9]+}}
int8x8_t test_vtbx1_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
  return vtbx1_s8(a, b, c);
}

// CHECK-LABEL: test_vtbx1_p8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}}, d{{[0-9]+}}
poly8x8_t test_vtbx1_p8(poly8x8_t a, poly8x8_t b, uint8x8_t c) {
  return vtbx1_p8(a, b, c);
}


// CHECK-LABEL: test_vtbx2_u8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
uint8x8_t test_vtbx2_u8(uint8x8_t a, uint8x8x2_t b, uint8x8_t c) {
  return vtbx2_u8(a, b, c);
}

// CHECK-LABEL: test_vtbx2_s8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
int8x8_t test_vtbx2_s8(int8x8_t a, int8x8x2_t b, int8x8_t c) {
  return vtbx2_s8(a, b, c);
}

// CHECK-LABEL: test_vtbx2_p8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
poly8x8_t test_vtbx2_p8(poly8x8_t a, poly8x8x2_t b, uint8x8_t c) {
  return vtbx2_p8(a, b, c);
}


// CHECK-LABEL: test_vtbx3_u8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
uint8x8_t test_vtbx3_u8(uint8x8_t a, uint8x8x3_t b, uint8x8_t c) {
  return vtbx3_u8(a, b, c);
}

// CHECK-LABEL: test_vtbx3_s8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
int8x8_t test_vtbx3_s8(int8x8_t a, int8x8x3_t b, int8x8_t c) {
  return vtbx3_s8(a, b, c);
}

// CHECK-LABEL: test_vtbx3_p8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
poly8x8_t test_vtbx3_p8(poly8x8_t a, poly8x8x3_t b, uint8x8_t c) {
  return vtbx3_p8(a, b, c);
}


// CHECK-LABEL: test_vtbx4_u8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
uint8x8_t test_vtbx4_u8(uint8x8_t a, uint8x8x4_t b, uint8x8_t c) {
  return vtbx4_u8(a, b, c);
}

// CHECK-LABEL: test_vtbx4_s8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
int8x8_t test_vtbx4_s8(int8x8_t a, int8x8x4_t b, int8x8_t c) {
  return vtbx4_s8(a, b, c);
}

// CHECK-LABEL: test_vtbx4_p8
// CHECK: vtbx.8 d{{[0-9]+}}, {d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}}, d{{[0-9]+}}
poly8x8_t test_vtbx4_p8(poly8x8_t a, poly8x8x4_t b, uint8x8_t c) {
  return vtbx4_p8(a, b, c);
}


// CHECK-LABEL: test_vtrn_s8
// CHECK: vtrn.8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8x2_t test_vtrn_s8(int8x8_t a, int8x8_t b) {
  return vtrn_s8(a, b);
}

// CHECK-LABEL: test_vtrn_s16
// CHECK: vtrn.16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4x2_t test_vtrn_s16(int16x4_t a, int16x4_t b) {
  return vtrn_s16(a, b);
}

// CHECK-LABEL: test_vtrn_s32
// CHECK: vtrn.32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2x2_t test_vtrn_s32(int32x2_t a, int32x2_t b) {
  return vtrn_s32(a, b);
}

// CHECK-LABEL: test_vtrn_u8
// CHECK: vtrn.8 d{{[0-9]+}}, d{{[0-9]+}}
uint8x8x2_t test_vtrn_u8(uint8x8_t a, uint8x8_t b) {
  return vtrn_u8(a, b);
}

// CHECK-LABEL: test_vtrn_u16
// CHECK: vtrn.16 d{{[0-9]+}}, d{{[0-9]+}}
uint16x4x2_t test_vtrn_u16(uint16x4_t a, uint16x4_t b) {
  return vtrn_u16(a, b);
}

// CHECK-LABEL: test_vtrn_u32
// CHECK: vtrn.32 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2x2_t test_vtrn_u32(uint32x2_t a, uint32x2_t b) {
  return vtrn_u32(a, b);
}

// CHECK-LABEL: test_vtrn_f32
// CHECK: vtrn.32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2x2_t test_vtrn_f32(float32x2_t a, float32x2_t b) {
  return vtrn_f32(a, b);
}

// CHECK-LABEL: test_vtrn_p8
// CHECK: vtrn.8 d{{[0-9]+}}, d{{[0-9]+}}
poly8x8x2_t test_vtrn_p8(poly8x8_t a, poly8x8_t b) {
  return vtrn_p8(a, b);
}

// CHECK-LABEL: test_vtrn_p16
// CHECK: vtrn.16 d{{[0-9]+}}, d{{[0-9]+}}
poly16x4x2_t test_vtrn_p16(poly16x4_t a, poly16x4_t b) {
  return vtrn_p16(a, b);
}

// CHECK-LABEL: test_vtrnq_s8
// CHECK: vtrn.8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16x2_t test_vtrnq_s8(int8x16_t a, int8x16_t b) {
  return vtrnq_s8(a, b);
}

// CHECK-LABEL: test_vtrnq_s16
// CHECK: vtrn.16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8x2_t test_vtrnq_s16(int16x8_t a, int16x8_t b) {
  return vtrnq_s16(a, b);
}

// CHECK-LABEL: test_vtrnq_s32
// CHECK: vtrn.32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4x2_t test_vtrnq_s32(int32x4_t a, int32x4_t b) {
  return vtrnq_s32(a, b);
}

// CHECK-LABEL: test_vtrnq_u8
// CHECK: vtrn.8 q{{[0-9]+}}, q{{[0-9]+}}
uint8x16x2_t test_vtrnq_u8(uint8x16_t a, uint8x16_t b) {
  return vtrnq_u8(a, b);
}

// CHECK-LABEL: test_vtrnq_u16
// CHECK: vtrn.16 q{{[0-9]+}}, q{{[0-9]+}}
uint16x8x2_t test_vtrnq_u16(uint16x8_t a, uint16x8_t b) {
  return vtrnq_u16(a, b);
}

// CHECK-LABEL: test_vtrnq_u32
// CHECK: vtrn.32 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4x2_t test_vtrnq_u32(uint32x4_t a, uint32x4_t b) {
  return vtrnq_u32(a, b);
}

// CHECK-LABEL: test_vtrnq_f32
// CHECK: vtrn.32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4x2_t test_vtrnq_f32(float32x4_t a, float32x4_t b) {
  return vtrnq_f32(a, b);
}

// CHECK-LABEL: test_vtrnq_p8
// CHECK: vtrn.8 q{{[0-9]+}}, q{{[0-9]+}}
poly8x16x2_t test_vtrnq_p8(poly8x16_t a, poly8x16_t b) {
  return vtrnq_p8(a, b);
}

// CHECK-LABEL: test_vtrnq_p16
// CHECK: vtrn.16 q{{[0-9]+}}, q{{[0-9]+}}
poly16x8x2_t test_vtrnq_p16(poly16x8_t a, poly16x8_t b) {
  return vtrnq_p16(a, b);
}


// CHECK-LABEL: test_vtst_s8
// CHECK: vtst.8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vtst_s8(int8x8_t a, int8x8_t b) {
  return vtst_s8(a, b);
}

// CHECK-LABEL: test_vtst_s16
// CHECK: vtst.16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vtst_s16(int16x4_t a, int16x4_t b) {
  return vtst_s16(a, b);
}

// CHECK-LABEL: test_vtst_s32
// CHECK: vtst.32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vtst_s32(int32x2_t a, int32x2_t b) {
  return vtst_s32(a, b);
}

// CHECK-LABEL: test_vtst_u8
// CHECK: vtst.8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vtst_u8(uint8x8_t a, uint8x8_t b) {
  return vtst_u8(a, b);
}

// CHECK-LABEL: test_vtst_u16
// CHECK: vtst.16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vtst_u16(uint16x4_t a, uint16x4_t b) {
  return vtst_u16(a, b);
}

// CHECK-LABEL: test_vtst_u32
// CHECK: vtst.32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint32x2_t test_vtst_u32(uint32x2_t a, uint32x2_t b) {
  return vtst_u32(a, b);
}

// CHECK-LABEL: test_vtst_p8
// CHECK: vtst.8 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint8x8_t test_vtst_p8(poly8x8_t a, poly8x8_t b) {
  return vtst_p8(a, b);
}

// CHECK-LABEL: test_vtst_p16
// CHECK: vtst.16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
uint16x4_t test_vtst_p16(poly16x4_t a, poly16x4_t b) {
  return vtst_p16(a, b);
}

// CHECK-LABEL: test_vtstq_s8
// CHECK: vtst.8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vtstq_s8(int8x16_t a, int8x16_t b) {
  return vtstq_s8(a, b);
}

// CHECK-LABEL: test_vtstq_s16
// CHECK: vtst.16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vtstq_s16(int16x8_t a, int16x8_t b) {
  return vtstq_s16(a, b);
}

// CHECK-LABEL: test_vtstq_s32
// CHECK: vtst.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vtstq_s32(int32x4_t a, int32x4_t b) {
  return vtstq_s32(a, b);
}

// CHECK-LABEL: test_vtstq_u8
// CHECK: vtst.8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vtstq_u8(uint8x16_t a, uint8x16_t b) {
  return vtstq_u8(a, b);
}

// CHECK-LABEL: test_vtstq_u16
// CHECK: vtst.16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vtstq_u16(uint16x8_t a, uint16x8_t b) {
  return vtstq_u16(a, b);
}

// CHECK-LABEL: test_vtstq_u32
// CHECK: vtst.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint32x4_t test_vtstq_u32(uint32x4_t a, uint32x4_t b) {
  return vtstq_u32(a, b);
}

// CHECK-LABEL: test_vtstq_p8
// CHECK: vtst.8 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint8x16_t test_vtstq_p8(poly8x16_t a, poly8x16_t b) {
  return vtstq_p8(a, b);
}

// CHECK-LABEL: test_vtstq_p16
// CHECK: vtst.16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
uint16x8_t test_vtstq_p16(poly16x8_t a, poly16x8_t b) {
  return vtstq_p16(a, b);
}


// CHECK-LABEL: test_vuzp_s8
// CHECK: vuzp.8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8x2_t test_vuzp_s8(int8x8_t a, int8x8_t b) {
  return vuzp_s8(a, b);
}

// CHECK-LABEL: test_vuzp_s16
// CHECK: vuzp.16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4x2_t test_vuzp_s16(int16x4_t a, int16x4_t b) {
  return vuzp_s16(a, b);
}

// CHECK-LABEL: test_vuzp_s32
// CHECK: {{vtrn|vuzp}}.32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2x2_t test_vuzp_s32(int32x2_t a, int32x2_t b) {
  return vuzp_s32(a, b);
}

// CHECK-LABEL: test_vuzp_u8
// CHECK: vuzp.8 d{{[0-9]+}}, d{{[0-9]+}}
uint8x8x2_t test_vuzp_u8(uint8x8_t a, uint8x8_t b) {
  return vuzp_u8(a, b);
}

// CHECK-LABEL: test_vuzp_u16
// CHECK: vuzp.16 d{{[0-9]+}}, d{{[0-9]+}}
uint16x4x2_t test_vuzp_u16(uint16x4_t a, uint16x4_t b) {
  return vuzp_u16(a, b);
}

// CHECK-LABEL: test_vuzp_u32
// CHECK: {{vtrn|vuzp}}.32 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2x2_t test_vuzp_u32(uint32x2_t a, uint32x2_t b) {
  return vuzp_u32(a, b);
}

// CHECK-LABEL: test_vuzp_f32
// CHECK: {{vtrn|vuzp}}.32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2x2_t test_vuzp_f32(float32x2_t a, float32x2_t b) {
  return vuzp_f32(a, b);
}

// CHECK-LABEL: test_vuzp_p8
// CHECK: vuzp.8 d{{[0-9]+}}, d{{[0-9]+}}
poly8x8x2_t test_vuzp_p8(poly8x8_t a, poly8x8_t b) {
  return vuzp_p8(a, b);
}

// CHECK-LABEL: test_vuzp_p16
// CHECK: vuzp.16 d{{[0-9]+}}, d{{[0-9]+}}
poly16x4x2_t test_vuzp_p16(poly16x4_t a, poly16x4_t b) {
  return vuzp_p16(a, b);
}

// CHECK-LABEL: test_vuzpq_s8
// CHECK: vuzp.8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16x2_t test_vuzpq_s8(int8x16_t a, int8x16_t b) {
  return vuzpq_s8(a, b);
}

// CHECK-LABEL: test_vuzpq_s16
// CHECK: vuzp.16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8x2_t test_vuzpq_s16(int16x8_t a, int16x8_t b) {
  return vuzpq_s16(a, b);
}

// CHECK-LABEL: test_vuzpq_s32
// CHECK: {{vtrn|vuzp}}.32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4x2_t test_vuzpq_s32(int32x4_t a, int32x4_t b) {
  return vuzpq_s32(a, b);
}

// CHECK-LABEL: test_vuzpq_u8
// CHECK: vuzp.8 q{{[0-9]+}}, q{{[0-9]+}}
uint8x16x2_t test_vuzpq_u8(uint8x16_t a, uint8x16_t b) {
  return vuzpq_u8(a, b);
}

// CHECK-LABEL: test_vuzpq_u16
// CHECK: vuzp.16 q{{[0-9]+}}, q{{[0-9]+}}
uint16x8x2_t test_vuzpq_u16(uint16x8_t a, uint16x8_t b) {
  return vuzpq_u16(a, b);
}

// CHECK-LABEL: test_vuzpq_u32
// CHECK: {{vtrn|vuzp}}.32 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4x2_t test_vuzpq_u32(uint32x4_t a, uint32x4_t b) {
  return vuzpq_u32(a, b);
}

// CHECK-LABEL: test_vuzpq_f32
// CHECK: {{vtrn|vuzp}}.32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4x2_t test_vuzpq_f32(float32x4_t a, float32x4_t b) {
  return vuzpq_f32(a, b);
}

// CHECK-LABEL: test_vuzpq_p8
// CHECK: vuzp.8 q{{[0-9]+}}, q{{[0-9]+}}
poly8x16x2_t test_vuzpq_p8(poly8x16_t a, poly8x16_t b) {
  return vuzpq_p8(a, b);
}

// CHECK-LABEL: test_vuzpq_p16
// CHECK: vuzp.16 q{{[0-9]+}}, q{{[0-9]+}}
poly16x8x2_t test_vuzpq_p16(poly16x8_t a, poly16x8_t b) {
  return vuzpq_p16(a, b);
}


// CHECK-LABEL: test_vzip_s8
// CHECK: vzip.8 d{{[0-9]+}}, d{{[0-9]+}}
int8x8x2_t test_vzip_s8(int8x8_t a, int8x8_t b) {
  return vzip_s8(a, b);
}

// CHECK-LABEL: test_vzip_s16
// CHECK: vzip.16 d{{[0-9]+}}, d{{[0-9]+}}
int16x4x2_t test_vzip_s16(int16x4_t a, int16x4_t b) {
  return vzip_s16(a, b);
}

// CHECK-LABEL: test_vzip_s32
// CHECK: {{vtrn|vzip}}.32 d{{[0-9]+}}, d{{[0-9]+}}
int32x2x2_t test_vzip_s32(int32x2_t a, int32x2_t b) {
  return vzip_s32(a, b);
}

// CHECK-LABEL: test_vzip_u8
// CHECK: vzip.8 d{{[0-9]+}}, d{{[0-9]+}}
uint8x8x2_t test_vzip_u8(uint8x8_t a, uint8x8_t b) {
  return vzip_u8(a, b);
}

// CHECK-LABEL: test_vzip_u16
// CHECK: vzip.16 d{{[0-9]+}}, d{{[0-9]+}}
uint16x4x2_t test_vzip_u16(uint16x4_t a, uint16x4_t b) {
  return vzip_u16(a, b);
}

// CHECK-LABEL: test_vzip_u32
// CHECK: {{vtrn|vzip}}.32 d{{[0-9]+}}, d{{[0-9]+}}
uint32x2x2_t test_vzip_u32(uint32x2_t a, uint32x2_t b) {
  return vzip_u32(a, b);
}

// CHECK-LABEL: test_vzip_f32
// CHECK: {{vtrn|vzip}}.32 d{{[0-9]+}}, d{{[0-9]+}}
float32x2x2_t test_vzip_f32(float32x2_t a, float32x2_t b) {
  return vzip_f32(a, b);
}

// CHECK-LABEL: test_vzip_p8
// CHECK: vzip.8 d{{[0-9]+}}, d{{[0-9]+}}
poly8x8x2_t test_vzip_p8(poly8x8_t a, poly8x8_t b) {
  return vzip_p8(a, b);
}

// CHECK-LABEL: test_vzip_p16
// CHECK: vzip.16 d{{[0-9]+}}, d{{[0-9]+}}
poly16x4x2_t test_vzip_p16(poly16x4_t a, poly16x4_t b) {
  return vzip_p16(a, b);
}

// CHECK-LABEL: test_vzipq_s8
// CHECK: vzip.8 q{{[0-9]+}}, q{{[0-9]+}}
int8x16x2_t test_vzipq_s8(int8x16_t a, int8x16_t b) {
  return vzipq_s8(a, b);
}

// CHECK-LABEL: test_vzipq_s16
// CHECK: vzip.16 q{{[0-9]+}}, q{{[0-9]+}}
int16x8x2_t test_vzipq_s16(int16x8_t a, int16x8_t b) {
  return vzipq_s16(a, b);
}

// CHECK-LABEL: test_vzipq_s32
// CHECK: {{vtrn|vzip}}.32 q{{[0-9]+}}, q{{[0-9]+}}
int32x4x2_t test_vzipq_s32(int32x4_t a, int32x4_t b) {
  return vzipq_s32(a, b);
}

// CHECK-LABEL: test_vzipq_u8
// CHECK: vzip.8 q{{[0-9]+}}, q{{[0-9]+}}
uint8x16x2_t test_vzipq_u8(uint8x16_t a, uint8x16_t b) {
  return vzipq_u8(a, b);
}

// CHECK-LABEL: test_vzipq_u16
// CHECK: vzip.16 q{{[0-9]+}}, q{{[0-9]+}}
uint16x8x2_t test_vzipq_u16(uint16x8_t a, uint16x8_t b) {
  return vzipq_u16(a, b);
}

// CHECK-LABEL: test_vzipq_u32
// CHECK: {{vtrn|vzip}}.32 q{{[0-9]+}}, q{{[0-9]+}}
uint32x4x2_t test_vzipq_u32(uint32x4_t a, uint32x4_t b) {
  return vzipq_u32(a, b);
}

// CHECK-LABEL: test_vzipq_f32
// CHECK: {{vtrn|vzip}}.32 q{{[0-9]+}}, q{{[0-9]+}}
float32x4x2_t test_vzipq_f32(float32x4_t a, float32x4_t b) {
  return vzipq_f32(a, b);
}

// CHECK-LABEL: test_vzipq_p8
// CHECK: vzip.8 q{{[0-9]+}}, q{{[0-9]+}}
poly8x16x2_t test_vzipq_p8(poly8x16_t a, poly8x16_t b) {
  return vzipq_p8(a, b);
}

// CHECK-LABEL: test_vzipq_p16
// CHECK: vzip.16 q{{[0-9]+}}, q{{[0-9]+}}
poly16x8x2_t test_vzipq_p16(poly16x8_t a, poly16x8_t b) {
  return vzipq_p16(a, b);
}


