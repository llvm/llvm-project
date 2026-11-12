// RUN: %clang_cc1 -triple aarch64 -target-feature +neon -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vand_s8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <8 x i8> %a, %b
// CHECK:   ret <8 x i8> [[AND_I]]
int8x8_t test_vand_s8(int8x8_t a, int8x8_t b) {
  return vand_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vandq_s8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <16 x i8> %a, %b
// CHECK:   ret <16 x i8> [[AND_I]]
int8x16_t test_vandq_s8(int8x16_t a, int8x16_t b) {
  return vandq_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vand_s16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <4 x i16> %a, %b
// CHECK:   ret <4 x i16> [[AND_I]]
int16x4_t test_vand_s16(int16x4_t a, int16x4_t b) {
  return vand_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vandq_s16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <8 x i16> %a, %b
// CHECK:   ret <8 x i16> [[AND_I]]
int16x8_t test_vandq_s16(int16x8_t a, int16x8_t b) {
  return vandq_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vand_s32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <2 x i32> %a, %b
// CHECK:   ret <2 x i32> [[AND_I]]
int32x2_t test_vand_s32(int32x2_t a, int32x2_t b) {
  return vand_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vandq_s32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <4 x i32> %a, %b
// CHECK:   ret <4 x i32> [[AND_I]]
int32x4_t test_vandq_s32(int32x4_t a, int32x4_t b) {
  return vandq_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vand_s64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <1 x i64> %a, %b
// CHECK:   ret <1 x i64> [[AND_I]]
int64x1_t test_vand_s64(int64x1_t a, int64x1_t b) {
  return vand_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vandq_s64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <2 x i64> %a, %b
// CHECK:   ret <2 x i64> [[AND_I]]
int64x2_t test_vandq_s64(int64x2_t a, int64x2_t b) {
  return vandq_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vand_u8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <8 x i8> %a, %b
// CHECK:   ret <8 x i8> [[AND_I]]
uint8x8_t test_vand_u8(uint8x8_t a, uint8x8_t b) {
  return vand_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vandq_u8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <16 x i8> %a, %b
// CHECK:   ret <16 x i8> [[AND_I]]
uint8x16_t test_vandq_u8(uint8x16_t a, uint8x16_t b) {
  return vandq_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vand_u16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <4 x i16> %a, %b
// CHECK:   ret <4 x i16> [[AND_I]]
uint16x4_t test_vand_u16(uint16x4_t a, uint16x4_t b) {
  return vand_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vandq_u16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <8 x i16> %a, %b
// CHECK:   ret <8 x i16> [[AND_I]]
uint16x8_t test_vandq_u16(uint16x8_t a, uint16x8_t b) {
  return vandq_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vand_u32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <2 x i32> %a, %b
// CHECK:   ret <2 x i32> [[AND_I]]
uint32x2_t test_vand_u32(uint32x2_t a, uint32x2_t b) {
  return vand_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vandq_u32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <4 x i32> %a, %b
// CHECK:   ret <4 x i32> [[AND_I]]
uint32x4_t test_vandq_u32(uint32x4_t a, uint32x4_t b) {
  return vandq_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vand_u64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <1 x i64> %a, %b
// CHECK:   ret <1 x i64> [[AND_I]]
uint64x1_t test_vand_u64(uint64x1_t a, uint64x1_t b) {
  return vand_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vandq_u64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[AND_I:%.*]] = and <2 x i64> %a, %b
// CHECK:   ret <2 x i64> [[AND_I]]
uint64x2_t test_vandq_u64(uint64x2_t a, uint64x2_t b) {
  return vandq_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vorr_s8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <8 x i8> %a, %b
// CHECK:   ret <8 x i8> [[OR_I]]
int8x8_t test_vorr_s8(int8x8_t a, int8x8_t b) {
  return vorr_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vorrq_s8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <16 x i8> %a, %b
// CHECK:   ret <16 x i8> [[OR_I]]
int8x16_t test_vorrq_s8(int8x16_t a, int8x16_t b) {
  return vorrq_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vorr_s16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <4 x i16> %a, %b
// CHECK:   ret <4 x i16> [[OR_I]]
int16x4_t test_vorr_s16(int16x4_t a, int16x4_t b) {
  return vorr_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vorrq_s16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <8 x i16> %a, %b
// CHECK:   ret <8 x i16> [[OR_I]]
int16x8_t test_vorrq_s16(int16x8_t a, int16x8_t b) {
  return vorrq_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vorr_s32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <2 x i32> %a, %b
// CHECK:   ret <2 x i32> [[OR_I]]
int32x2_t test_vorr_s32(int32x2_t a, int32x2_t b) {
  return vorr_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vorrq_s32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <4 x i32> %a, %b
// CHECK:   ret <4 x i32> [[OR_I]]
int32x4_t test_vorrq_s32(int32x4_t a, int32x4_t b) {
  return vorrq_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vorr_s64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <1 x i64> %a, %b
// CHECK:   ret <1 x i64> [[OR_I]]
int64x1_t test_vorr_s64(int64x1_t a, int64x1_t b) {
  return vorr_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vorrq_s64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <2 x i64> %a, %b
// CHECK:   ret <2 x i64> [[OR_I]]
int64x2_t test_vorrq_s64(int64x2_t a, int64x2_t b) {
  return vorrq_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vorr_u8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <8 x i8> %a, %b
// CHECK:   ret <8 x i8> [[OR_I]]
uint8x8_t test_vorr_u8(uint8x8_t a, uint8x8_t b) {
  return vorr_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vorrq_u8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <16 x i8> %a, %b
// CHECK:   ret <16 x i8> [[OR_I]]
uint8x16_t test_vorrq_u8(uint8x16_t a, uint8x16_t b) {
  return vorrq_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vorr_u16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <4 x i16> %a, %b
// CHECK:   ret <4 x i16> [[OR_I]]
uint16x4_t test_vorr_u16(uint16x4_t a, uint16x4_t b) {
  return vorr_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vorrq_u16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <8 x i16> %a, %b
// CHECK:   ret <8 x i16> [[OR_I]]
uint16x8_t test_vorrq_u16(uint16x8_t a, uint16x8_t b) {
  return vorrq_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vorr_u32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <2 x i32> %a, %b
// CHECK:   ret <2 x i32> [[OR_I]]
uint32x2_t test_vorr_u32(uint32x2_t a, uint32x2_t b) {
  return vorr_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vorrq_u32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <4 x i32> %a, %b
// CHECK:   ret <4 x i32> [[OR_I]]
uint32x4_t test_vorrq_u32(uint32x4_t a, uint32x4_t b) {
  return vorrq_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vorr_u64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <1 x i64> %a, %b
// CHECK:   ret <1 x i64> [[OR_I]]
uint64x1_t test_vorr_u64(uint64x1_t a, uint64x1_t b) {
  return vorr_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vorrq_u64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[OR_I:%.*]] = or <2 x i64> %a, %b
// CHECK:   ret <2 x i64> [[OR_I]]
uint64x2_t test_vorrq_u64(uint64x2_t a, uint64x2_t b) {
  return vorrq_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_veor_s8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <8 x i8> %a, %b
// CHECK:   ret <8 x i8> [[XOR_I]]
int8x8_t test_veor_s8(int8x8_t a, int8x8_t b) {
  return veor_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_veorq_s8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <16 x i8> %a, %b
// CHECK:   ret <16 x i8> [[XOR_I]]
int8x16_t test_veorq_s8(int8x16_t a, int8x16_t b) {
  return veorq_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_veor_s16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <4 x i16> %a, %b
// CHECK:   ret <4 x i16> [[XOR_I]]
int16x4_t test_veor_s16(int16x4_t a, int16x4_t b) {
  return veor_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_veorq_s16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <8 x i16> %a, %b
// CHECK:   ret <8 x i16> [[XOR_I]]
int16x8_t test_veorq_s16(int16x8_t a, int16x8_t b) {
  return veorq_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_veor_s32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <2 x i32> %a, %b
// CHECK:   ret <2 x i32> [[XOR_I]]
int32x2_t test_veor_s32(int32x2_t a, int32x2_t b) {
  return veor_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_veorq_s32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <4 x i32> %a, %b
// CHECK:   ret <4 x i32> [[XOR_I]]
int32x4_t test_veorq_s32(int32x4_t a, int32x4_t b) {
  return veorq_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_veor_s64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <1 x i64> %a, %b
// CHECK:   ret <1 x i64> [[XOR_I]]
int64x1_t test_veor_s64(int64x1_t a, int64x1_t b) {
  return veor_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_veorq_s64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <2 x i64> %a, %b
// CHECK:   ret <2 x i64> [[XOR_I]]
int64x2_t test_veorq_s64(int64x2_t a, int64x2_t b) {
  return veorq_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_veor_u8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <8 x i8> %a, %b
// CHECK:   ret <8 x i8> [[XOR_I]]
uint8x8_t test_veor_u8(uint8x8_t a, uint8x8_t b) {
  return veor_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_veorq_u8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <16 x i8> %a, %b
// CHECK:   ret <16 x i8> [[XOR_I]]
uint8x16_t test_veorq_u8(uint8x16_t a, uint8x16_t b) {
  return veorq_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_veor_u16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <4 x i16> %a, %b
// CHECK:   ret <4 x i16> [[XOR_I]]
uint16x4_t test_veor_u16(uint16x4_t a, uint16x4_t b) {
  return veor_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_veorq_u16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <8 x i16> %a, %b
// CHECK:   ret <8 x i16> [[XOR_I]]
uint16x8_t test_veorq_u16(uint16x8_t a, uint16x8_t b) {
  return veorq_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_veor_u32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <2 x i32> %a, %b
// CHECK:   ret <2 x i32> [[XOR_I]]
uint32x2_t test_veor_u32(uint32x2_t a, uint32x2_t b) {
  return veor_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_veorq_u32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <4 x i32> %a, %b
// CHECK:   ret <4 x i32> [[XOR_I]]
uint32x4_t test_veorq_u32(uint32x4_t a, uint32x4_t b) {
  return veorq_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_veor_u64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <1 x i64> %a, %b
// CHECK:   ret <1 x i64> [[XOR_I]]
uint64x1_t test_veor_u64(uint64x1_t a, uint64x1_t b) {
  return veor_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_veorq_u64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[XOR_I:%.*]] = xor <2 x i64> %a, %b
// CHECK:   ret <2 x i64> [[XOR_I]]
uint64x2_t test_veorq_u64(uint64x2_t a, uint64x2_t b) {
  return veorq_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vbic_s8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <8 x i8> %b, splat (i8 -1)
// CHECK:   [[AND_I:%.*]] = and <8 x i8> %a, [[NEG_I]]
// CHECK:   ret <8 x i8> [[AND_I]]
int8x8_t test_vbic_s8(int8x8_t a, int8x8_t b) {
  return vbic_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vbicq_s8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <16 x i8> %b, splat (i8 -1)
// CHECK:   [[AND_I:%.*]] = and <16 x i8> %a, [[NEG_I]]
// CHECK:   ret <16 x i8> [[AND_I]]
int8x16_t test_vbicq_s8(int8x16_t a, int8x16_t b) {
  return vbicq_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vbic_s16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <4 x i16> %b, splat (i16 -1)
// CHECK:   [[AND_I:%.*]] = and <4 x i16> %a, [[NEG_I]]
// CHECK:   ret <4 x i16> [[AND_I]]
int16x4_t test_vbic_s16(int16x4_t a, int16x4_t b) {
  return vbic_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vbicq_s16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <8 x i16> %b, splat (i16 -1)
// CHECK:   [[AND_I:%.*]] = and <8 x i16> %a, [[NEG_I]]
// CHECK:   ret <8 x i16> [[AND_I]]
int16x8_t test_vbicq_s16(int16x8_t a, int16x8_t b) {
  return vbicq_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vbic_s32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <2 x i32> %b, splat (i32 -1)
// CHECK:   [[AND_I:%.*]] = and <2 x i32> %a, [[NEG_I]]
// CHECK:   ret <2 x i32> [[AND_I]]
int32x2_t test_vbic_s32(int32x2_t a, int32x2_t b) {
  return vbic_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vbicq_s32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <4 x i32> %b, splat (i32 -1)
// CHECK:   [[AND_I:%.*]] = and <4 x i32> %a, [[NEG_I]]
// CHECK:   ret <4 x i32> [[AND_I]]
int32x4_t test_vbicq_s32(int32x4_t a, int32x4_t b) {
  return vbicq_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vbic_s64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <1 x i64> %b, splat (i64 -1)
// CHECK:   [[AND_I:%.*]] = and <1 x i64> %a, [[NEG_I]]
// CHECK:   ret <1 x i64> [[AND_I]]
int64x1_t test_vbic_s64(int64x1_t a, int64x1_t b) {
  return vbic_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vbicq_s64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <2 x i64> %b, splat (i64 -1)
// CHECK:   [[AND_I:%.*]] = and <2 x i64> %a, [[NEG_I]]
// CHECK:   ret <2 x i64> [[AND_I]]
int64x2_t test_vbicq_s64(int64x2_t a, int64x2_t b) {
  return vbicq_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vbic_u8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <8 x i8> %b, splat (i8 -1)
// CHECK:   [[AND_I:%.*]] = and <8 x i8> %a, [[NEG_I]]
// CHECK:   ret <8 x i8> [[AND_I]]
uint8x8_t test_vbic_u8(uint8x8_t a, uint8x8_t b) {
  return vbic_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vbicq_u8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <16 x i8> %b, splat (i8 -1)
// CHECK:   [[AND_I:%.*]] = and <16 x i8> %a, [[NEG_I]]
// CHECK:   ret <16 x i8> [[AND_I]]
uint8x16_t test_vbicq_u8(uint8x16_t a, uint8x16_t b) {
  return vbicq_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vbic_u16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <4 x i16> %b, splat (i16 -1)
// CHECK:   [[AND_I:%.*]] = and <4 x i16> %a, [[NEG_I]]
// CHECK:   ret <4 x i16> [[AND_I]]
uint16x4_t test_vbic_u16(uint16x4_t a, uint16x4_t b) {
  return vbic_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vbicq_u16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <8 x i16> %b, splat (i16 -1)
// CHECK:   [[AND_I:%.*]] = and <8 x i16> %a, [[NEG_I]]
// CHECK:   ret <8 x i16> [[AND_I]]
uint16x8_t test_vbicq_u16(uint16x8_t a, uint16x8_t b) {
  return vbicq_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vbic_u32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <2 x i32> %b, splat (i32 -1)
// CHECK:   [[AND_I:%.*]] = and <2 x i32> %a, [[NEG_I]]
// CHECK:   ret <2 x i32> [[AND_I]]
uint32x2_t test_vbic_u32(uint32x2_t a, uint32x2_t b) {
  return vbic_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vbicq_u32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <4 x i32> %b, splat (i32 -1)
// CHECK:   [[AND_I:%.*]] = and <4 x i32> %a, [[NEG_I]]
// CHECK:   ret <4 x i32> [[AND_I]]
uint32x4_t test_vbicq_u32(uint32x4_t a, uint32x4_t b) {
  return vbicq_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vbic_u64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <1 x i64> %b, splat (i64 -1)
// CHECK:   [[AND_I:%.*]] = and <1 x i64> %a, [[NEG_I]]
// CHECK:   ret <1 x i64> [[AND_I]]
uint64x1_t test_vbic_u64(uint64x1_t a, uint64x1_t b) {
  return vbic_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vbicq_u64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <2 x i64> %b, splat (i64 -1)
// CHECK:   [[AND_I:%.*]] = and <2 x i64> %a, [[NEG_I]]
// CHECK:   ret <2 x i64> [[AND_I]]
uint64x2_t test_vbicq_u64(uint64x2_t a, uint64x2_t b) {
  return vbicq_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vorn_s8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <8 x i8> %b, splat (i8 -1)
// CHECK:   [[OR_I:%.*]] = or <8 x i8> %a, [[NEG_I]]
// CHECK:   ret <8 x i8> [[OR_I]]
int8x8_t test_vorn_s8(int8x8_t a, int8x8_t b) {
  return vorn_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vornq_s8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <16 x i8> %b, splat (i8 -1)
// CHECK:   [[OR_I:%.*]] = or <16 x i8> %a, [[NEG_I]]
// CHECK:   ret <16 x i8> [[OR_I]]
int8x16_t test_vornq_s8(int8x16_t a, int8x16_t b) {
  return vornq_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vorn_s16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <4 x i16> %b, splat (i16 -1)
// CHECK:   [[OR_I:%.*]] = or <4 x i16> %a, [[NEG_I]]
// CHECK:   ret <4 x i16> [[OR_I]]
int16x4_t test_vorn_s16(int16x4_t a, int16x4_t b) {
  return vorn_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vornq_s16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <8 x i16> %b, splat (i16 -1)
// CHECK:   [[OR_I:%.*]] = or <8 x i16> %a, [[NEG_I]]
// CHECK:   ret <8 x i16> [[OR_I]]
int16x8_t test_vornq_s16(int16x8_t a, int16x8_t b) {
  return vornq_s16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vorn_s32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <2 x i32> %b, splat (i32 -1)
// CHECK:   [[OR_I:%.*]] = or <2 x i32> %a, [[NEG_I]]
// CHECK:   ret <2 x i32> [[OR_I]]
int32x2_t test_vorn_s32(int32x2_t a, int32x2_t b) {
  return vorn_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vornq_s32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <4 x i32> %b, splat (i32 -1)
// CHECK:   [[OR_I:%.*]] = or <4 x i32> %a, [[NEG_I]]
// CHECK:   ret <4 x i32> [[OR_I]]
int32x4_t test_vornq_s32(int32x4_t a, int32x4_t b) {
  return vornq_s32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vorn_s64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <1 x i64> %b, splat (i64 -1)
// CHECK:   [[OR_I:%.*]] = or <1 x i64> %a, [[NEG_I]]
// CHECK:   ret <1 x i64> [[OR_I]]
int64x1_t test_vorn_s64(int64x1_t a, int64x1_t b) {
  return vorn_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vornq_s64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <2 x i64> %b, splat (i64 -1)
// CHECK:   [[OR_I:%.*]] = or <2 x i64> %a, [[NEG_I]]
// CHECK:   ret <2 x i64> [[OR_I]]
int64x2_t test_vornq_s64(int64x2_t a, int64x2_t b) {
  return vornq_s64(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vorn_u8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <8 x i8> %b, splat (i8 -1)
// CHECK:   [[OR_I:%.*]] = or <8 x i8> %a, [[NEG_I]]
// CHECK:   ret <8 x i8> [[OR_I]]
uint8x8_t test_vorn_u8(uint8x8_t a, uint8x8_t b) {
  return vorn_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vornq_u8(<16 x i8> noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <16 x i8> %b, splat (i8 -1)
// CHECK:   [[OR_I:%.*]] = or <16 x i8> %a, [[NEG_I]]
// CHECK:   ret <16 x i8> [[OR_I]]
uint8x16_t test_vornq_u8(uint8x16_t a, uint8x16_t b) {
  return vornq_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vorn_u16(<4 x i16> noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <4 x i16> %b, splat (i16 -1)
// CHECK:   [[OR_I:%.*]] = or <4 x i16> %a, [[NEG_I]]
// CHECK:   ret <4 x i16> [[OR_I]]
uint16x4_t test_vorn_u16(uint16x4_t a, uint16x4_t b) {
  return vorn_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vornq_u16(<8 x i16> noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <8 x i16> %b, splat (i16 -1)
// CHECK:   [[OR_I:%.*]] = or <8 x i16> %a, [[NEG_I]]
// CHECK:   ret <8 x i16> [[OR_I]]
uint16x8_t test_vornq_u16(uint16x8_t a, uint16x8_t b) {
  return vornq_u16(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vorn_u32(<2 x i32> noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <2 x i32> %b, splat (i32 -1)
// CHECK:   [[OR_I:%.*]] = or <2 x i32> %a, [[NEG_I]]
// CHECK:   ret <2 x i32> [[OR_I]]
uint32x2_t test_vorn_u32(uint32x2_t a, uint32x2_t b) {
  return vorn_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vornq_u32(<4 x i32> noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <4 x i32> %b, splat (i32 -1)
// CHECK:   [[OR_I:%.*]] = or <4 x i32> %a, [[NEG_I]]
// CHECK:   ret <4 x i32> [[OR_I]]
uint32x4_t test_vornq_u32(uint32x4_t a, uint32x4_t b) {
  return vornq_u32(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vorn_u64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <1 x i64> %b, splat (i64 -1)
// CHECK:   [[OR_I:%.*]] = or <1 x i64> %a, [[NEG_I]]
// CHECK:   ret <1 x i64> [[OR_I]]
uint64x1_t test_vorn_u64(uint64x1_t a, uint64x1_t b) {
  return vorn_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vornq_u64(<2 x i64> noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[NEG_I:%.*]] = xor <2 x i64> %b, splat (i64 -1)
// CHECK:   [[OR_I:%.*]] = or <2 x i64> %a, [[NEG_I]]
// CHECK:   ret <2 x i64> [[OR_I]]
uint64x2_t test_vornq_u64(uint64x2_t a, uint64x2_t b) {
  return vornq_u64(a, b);
}
