// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=sroa,instcombine | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=sroa,instcombine | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-cir  %s -disable-O0-optnone |                                   FileCheck %s --check-prefixes=ALL,CIR %}

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.1.10.1. Stride
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#stride
//===------------------------------------------------------===//

// ALL-LABEL: @test_vld1q_f16(
float16x8_t test_vld1q_f16(float16_t const *a) {
// CIR: cir.load align(2) {{.*}} : !cir.ptr<!cir.vector<8 x !cir.f16>>, !cir.vector<8 x !cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x half>, ptr [[A]], align 2
// LLVM: ret <8 x half> [[TMP0]]
  return vld1q_f16(a);
}

// ALL-LABEL: @test_vld1q_f32(
float32x4_t test_vld1q_f32(float32_t const *a) {
// CIR: cir.load align(4) {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x float>, ptr [[A]], align 4
// LLVM: ret <4 x float> [[TMP0]]
  return vld1q_f32(a);
}

// ALL-LABEL: @test_vld1q_f64(
float64x2_t test_vld1q_f64(float64_t const *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x double>, ptr [[A]], align 8
// LLVM: ret <2 x double> [[TMP0]]
  return vld1q_f64(a);
}

// ALL-LABEL: @test_vld1q_mf8(
mfloat8x16_t test_vld1q_mf8(mfloat8_t const *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<16 x !u8i>>, !cir.vector<16 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <16 x i8>, ptr [[A]], align 1
// LLVM: ret <16 x i8> [[TMP0]]
  return vld1q_mf8(a);
}

// ALL-LABEL: @test_vld1q_p16(
poly16x8_t test_vld1q_p16(poly16_t const *a) {
// CIR: cir.load align(2) {{.*}} : !cir.ptr<!cir.vector<8 x !s16i>>, !cir.vector<8 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i16>, ptr [[A]], align 2
// LLVM: ret <8 x i16> [[TMP0]]
  return vld1q_p16(a);
}

// ALL-LABEL: @test_vld1q_p64(
poly64x2_t test_vld1q_p64(poly64_t const * ptr) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>

// LLVM-SAME: ptr {{.*}} [[PTR:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x i64>, ptr [[PTR]], align 8
// LLVM: ret <2 x i64> [[TMP0]]
  return vld1q_p64(ptr);
}

// ALL-LABEL: @test_vld1q_p8(
poly8x16_t test_vld1q_p8(poly8_t const *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <16 x i8>, ptr [[A]], align 1
// LLVM: ret <16 x i8> [[TMP0]]
  return vld1q_p8(a);
}

// ALL-LABEL: @test_vld1q_s16(
int16x8_t test_vld1q_s16(int16_t const *a) {
// CIR: cir.load align(2) {{.*}} : !cir.ptr<!cir.vector<8 x !s16i>>, !cir.vector<8 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i16>, ptr [[A]], align 2
// LLVM: ret <8 x i16> [[TMP0]]
  return vld1q_s16(a);
}

// ALL-LABEL: @test_vld1q_s32(
int32x4_t test_vld1q_s32(int32_t const *a) {
// CIR: cir.load align(4) {{.*}} : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x i32>, ptr [[A]], align 4
// LLVM: ret <4 x i32> [[TMP0]]
  return vld1q_s32(a);
}

// ALL-LABEL: @test_vld1q_s64(
int64x2_t test_vld1q_s64(int64_t const *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x i64>, ptr [[A]], align 8
// LLVM: ret <2 x i64> [[TMP0]]
  return vld1q_s64(a);
}

// ALL-LABEL: @test_vld1q_s8(
int8x16_t test_vld1q_s8(int8_t const *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <16 x i8>, ptr [[A]], align 1
// LLVM: ret <16 x i8> [[TMP0]]
  return vld1q_s8(a);
}

// ALL-LABEL: @test_vld1q_u16(
uint16x8_t test_vld1q_u16(uint16_t const *a) {
// CIR: cir.load align(2) {{.*}} : !cir.ptr<!cir.vector<8 x !u16i>>, !cir.vector<8 x !u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i16>, ptr [[A]], align 2
// LLVM: ret <8 x i16> [[TMP0]]
  return vld1q_u16(a);
}

// ALL-LABEL: @test_vld1q_u32(
uint32x4_t test_vld1q_u32(uint32_t const *a) {
// CIR: cir.load align(4) {{.*}} : !cir.ptr<!cir.vector<4 x !u32i>>, !cir.vector<4 x !u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x i32>, ptr [[A]], align 4
// LLVM: ret <4 x i32> [[TMP0]]
  return vld1q_u32(a);
}

// ALL-LABEL: @test_vld1q_u64(
uint64x2_t test_vld1q_u64(uint64_t const *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<2 x !u64i>>, !cir.vector<2 x !u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x i64>, ptr [[A]], align 8
// LLVM: ret <2 x i64> [[TMP0]]
  return vld1q_u64(a);
}

// ALL-LABEL: @test_vld1q_u8(
uint8x16_t test_vld1q_u8(uint8_t const *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<16 x !u8i>>, !cir.vector<16 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <16 x i8>, ptr [[A]], align 1
// LLVM: ret <16 x i8> [[TMP0]]
  return vld1q_u8(a);
}

// ALL-LABEL: @test_vld1_f16(
float16x4_t test_vld1_f16(float16_t const *a) {
// CIR: cir.load align(2) {{.*}} : !cir.ptr<!cir.vector<4 x !cir.f16>>, !cir.vector<4 x !cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x half>, ptr [[A]], align 2
// LLVM: ret <4 x half> [[TMP0]]
  return vld1_f16(a);
}

// ALL-LABEL: @test_vld1_f32(
float32x2_t test_vld1_f32(float32_t const *a) {
// CIR: cir.load align(4) {{.*}} : !cir.ptr<!cir.vector<2 x !cir.float>>, !cir.vector<2 x !cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x float>, ptr [[A]], align 4
// LLVM: ret <2 x float> [[TMP0]]
  return vld1_f32(a);
}

// ALL-LABEL: @test_vld1_f64(
float64x1_t test_vld1_f64(float64_t const *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<1 x !cir.double>>, !cir.vector<1 x !cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <1 x double>, ptr [[A]], align 8
// LLVM: ret <1 x double> [[TMP0]]
  return vld1_f64(a);
}

// ALL-LABEL: @test_vld1_mf8(
mfloat8x8_t test_vld1_mf8(mfloat8_t const *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<8 x !u8i>>, !cir.vector<8 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i8>, ptr [[A]], align 1
// LLVM: ret <8 x i8> [[TMP0]]
  return vld1_mf8(a);
}

// ALL-LABEL: @test_vld1_p16(
poly16x4_t test_vld1_p16(poly16_t const *a) {
// CIR: cir.load align(2) {{.*}} : !cir.ptr<!cir.vector<4 x !s16i>>, !cir.vector<4 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x i16>, ptr [[A]], align 2
// LLVM: ret <4 x i16> [[TMP0]]
  return vld1_p16(a);
}

// ALL-LABEL: @test_vld1_p64(
poly64x1_t test_vld1_p64(poly64_t const * ptr) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<1 x !s64i>>, !cir.vector<1 x !s64i>

// LLVM-SAME: ptr {{.*}} [[PTR:%.*]])
// LLVM: [[TMP0:%.*]] = load <1 x i64>, ptr [[PTR]], align 8
// LLVM: ret <1 x i64> [[TMP0]]
  return vld1_p64(ptr);
}

// ALL-LABEL: @test_vld1_p8(
poly8x8_t test_vld1_p8(poly8_t const *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i8>, ptr [[A]], align 1
// LLVM: ret <8 x i8> [[TMP0]]
  return vld1_p8(a);
}

// ALL-LABEL: @test_vld1_s16(
int16x4_t test_vld1_s16(int16_t const *a) {
// CIR: cir.load align(2) {{.*}} : !cir.ptr<!cir.vector<4 x !s16i>>, !cir.vector<4 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x i16>, ptr [[A]], align 2
// LLVM: ret <4 x i16> [[TMP0]]
  return vld1_s16(a);
}

// ALL-LABEL: @test_vld1_s32(
int32x2_t test_vld1_s32(int32_t const *a) {
// CIR: cir.load align(4) {{.*}} : !cir.ptr<!cir.vector<2 x !s32i>>, !cir.vector<2 x !s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x i32>, ptr [[A]], align 4
// LLVM: ret <2 x i32> [[TMP0]]
  return vld1_s32(a);
}

// ALL-LABEL: @test_vld1_s64(
int64x1_t test_vld1_s64(int64_t const *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<1 x !s64i>>, !cir.vector<1 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <1 x i64>, ptr [[A]], align 8
// LLVM: ret <1 x i64> [[TMP0]]
  return vld1_s64(a);
}

// ALL-LABEL: @test_vld1_s8(
int8x8_t test_vld1_s8(int8_t const *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i8>, ptr [[A]], align 1
// LLVM: ret <8 x i8> [[TMP0]]
  return vld1_s8(a);
}

// ALL-LABEL: @test_vld1_u16(
uint16x4_t test_vld1_u16(uint16_t const *a) {
// CIR: cir.load align(2) {{.*}} : !cir.ptr<!cir.vector<4 x !u16i>>, !cir.vector<4 x !u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x i16>, ptr [[A]], align 2
// LLVM: ret <4 x i16> [[TMP0]]
  return vld1_u16(a);
}

// ALL-LABEL: @test_vld1_u32(
uint32x2_t test_vld1_u32(uint32_t const *a) {
// CIR: cir.load align(4) {{.*}} : !cir.ptr<!cir.vector<2 x !u32i>>, !cir.vector<2 x !u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x i32>, ptr [[A]], align 4
// LLVM: ret <2 x i32> [[TMP0]]
  return vld1_u32(a);
}

// ALL-LABEL: @test_vld1_u64(
uint64x1_t test_vld1_u64(uint64_t const *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<1 x !u64i>>, !cir.vector<1 x !u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <1 x i64>, ptr [[A]], align 8
// LLVM: ret <1 x i64> [[TMP0]]
  return vld1_u64(a);
}

// ALL-LABEL: @test_vld1_u8(
uint8x8_t test_vld1_u8(uint8_t const *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<8 x !u8i>>, !cir.vector<8 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i8>, ptr [[A]], align 1
// LLVM: ret <8 x i8> [[TMP0]]
  return vld1_u8(a);
}

// Loading through a `void *` gives the vector no more than byte alignment.

// ALL-LABEL: @test_vld1_u8_void(
uint8x8_t test_vld1_u8_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<8 x !u8i>>, !cir.vector<8 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i8>, ptr [[A]], align 1
// LLVM: ret <8 x i8> [[TMP0]]
  return vld1_u8(a);
}

// ALL-LABEL: @test_vld1_u16_void(
uint16x4_t test_vld1_u16_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<4 x !u16i>>, !cir.vector<4 x !u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x i16>, ptr [[A]], align 1
// LLVM: ret <4 x i16> [[TMP0]]
  return vld1_u16(a);
}

// ALL-LABEL: @test_vld1_u32_void(
uint32x2_t test_vld1_u32_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<2 x !u32i>>, !cir.vector<2 x !u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x i32>, ptr [[A]], align 1
// LLVM: ret <2 x i32> [[TMP0]]
  return vld1_u32(a);
}

// ALL-LABEL: @test_vld1_u64_void(
uint64x1_t test_vld1_u64_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<1 x !u64i>>, !cir.vector<1 x !u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <1 x i64>, ptr [[A]], align 1
// LLVM: ret <1 x i64> [[TMP0]]
  return vld1_u64(a);
}

// ALL-LABEL: @test_vld1_s8_void(
int8x8_t test_vld1_s8_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i8>, ptr [[A]], align 1
// LLVM: ret <8 x i8> [[TMP0]]
  return vld1_s8(a);
}

// ALL-LABEL: @test_vld1_s16_void(
int16x4_t test_vld1_s16_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<4 x !s16i>>, !cir.vector<4 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x i16>, ptr [[A]], align 1
// LLVM: ret <4 x i16> [[TMP0]]
  return vld1_s16(a);
}

// ALL-LABEL: @test_vld1_s32_void(
int32x2_t test_vld1_s32_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<2 x !s32i>>, !cir.vector<2 x !s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x i32>, ptr [[A]], align 1
// LLVM: ret <2 x i32> [[TMP0]]
  return vld1_s32(a);
}

// ALL-LABEL: @test_vld1_s64_void(
int64x1_t test_vld1_s64_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<1 x !s64i>>, !cir.vector<1 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <1 x i64>, ptr [[A]], align 1
// LLVM: ret <1 x i64> [[TMP0]]
  return vld1_s64(a);
}

// ALL-LABEL: @test_vld1_f16_void(
float16x4_t test_vld1_f16_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<4 x !cir.f16>>, !cir.vector<4 x !cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x half>, ptr [[A]], align 1
// LLVM: ret <4 x half> [[TMP0]]
  return vld1_f16(a);
}

// ALL-LABEL: @test_vld1_f32_void(
float32x2_t test_vld1_f32_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<2 x !cir.float>>, !cir.vector<2 x !cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <2 x float>, ptr [[A]], align 1
// LLVM: ret <2 x float> [[TMP0]]
  return vld1_f32(a);
}

// ALL-LABEL: @test_vld1_f64_void(
float64x1_t test_vld1_f64_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<1 x !cir.double>>, !cir.vector<1 x !cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <1 x double>, ptr [[A]], align 1
// LLVM: ret <1 x double> [[TMP0]]
  return vld1_f64(a);
}

// ALL-LABEL: @test_vld1_p8_void(
poly8x8_t test_vld1_p8_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <8 x i8>, ptr [[A]], align 1
// LLVM: ret <8 x i8> [[TMP0]]
  return vld1_p8(a);
}

// ALL-LABEL: @test_vld1_p16_void(
poly16x4_t test_vld1_p16_void(void *a) {
// CIR: cir.load align(1) {{.*}} : !cir.ptr<!cir.vector<4 x !s16i>>, !cir.vector<4 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load <4 x i16>, ptr [[A]], align 1
// LLVM: ret <4 x i16> [[TMP0]]
  return vld1_p16(a);
}

// ALL-LABEL: @test_vld1q_f16_x2(
float16x8x2_t test_vld1q_f16_x2(float16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x2.v8f16.p0(ptr [[A]])
  return vld1q_f16_x2(a);
}

// ALL-LABEL: @test_vld1q_f32_x2(
float32x4x2_t test_vld1q_f32_x2(float32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x float>, <4 x float> } @llvm.aarch64.neon.ld1x2.v4f32.p0(ptr [[A]])
  return vld1q_f32_x2(a);
}

// ALL-LABEL: @test_vld1q_f64_x2(
float64x2x2_t test_vld1q_f64_x2(float64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x2.v2f64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double> } [[VLD1XN]], 1
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.float64x2x2_t poison, <2 x double> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.float64x2x2_t [[DOTFCA_0_0_INSERT]], <2 x double> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: ret %struct.float64x2x2_t [[DOTFCA_0_1_INSERT]]
  return vld1q_f64_x2(a);
}

// ALL-LABEL: @test_vld1q_mf8_x2(
mfloat8x16x2_t test_vld1q_mf8_x2(mfloat8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x2.v16i8.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8> } [[VLD1XN]], 1
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.mfloat8x16x2_t poison, <16 x i8> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.mfloat8x16x2_t [[DOTFCA_0_0_INSERT]], <16 x i8> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: ret %struct.mfloat8x16x2_t [[DOTFCA_0_1_INSERT]]
  return vld1q_mf8_x2(a);
}

// ALL-LABEL: @test_vld1q_p16_x2(
poly16x8x2_t test_vld1q_p16_x2(poly16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x2.v8i16.p0(ptr [[A]])
  return vld1q_p16_x2(a);
}

// ALL-LABEL: @test_vld1q_p64_x2(
poly64x2x2_t test_vld1q_p64_x2(poly64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x2.v2i64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64> } [[VLD1XN]], 1
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.poly64x2x2_t poison, <2 x i64> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.poly64x2x2_t [[DOTFCA_0_0_INSERT]], <2 x i64> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: ret %struct.poly64x2x2_t [[DOTFCA_0_1_INSERT]]
  return vld1q_p64_x2(a);
}

// ALL-LABEL: @test_vld1q_p8_x2(
poly8x16x2_t test_vld1q_p8_x2(poly8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x2.v16i8.p0(ptr [[A]])
  return vld1q_p8_x2(a);
}

// ALL-LABEL: @test_vld1q_s16_x2(
int16x8x2_t test_vld1q_s16_x2(int16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x2.v8i16.p0(ptr [[A]])
  return vld1q_s16_x2(a);
}

// ALL-LABEL: @test_vld1q_s32_x2(
int32x4x2_t test_vld1q_s32_x2(int32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld1x2.v4i32.p0(ptr [[A]])
  return vld1q_s32_x2(a);
}

// ALL-LABEL: @test_vld1q_s64_x2(
int64x2x2_t test_vld1q_s64_x2(int64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x2.v2i64.p0(ptr [[A]])
  return vld1q_s64_x2(a);
}

// ALL-LABEL: @test_vld1q_s8_x2(
int8x16x2_t test_vld1q_s8_x2(int8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x2.v16i8.p0(ptr [[A]])
  return vld1q_s8_x2(a);
}

// ALL-LABEL: @test_vld1q_u16_x2(
uint16x8x2_t test_vld1q_u16_x2(uint16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x2.v8i16.p0(ptr [[A]])
  return vld1q_u16_x2(a);
}

// ALL-LABEL: @test_vld1q_u32_x2(
uint32x4x2_t test_vld1q_u32_x2(uint32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld1x2.v4i32.p0(ptr [[A]])
  return vld1q_u32_x2(a);
}

// ALL-LABEL: @test_vld1q_u64_x2(
uint64x2x2_t test_vld1q_u64_x2(uint64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x2.v2i64.p0(ptr [[A]])
  return vld1q_u64_x2(a);
}

// ALL-LABEL: @test_vld1q_u8_x2(
uint8x16x2_t test_vld1q_u8_x2(uint8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x2.v16i8.p0(ptr [[A]])
  return vld1q_u8_x2(a);
}

// ALL-LABEL: @test_vld1_f16_x2(
float16x4x2_t test_vld1_f16_x2(float16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x2.v4f16.p0(ptr [[A]])
  return vld1_f16_x2(a);
}

// ALL-LABEL: @test_vld1_f32_x2(
float32x2x2_t test_vld1_f32_x2(float32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x float>, <2 x float> } @llvm.aarch64.neon.ld1x2.v2f32.p0(ptr [[A]])
  return vld1_f32_x2(a);
}

// ALL-LABEL: @test_vld1_f64_x2(
float64x1x2_t test_vld1_f64_x2(float64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x2.v1f64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double> } [[VLD1XN]], 1
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.float64x1x2_t poison, <1 x double> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.float64x1x2_t [[DOTFCA_0_0_INSERT]], <1 x double> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: ret %struct.float64x1x2_t [[DOTFCA_0_1_INSERT]]
  return vld1_f64_x2(a);
}

// ALL-LABEL: @test_vld1_mf8_x2(
mfloat8x8x2_t test_vld1_mf8_x2(mfloat8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x2.v8i8.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8> } [[VLD1XN]], 1
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.mfloat8x8x2_t poison, <8 x i8> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.mfloat8x8x2_t [[DOTFCA_0_0_INSERT]], <8 x i8> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: ret %struct.mfloat8x8x2_t [[DOTFCA_0_1_INSERT]]
  return vld1_mf8_x2(a);
}

// ALL-LABEL: @test_vld1_p16_x2(
poly16x4x2_t test_vld1_p16_x2(poly16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x2.v4i16.p0(ptr [[A]])
  return vld1_p16_x2(a);
}

// ALL-LABEL: @test_vld1_p64_x2(
poly64x1x2_t test_vld1_p64_x2(poly64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x2.v1i64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64> } [[VLD1XN]], 1
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.poly64x1x2_t poison, <1 x i64> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.poly64x1x2_t [[DOTFCA_0_0_INSERT]], <1 x i64> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: ret %struct.poly64x1x2_t [[DOTFCA_0_1_INSERT]]
  return vld1_p64_x2(a);
}

// ALL-LABEL: @test_vld1_p8_x2(
poly8x8x2_t test_vld1_p8_x2(poly8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x2.v8i8.p0(ptr [[A]])
  return vld1_p8_x2(a);
}

// ALL-LABEL: @test_vld1_s16_x2(
int16x4x2_t test_vld1_s16_x2(int16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x2.v4i16.p0(ptr [[A]])
  return vld1_s16_x2(a);
}

// ALL-LABEL: @test_vld1_s32_x2(
int32x2x2_t test_vld1_s32_x2(int32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld1x2.v2i32.p0(ptr [[A]])
  return vld1_s32_x2(a);
}

// ALL-LABEL: @test_vld1_s64_x2(
int64x1x2_t test_vld1_s64_x2(int64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x2.v1i64.p0(ptr [[A]])
  return vld1_s64_x2(a);
}

// ALL-LABEL: @test_vld1_s8_x2(
int8x8x2_t test_vld1_s8_x2(int8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x2.v8i8.p0(ptr [[A]])
  return vld1_s8_x2(a);
}

// ALL-LABEL: @test_vld1_u16_x2(
uint16x4x2_t test_vld1_u16_x2(uint16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x2.v4i16.p0(ptr [[A]])
  return vld1_u16_x2(a);
}

// ALL-LABEL: @test_vld1_u32_x2(
uint32x2x2_t test_vld1_u32_x2(uint32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld1x2.v2i32.p0(ptr [[A]])
  return vld1_u32_x2(a);
}

// ALL-LABEL: @test_vld1_u64_x2(
uint64x1x2_t test_vld1_u64_x2(uint64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x2.v1i64.p0(ptr [[A]])
  return vld1_u64_x2(a);
}

// ALL-LABEL: @test_vld1_u8_x2(
uint8x8x2_t test_vld1_u8_x2(uint8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x2" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x2.v8i8.p0(ptr [[A]])
  return vld1_u8_x2(a);
}

// ALL-LABEL: @test_vld1q_f16_x3(
float16x8x3_t test_vld1q_f16_x3(float16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x3.v8f16.p0(ptr [[A]])
  return vld1q_f16_x3(a);
}

// ALL-LABEL: @test_vld1q_f32_x3(
float32x4x3_t test_vld1q_f32_x3(float32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld1x3.v4f32.p0(ptr [[A]])
  return vld1q_f32_x3(a);
}

// ALL-LABEL: @test_vld1q_f64_x3(
float64x2x3_t test_vld1q_f64_x3(float64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x3.v2f64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], 2
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.float64x2x3_t poison, <2 x double> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.float64x2x3_t [[DOTFCA_0_0_INSERT]], <2 x double> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.float64x2x3_t [[DOTFCA_0_1_INSERT]], <2 x double> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: ret %struct.float64x2x3_t [[DOTFCA_0_2_INSERT]]
  return vld1q_f64_x3(a);
}

// ALL-LABEL: @test_vld1q_mf8_x3(
mfloat8x16x3_t test_vld1q_mf8_x3(mfloat8_t const *ptr) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[PTR:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x3.v16i8.p0(ptr [[PTR]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], 2
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.mfloat8x16x3_t poison, <16 x i8> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.mfloat8x16x3_t [[DOTFCA_0_0_INSERT]], <16 x i8> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.mfloat8x16x3_t [[DOTFCA_0_1_INSERT]], <16 x i8> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: ret %struct.mfloat8x16x3_t [[DOTFCA_0_2_INSERT]]
  return vld1q_mf8_x3(ptr);
}

// ALL-LABEL: @test_vld1q_p16_x3(
poly16x8x3_t test_vld1q_p16_x3(poly16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x3.v8i16.p0(ptr [[A]])
  return vld1q_p16_x3(a);
}

// ALL-LABEL: @test_vld1q_p64_x3(
poly64x2x3_t test_vld1q_p64_x3(poly64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x3.v2i64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], 2
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.poly64x2x3_t poison, <2 x i64> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.poly64x2x3_t [[DOTFCA_0_0_INSERT]], <2 x i64> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.poly64x2x3_t [[DOTFCA_0_1_INSERT]], <2 x i64> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: ret %struct.poly64x2x3_t [[DOTFCA_0_2_INSERT]]
  return vld1q_p64_x3(a);
}

// ALL-LABEL: @test_vld1q_p8_x3(
poly8x16x3_t test_vld1q_p8_x3(poly8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x3.v16i8.p0(ptr [[A]])
  return vld1q_p8_x3(a);
}

// ALL-LABEL: @test_vld1q_s16_x3(
int16x8x3_t test_vld1q_s16_x3(int16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x3.v8i16.p0(ptr [[A]])
  return vld1q_s16_x3(a);
}

// ALL-LABEL: @test_vld1q_s32_x3(
int32x4x3_t test_vld1q_s32_x3(int32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld1x3.v4i32.p0(ptr [[A]])
  return vld1q_s32_x3(a);
}

// ALL-LABEL: @test_vld1q_s64_x3(
int64x2x3_t test_vld1q_s64_x3(int64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x3.v2i64.p0(ptr [[A]])
  return vld1q_s64_x3(a);
}

// ALL-LABEL: @test_vld1q_s8_x3(
int8x16x3_t test_vld1q_s8_x3(int8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x3.v16i8.p0(ptr [[A]])
  return vld1q_s8_x3(a);
}

// ALL-LABEL: @test_vld1q_u16_x3(
uint16x8x3_t test_vld1q_u16_x3(uint16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x3.v8i16.p0(ptr [[A]])
  return vld1q_u16_x3(a);
}

// ALL-LABEL: @test_vld1q_u32_x3(
uint32x4x3_t test_vld1q_u32_x3(uint32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld1x3.v4i32.p0(ptr [[A]])
  return vld1q_u32_x3(a);
}

// ALL-LABEL: @test_vld1q_u64_x3(
uint64x2x3_t test_vld1q_u64_x3(uint64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x3.v2i64.p0(ptr [[A]])
  return vld1q_u64_x3(a);
}

// ALL-LABEL: @test_vld1q_u8_x3(
uint8x16x3_t test_vld1q_u8_x3(uint8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x3.v16i8.p0(ptr [[A]])
  return vld1q_u8_x3(a);
}

// ALL-LABEL: @test_vld1_f16_x3(
float16x4x3_t test_vld1_f16_x3(float16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x3.v4f16.p0(ptr [[A]])
  return vld1_f16_x3(a);
}

// ALL-LABEL: @test_vld1_f32_x3(
float32x2x3_t test_vld1_f32_x3(float32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld1x3.v2f32.p0(ptr [[A]])
  return vld1_f32_x3(a);
}

// ALL-LABEL: @test_vld1_f64_x3(
float64x1x3_t test_vld1_f64_x3(float64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x3.v1f64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], 2
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.float64x1x3_t poison, <1 x double> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.float64x1x3_t [[DOTFCA_0_0_INSERT]], <1 x double> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.float64x1x3_t [[DOTFCA_0_1_INSERT]], <1 x double> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: ret %struct.float64x1x3_t [[DOTFCA_0_2_INSERT]]
  return vld1_f64_x3(a);
}

// ALL-LABEL: @test_vld1_mf8_x3(
mfloat8x8x3_t test_vld1_mf8_x3(mfloat8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x3.v8i8.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], 2
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.mfloat8x8x3_t poison, <8 x i8> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.mfloat8x8x3_t [[DOTFCA_0_0_INSERT]], <8 x i8> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.mfloat8x8x3_t [[DOTFCA_0_1_INSERT]], <8 x i8> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: ret %struct.mfloat8x8x3_t [[DOTFCA_0_2_INSERT]]
  return vld1_mf8_x3(a);
}

// ALL-LABEL: @test_vld1_p16_x3(
poly16x4x3_t test_vld1_p16_x3(poly16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x3.v4i16.p0(ptr [[A]])
  return vld1_p16_x3(a);
}

// ALL-LABEL: @test_vld1_p64_x3(
poly64x1x3_t test_vld1_p64_x3(poly64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x3.v1i64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], 2
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.poly64x1x3_t poison, <1 x i64> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.poly64x1x3_t [[DOTFCA_0_0_INSERT]], <1 x i64> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.poly64x1x3_t [[DOTFCA_0_1_INSERT]], <1 x i64> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: ret %struct.poly64x1x3_t [[DOTFCA_0_2_INSERT]]
  return vld1_p64_x3(a);
}

// ALL-LABEL: @test_vld1_p8_x3(
poly8x8x3_t test_vld1_p8_x3(poly8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x3.v8i8.p0(ptr [[A]])
  return vld1_p8_x3(a);
}

// ALL-LABEL: @test_vld1_s16_x3(
int16x4x3_t test_vld1_s16_x3(int16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x3.v4i16.p0(ptr [[A]])
  return vld1_s16_x3(a);
}

// ALL-LABEL: @test_vld1_s32_x3(
int32x2x3_t test_vld1_s32_x3(int32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld1x3.v2i32.p0(ptr [[A]])
  return vld1_s32_x3(a);
}

// ALL-LABEL: @test_vld1_s64_x3(
int64x1x3_t test_vld1_s64_x3(int64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x3.v1i64.p0(ptr [[A]])
  return vld1_s64_x3(a);
}

// ALL-LABEL: @test_vld1_s8_x3(
int8x8x3_t test_vld1_s8_x3(int8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x3.v8i8.p0(ptr [[A]])
  return vld1_s8_x3(a);
}

// ALL-LABEL: @test_vld1_u16_x3(
uint16x4x3_t test_vld1_u16_x3(uint16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x3.v4i16.p0(ptr [[A]])
  return vld1_u16_x3(a);
}

// ALL-LABEL: @test_vld1_u32_x3(
uint32x2x3_t test_vld1_u32_x3(uint32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld1x3.v2i32.p0(ptr [[A]])
  return vld1_u32_x3(a);
}

// ALL-LABEL: @test_vld1_u64_x3(
uint64x1x3_t test_vld1_u64_x3(uint64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x3.v1i64.p0(ptr [[A]])
  return vld1_u64_x3(a);
}

// ALL-LABEL: @test_vld1_u8_x3(
uint8x8x3_t test_vld1_u8_x3(uint8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x3" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x3.v8i8.p0(ptr [[A]])
  return vld1_u8_x3(a);
}

// ALL-LABEL: @test_vld1q_f16_x4(
float16x8x4_t test_vld1q_f16_x4(float16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x4.v8f16.p0(ptr [[A]])
  return vld1q_f16_x4(a);
}

// ALL-LABEL: @test_vld1q_f32_x4(
float32x4x4_t test_vld1q_f32_x4(float32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld1x4.v4f32.p0(ptr [[A]])
  return vld1q_f32_x4(a);
}

// ALL-LABEL: @test_vld1q_f64_x4(
float64x2x4_t test_vld1q_f64_x4(float64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x4.v2f64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], 2
// LLVM-DAG: [[VLD1XN_FCA_3_EXTRACT:%.*]] = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], 3
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.float64x2x4_t poison, <2 x double> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.float64x2x4_t [[DOTFCA_0_0_INSERT]], <2 x double> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.float64x2x4_t [[DOTFCA_0_1_INSERT]], <2 x double> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: [[DOTFCA_0_3_INSERT:%.*]] = insertvalue %struct.float64x2x4_t [[DOTFCA_0_2_INSERT]], <2 x double> [[VLD1XN_FCA_3_EXTRACT]], 0, 3
// LLVM: ret %struct.float64x2x4_t [[DOTFCA_0_3_INSERT]]
  return vld1q_f64_x4(a);
}

// ALL-LABEL: @test_vld1q_mf8_x4(
mfloat8x16x4_t test_vld1q_mf8_x4(mfloat8_t const *ptr) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[PTR:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x4.v16i8.p0(ptr [[PTR]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], 2
// LLVM-DAG: [[VLD1XN_FCA_3_EXTRACT:%.*]] = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], 3
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.mfloat8x16x4_t poison, <16 x i8> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.mfloat8x16x4_t [[DOTFCA_0_0_INSERT]], <16 x i8> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.mfloat8x16x4_t [[DOTFCA_0_1_INSERT]], <16 x i8> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: [[DOTFCA_0_3_INSERT:%.*]] = insertvalue %struct.mfloat8x16x4_t [[DOTFCA_0_2_INSERT]], <16 x i8> [[VLD1XN_FCA_3_EXTRACT]], 0, 3
// LLVM: ret %struct.mfloat8x16x4_t [[DOTFCA_0_3_INSERT]]
  return vld1q_mf8_x4(ptr);
}

// ALL-LABEL: @test_vld1q_p16_x4(
poly16x8x4_t test_vld1q_p16_x4(poly16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x4.v8i16.p0(ptr [[A]])
  return vld1q_p16_x4(a);
}

// ALL-LABEL: @test_vld1q_p64_x4(
poly64x2x4_t test_vld1q_p64_x4(poly64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x4.v2i64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], 2
// LLVM-DAG: [[VLD1XN_FCA_3_EXTRACT:%.*]] = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], 3
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.poly64x2x4_t poison, <2 x i64> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.poly64x2x4_t [[DOTFCA_0_0_INSERT]], <2 x i64> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.poly64x2x4_t [[DOTFCA_0_1_INSERT]], <2 x i64> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: [[DOTFCA_0_3_INSERT:%.*]] = insertvalue %struct.poly64x2x4_t [[DOTFCA_0_2_INSERT]], <2 x i64> [[VLD1XN_FCA_3_EXTRACT]], 0, 3
// LLVM: ret %struct.poly64x2x4_t [[DOTFCA_0_3_INSERT]]
  return vld1q_p64_x4(a);
}

// ALL-LABEL: @test_vld1q_p8_x4(
poly8x16x4_t test_vld1q_p8_x4(poly8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x4.v16i8.p0(ptr [[A]])
  return vld1q_p8_x4(a);
}

// ALL-LABEL: @test_vld1q_s16_x4(
int16x8x4_t test_vld1q_s16_x4(int16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x4.v8i16.p0(ptr [[A]])
  return vld1q_s16_x4(a);
}

// ALL-LABEL: @test_vld1q_s32_x4(
int32x4x4_t test_vld1q_s32_x4(int32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld1x4.v4i32.p0(ptr [[A]])
  return vld1q_s32_x4(a);
}

// ALL-LABEL: @test_vld1q_s64_x4(
int64x2x4_t test_vld1q_s64_x4(int64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x4.v2i64.p0(ptr [[A]])
  return vld1q_s64_x4(a);
}

// ALL-LABEL: @test_vld1q_s8_x4(
int8x16x4_t test_vld1q_s8_x4(int8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x4.v16i8.p0(ptr [[A]])
  return vld1q_s8_x4(a);
}

// ALL-LABEL: @test_vld1q_u16_x4(
uint16x8x4_t test_vld1q_u16_x4(uint16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld1x4.v8i16.p0(ptr [[A]])
  return vld1q_u16_x4(a);
}

// ALL-LABEL: @test_vld1q_u32_x4(
uint32x4x4_t test_vld1q_u32_x4(uint32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld1x4.v4i32.p0(ptr [[A]])
  return vld1q_u32_x4(a);
}

// ALL-LABEL: @test_vld1q_u64_x4(
uint64x2x4_t test_vld1q_u64_x4(uint64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x4.v2i64.p0(ptr [[A]])
  return vld1q_u64_x4(a);
}

// ALL-LABEL: @test_vld1q_u8_x4(
uint8x16x4_t test_vld1q_u8_x4(uint8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(16) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld1x4.v16i8.p0(ptr [[A]])
  return vld1q_u8_x4(a);
}

// ALL-LABEL: @test_vld1_f16_x4(
float16x4x4_t test_vld1_f16_x4(float16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x4.v4f16.p0(ptr [[A]])
  return vld1_f16_x4(a);
}

// ALL-LABEL: @test_vld1_f32_x4(
float32x2x4_t test_vld1_f32_x4(float32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld1x4.v2f32.p0(ptr [[A]])
  return vld1_f32_x4(a);
}

// ALL-LABEL: @test_vld1_f64_x4(
float64x1x4_t test_vld1_f64_x4(float64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x4.v1f64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], 2
// LLVM-DAG: [[VLD1XN_FCA_3_EXTRACT:%.*]] = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], 3
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.float64x1x4_t poison, <1 x double> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.float64x1x4_t [[DOTFCA_0_0_INSERT]], <1 x double> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.float64x1x4_t [[DOTFCA_0_1_INSERT]], <1 x double> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: [[DOTFCA_0_3_INSERT:%.*]] = insertvalue %struct.float64x1x4_t [[DOTFCA_0_2_INSERT]], <1 x double> [[VLD1XN_FCA_3_EXTRACT]], 0, 3
// LLVM: ret %struct.float64x1x4_t [[DOTFCA_0_3_INSERT]]
  return vld1_f64_x4(a);
}

// ALL-LABEL: @test_vld1_mf8_x4(
mfloat8x8x4_t test_vld1_mf8_x4(mfloat8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x4.v8i8.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], 2
// LLVM-DAG: [[VLD1XN_FCA_3_EXTRACT:%.*]] = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], 3
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.mfloat8x8x4_t poison, <8 x i8> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.mfloat8x8x4_t [[DOTFCA_0_0_INSERT]], <8 x i8> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.mfloat8x8x4_t [[DOTFCA_0_1_INSERT]], <8 x i8> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: [[DOTFCA_0_3_INSERT:%.*]] = insertvalue %struct.mfloat8x8x4_t [[DOTFCA_0_2_INSERT]], <8 x i8> [[VLD1XN_FCA_3_EXTRACT]], 0, 3
// LLVM: ret %struct.mfloat8x8x4_t [[DOTFCA_0_3_INSERT]]
  return vld1_mf8_x4(a);
}

// ALL-LABEL: @test_vld1_p16_x4(
poly16x4x4_t test_vld1_p16_x4(poly16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x4.v4i16.p0(ptr [[A]])
  return vld1_p16_x4(a);
}

// ALL-LABEL: @test_vld1_p64_x4(
poly64x1x4_t test_vld1_p64_x4(poly64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x4.v1i64.p0(ptr [[A]])
// LLVM-DAG: [[VLD1XN_FCA_0_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], 0
// LLVM-DAG: [[VLD1XN_FCA_1_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], 1
// LLVM-DAG: [[VLD1XN_FCA_2_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], 2
// LLVM-DAG: [[VLD1XN_FCA_3_EXTRACT:%.*]] = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], 3
// LLVM: [[DOTFCA_0_0_INSERT:%.*]] = insertvalue %struct.poly64x1x4_t poison, <1 x i64> [[VLD1XN_FCA_0_EXTRACT]], 0, 0
// LLVM: [[DOTFCA_0_1_INSERT:%.*]] = insertvalue %struct.poly64x1x4_t [[DOTFCA_0_0_INSERT]], <1 x i64> [[VLD1XN_FCA_1_EXTRACT]], 0, 1
// LLVM: [[DOTFCA_0_2_INSERT:%.*]] = insertvalue %struct.poly64x1x4_t [[DOTFCA_0_1_INSERT]], <1 x i64> [[VLD1XN_FCA_2_EXTRACT]], 0, 2
// LLVM: [[DOTFCA_0_3_INSERT:%.*]] = insertvalue %struct.poly64x1x4_t [[DOTFCA_0_2_INSERT]], <1 x i64> [[VLD1XN_FCA_3_EXTRACT]], 0, 3
// LLVM: ret %struct.poly64x1x4_t [[DOTFCA_0_3_INSERT]]
  return vld1_p64_x4(a);
}

// ALL-LABEL: @test_vld1_p8_x4(
poly8x8x4_t test_vld1_p8_x4(poly8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x4.v8i8.p0(ptr [[A]])
  return vld1_p8_x4(a);
}

// ALL-LABEL: @test_vld1_s16_x4(
int16x4x4_t test_vld1_s16_x4(int16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x4.v4i16.p0(ptr [[A]])
  return vld1_s16_x4(a);
}

// ALL-LABEL: @test_vld1_s32_x4(
int32x2x4_t test_vld1_s32_x4(int32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld1x4.v2i32.p0(ptr [[A]])
  return vld1_s32_x4(a);
}

// ALL-LABEL: @test_vld1_s64_x4(
int64x1x4_t test_vld1_s64_x4(int64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x4.v1i64.p0(ptr [[A]])
  return vld1_s64_x4(a);
}

// ALL-LABEL: @test_vld1_s8_x4(
int8x8x4_t test_vld1_s8_x4(int8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x4.v8i8.p0(ptr [[A]])
  return vld1_s8_x4(a);
}

// ALL-LABEL: @test_vld1_u16_x4(
uint16x4x4_t test_vld1_u16_x4(uint16_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x4.v4i16.p0(ptr [[A]])
  return vld1_u16_x4(a);
}

// ALL-LABEL: @test_vld1_u32_x4(
uint32x2x4_t test_vld1_u32_x4(uint32_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld1x4.v2i32.p0(ptr [[A]])
  return vld1_u32_x4(a);
}

// ALL-LABEL: @test_vld1_u64_x4(
uint64x1x4_t test_vld1_u64_x4(uint64_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x4.v1i64.p0(ptr [[A]])
  return vld1_u64_x4(a);
}

// ALL-LABEL: @test_vld1_u8_x4(
uint8x8x4_t test_vld1_u8_x4(uint8_t const *a) {
// CIR: [[VLD:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.ld1x4" {{.*}} : (!cir.ptr<!void>) -> [[STY:!rec_anon_struct[0-9]*]]
// CIR: cir.store align(8) [[VLD]], {{.*}} : [[STY]], !cir.ptr<[[STY]]>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld1x4.v8i8.p0(ptr [[A]])
  return vld1_u8_x4(a);
}

// ALL-LABEL: @test_vld1q_lane_f16(
float16x8_t test_vld1q_lane_f16(float16_t  *a, float16x8_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!cir.f16>, !cir
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<8 x !cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load half, ptr [[A]], align 2
// LLVM: [[VLD1_LANE:%.*]] = insertelement <8 x half> [[B]], half [[TMP0]], i64 7
// LLVM: ret <8 x half> [[VLD1_LANE]]
  return vld1q_lane_f16(a, b, 7);
}

// ALL-LABEL: @test_vld1q_lane_f32(
float32x4_t test_vld1q_lane_f32(float32_t  *a, float32x4_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!cir.float>, !cir
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<4 x !cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load float, ptr [[A]], align 4
// LLVM: [[VLD1_LANE:%.*]] = insertelement <4 x float> [[B]], float [[TMP0]], i64 3
// LLVM: ret <4 x float> [[VLD1_LANE]]
  return vld1q_lane_f32(a, b, 3);
}

// ALL-LABEL: @test_vld1q_lane_f64(
float64x2_t test_vld1q_lane_f64(float64_t  *a, float64x2_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.double>, !cir
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<2 x !cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load double, ptr [[A]], align 8
// LLVM: [[VLD1_LANE:%.*]] = insertelement <2 x double> [[B]], double [[TMP0]], i64 1
// LLVM: ret <2 x double> [[VLD1_LANE]]
  return vld1q_lane_f64(a, b, 1);
}

// ALL-LABEL: @test_vld1q_lane_mf8(
mfloat8x16_t test_vld1q_lane_mf8(mfloat8_t  *a, mfloat8x16_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<16 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[VLD1_LANE:%.*]] = insertelement <16 x i8> [[B]], i8 [[TMP0]], i64 15
// LLVM: ret <16 x i8> [[VLD1_LANE]]
  return vld1q_lane_mf8(a, b, 15);
}

// ALL-LABEL: @test_vld1q_lane_p16(
poly16x8_t test_vld1q_lane_p16(poly16_t  *a, poly16x8_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<8 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[VLD1_LANE:%.*]] = insertelement <8 x i16> [[B]], i16 [[TMP0]], i64 7
// LLVM: ret <8 x i16> [[VLD1_LANE]]
  return vld1q_lane_p16(a, b, 7);
}

// ALL-LABEL: @test_vld1q_lane_p64(
poly64x2_t test_vld1q_lane_p64(poly64_t  *a, poly64x2_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<2 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[VLD1_LANE:%.*]] = insertelement <2 x i64> [[B]], i64 [[TMP0]], i64 1
// LLVM: ret <2 x i64> [[VLD1_LANE]]
  return vld1q_lane_p64(a, b, 1);
}

// ALL-LABEL: @test_vld1q_lane_p8(
poly8x16_t test_vld1q_lane_p8(poly8_t  *a, poly8x16_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<16 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[VLD1_LANE:%.*]] = insertelement <16 x i8> [[B]], i8 [[TMP0]], i64 15
// LLVM: ret <16 x i8> [[VLD1_LANE]]
  return vld1q_lane_p8(a, b, 15);
}

// ALL-LABEL: @test_vld1q_lane_s16(
int16x8_t test_vld1q_lane_s16(int16_t  *a, int16x8_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<8 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[VLD1_LANE:%.*]] = insertelement <8 x i16> [[B]], i16 [[TMP0]], i64 7
// LLVM: ret <8 x i16> [[VLD1_LANE]]
  return vld1q_lane_s16(a, b, 7);
}

// ALL-LABEL: @test_vld1q_lane_s32(
int32x4_t test_vld1q_lane_s32(int32_t  *a, int32x4_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<4 x !s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i32, ptr [[A]], align 4
// LLVM: [[VLD1_LANE:%.*]] = insertelement <4 x i32> [[B]], i32 [[TMP0]], i64 3
// LLVM: ret <4 x i32> [[VLD1_LANE]]
  return vld1q_lane_s32(a, b, 3);
}

// ALL-LABEL: @test_vld1q_lane_s64(
int64x2_t test_vld1q_lane_s64(int64_t  *a, int64x2_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<2 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[VLD1_LANE:%.*]] = insertelement <2 x i64> [[B]], i64 [[TMP0]], i64 1
// LLVM: ret <2 x i64> [[VLD1_LANE]]
  return vld1q_lane_s64(a, b, 1);
}

// ALL-LABEL: @test_vld1q_lane_s8(
int8x16_t test_vld1q_lane_s8(int8_t  *a, int8x16_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<16 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[VLD1_LANE:%.*]] = insertelement <16 x i8> [[B]], i8 [[TMP0]], i64 15
// LLVM: ret <16 x i8> [[VLD1_LANE]]
  return vld1q_lane_s8(a, b, 15);
}

// ALL-LABEL: @test_vld1q_lane_u16(
uint16x8_t test_vld1q_lane_u16(uint16_t  *a, uint16x8_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!u16i>, !u16i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<8 x !u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[VLD1_LANE:%.*]] = insertelement <8 x i16> [[B]], i16 [[TMP0]], i64 7
// LLVM: ret <8 x i16> [[VLD1_LANE]]
  return vld1q_lane_u16(a, b, 7);
}

// ALL-LABEL: @test_vld1q_lane_u32(
uint32x4_t test_vld1q_lane_u32(uint32_t  *a, uint32x4_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<4 x !u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i32, ptr [[A]], align 4
// LLVM: [[VLD1_LANE:%.*]] = insertelement <4 x i32> [[B]], i32 [[TMP0]], i64 3
// LLVM: ret <4 x i32> [[VLD1_LANE]]
  return vld1q_lane_u32(a, b, 3);
}

// ALL-LABEL: @test_vld1q_lane_u64(
uint64x2_t test_vld1q_lane_u64(uint64_t  *a, uint64x2_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!u64i>, !u64i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<2 x !u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[VLD1_LANE:%.*]] = insertelement <2 x i64> [[B]], i64 [[TMP0]], i64 1
// LLVM: ret <2 x i64> [[VLD1_LANE]]
  return vld1q_lane_u64(a, b, 1);
}

// ALL-LABEL: @test_vld1q_lane_u8(
uint8x16_t test_vld1q_lane_u8(uint8_t  *a, uint8x16_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<16 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[VLD1_LANE:%.*]] = insertelement <16 x i8> [[B]], i8 [[TMP0]], i64 15
// LLVM: ret <16 x i8> [[VLD1_LANE]]
  return vld1q_lane_u8(a, b, 15);
}

// ALL-LABEL: @test_vld1_lane_f16(
float16x4_t test_vld1_lane_f16(float16_t  *a, float16x4_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!cir.f16>, !cir
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<4 x !cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load half, ptr [[A]], align 2
// LLVM: [[VLD1_LANE:%.*]] = insertelement <4 x half> [[B]], half [[TMP0]], i64 3
// LLVM: ret <4 x half> [[VLD1_LANE]]
  return vld1_lane_f16(a, b, 3);
}

// ALL-LABEL: @test_vld1_lane_f32(
float32x2_t test_vld1_lane_f32(float32_t  *a, float32x2_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!cir.float>, !cir
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<2 x !cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load float, ptr [[A]], align 4
// LLVM: [[VLD1_LANE:%.*]] = insertelement <2 x float> [[B]], float [[TMP0]], i64 1
// LLVM: ret <2 x float> [[VLD1_LANE]]
  return vld1_lane_f32(a, b, 1);
}

// ALL-LABEL: @test_vld1_lane_f64(
float64x1_t test_vld1_lane_f64(float64_t  *a, float64x1_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<0> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.double>, !cir
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<1 x !cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load double, ptr [[A]], align 8
// LLVM: [[VLD1_LANE:%.*]] = insertelement <1 x double> poison, double [[TMP0]], i64 0
// LLVM: ret <1 x double> [[VLD1_LANE]]
  return vld1_lane_f64(a, b, 0);
}

// ALL-LABEL: @test_vld1_lane_mf8(
mfloat8x8_t test_vld1_lane_mf8(mfloat8_t  *a, mfloat8x8_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<8 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[VLD1_LANE:%.*]] = insertelement <8 x i8> [[B]], i8 [[TMP0]], i64 7
// LLVM: ret <8 x i8> [[VLD1_LANE]]
  return vld1_lane_mf8(a, b, 7);
}

// ALL-LABEL: @test_vld1_lane_p16(
poly16x4_t test_vld1_lane_p16(poly16_t  *a, poly16x4_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<4 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[VLD1_LANE:%.*]] = insertelement <4 x i16> [[B]], i16 [[TMP0]], i64 3
// LLVM: ret <4 x i16> [[VLD1_LANE]]
  return vld1_lane_p16(a, b, 3);
}

// ALL-LABEL: @test_vld1_lane_p64(
poly64x1_t test_vld1_lane_p64(poly64_t  *a, poly64x1_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<0> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<1 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[VLD1_LANE:%.*]] = insertelement <1 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: ret <1 x i64> [[VLD1_LANE]]
  return vld1_lane_p64(a, b, 0);
}

// ALL-LABEL: @test_vld1_lane_p8(
poly8x8_t test_vld1_lane_p8(poly8_t  *a, poly8x8_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<8 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[VLD1_LANE:%.*]] = insertelement <8 x i8> [[B]], i8 [[TMP0]], i64 7
// LLVM: ret <8 x i8> [[VLD1_LANE]]
  return vld1_lane_p8(a, b, 7);
}

// ALL-LABEL: @test_vld1_lane_s16(
int16x4_t test_vld1_lane_s16(int16_t  *a, int16x4_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<4 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[VLD1_LANE:%.*]] = insertelement <4 x i16> [[B]], i16 [[TMP0]], i64 3
// LLVM: ret <4 x i16> [[VLD1_LANE]]
  return vld1_lane_s16(a, b, 3);
}

// ALL-LABEL: @test_vld1_lane_s32(
int32x2_t test_vld1_lane_s32(int32_t  *a, int32x2_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<2 x !s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i32, ptr [[A]], align 4
// LLVM: [[VLD1_LANE:%.*]] = insertelement <2 x i32> [[B]], i32 [[TMP0]], i64 1
// LLVM: ret <2 x i32> [[VLD1_LANE]]
  return vld1_lane_s32(a, b, 1);
}

// ALL-LABEL: @test_vld1_lane_s64(
int64x1_t test_vld1_lane_s64(int64_t  *a, int64x1_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<0> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<1 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[VLD1_LANE:%.*]] = insertelement <1 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: ret <1 x i64> [[VLD1_LANE]]
  return vld1_lane_s64(a, b, 0);
}

// ALL-LABEL: @test_vld1_lane_s8(
int8x8_t test_vld1_lane_s8(int8_t  *a, int8x8_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<8 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[VLD1_LANE:%.*]] = insertelement <8 x i8> [[B]], i8 [[TMP0]], i64 7
// LLVM: ret <8 x i8> [[VLD1_LANE]]
  return vld1_lane_s8(a, b, 7);
}

// ALL-LABEL: @test_vld1_lane_u16(
uint16x4_t test_vld1_lane_u16(uint16_t  *a, uint16x4_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!u16i>, !u16i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<4 x !u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[VLD1_LANE:%.*]] = insertelement <4 x i16> [[B]], i16 [[TMP0]], i64 3
// LLVM: ret <4 x i16> [[VLD1_LANE]]
  return vld1_lane_u16(a, b, 3);
}

// ALL-LABEL: @test_vld1_lane_u32(
uint32x2_t test_vld1_lane_u32(uint32_t  *a, uint32x2_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<2 x !u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i32, ptr [[A]], align 4
// LLVM: [[VLD1_LANE:%.*]] = insertelement <2 x i32> [[B]], i32 [[TMP0]], i64 1
// LLVM: ret <2 x i32> [[VLD1_LANE]]
  return vld1_lane_u32(a, b, 1);
}

// ALL-LABEL: @test_vld1_lane_u64(
uint64x1_t test_vld1_lane_u64(uint64_t  *a, uint64x1_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<0> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!u64i>, !u64i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<1 x !u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[VLD1_LANE:%.*]] = insertelement <1 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: ret <1 x i64> [[VLD1_LANE]]
  return vld1_lane_u64(a, b, 0);
}

// ALL-LABEL: @test_vld1_lane_u8(
uint8x8_t test_vld1_lane_u8(uint8_t  *a, uint8x8_t b) {
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.vec.insert [[SCALAR]], {{.*}}[[IDX]] : !s32i] : !cir.vector<8 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[VLD1_LANE:%.*]] = insertelement <8 x i8> [[B]], i8 [[TMP0]], i64 7
// LLVM: ret <8 x i8> [[VLD1_LANE]]
  return vld1_lane_u8(a, b, 7);
}

// ALL-LABEL: @test_vld1q_dup_f16(
float16x8_t test_vld1q_dup_f16(float16_t  *a) {
// CIR: cir.load align(16) {{.*}} : !cir.ptr<!cir.vector<8 x !cir.f16>>, !cir.vector<8 x !cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load half, ptr [[A]], align 2
// LLVM: [[TMP1:%.*]] = insertelement <8 x half> poison, half [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <8 x half> [[TMP1]], <8 x half> poison, <8 x i32> zeroinitializer
// LLVM: ret <8 x half> [[LANE]]
  return vld1q_dup_f16(a);
}

// ALL-LABEL: @test_vld1q_dup_f32(
float32x4_t test_vld1q_dup_f32(float32_t  *a) {
// CIR: cir.load align(16) {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load float, ptr [[A]], align 4
// LLVM: [[TMP1:%.*]] = insertelement <4 x float> poison, float [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <4 x float> [[TMP1]], <4 x float> poison, <4 x i32> zeroinitializer
// LLVM: ret <4 x float> [[LANE]]
  return vld1q_dup_f32(a);
}

// ALL-LABEL: @test_vld1q_dup_f64(
float64x2_t test_vld1q_dup_f64(float64_t  *a) {
// CIR: cir.load align(16) {{.*}} : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load double, ptr [[A]], align 8
// LLVM: [[TMP1:%.*]] = insertelement <2 x double> poison, double [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <2 x double> [[TMP1]], <2 x double> poison, <2 x i32> zeroinitializer
// LLVM: ret <2 x double> [[LANE]]
  return vld1q_dup_f64(a);
}

// ALL-LABEL: @test_vld1q_dup_mf8(
mfloat8x16_t test_vld1q_dup_mf8(mfloat8_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.vec.splat [[SCALAR]] : !u8i, !cir.vector<16 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[TMP1:%.*]] = insertelement <16 x i8> poison, i8 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <16 x i8> [[TMP1]], <16 x i8> poison, <16 x i32> zeroinitializer
// LLVM: ret <16 x i8> [[LANE]]
  return vld1q_dup_mf8(a);
}

// ALL-LABEL: @test_vld1q_dup_p16(
poly16x8_t test_vld1q_dup_p16(poly16_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.vec.splat [[SCALAR]] : !s16i, !cir.vector<8 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[TMP1:%.*]] = insertelement <8 x i16> poison, i16 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <8 x i16> [[TMP1]], <8 x i16> poison, <8 x i32> zeroinitializer
// LLVM: ret <8 x i16> [[LANE]]
  return vld1q_dup_p16(a);
}

// ALL-LABEL: @test_vld1q_dup_p64(
poly64x2_t test_vld1q_dup_p64(poly64_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.vec.splat [[SCALAR]] : !s64i, !cir.vector<2 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[TMP1:%.*]] = insertelement <2 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <2 x i64> [[TMP1]], <2 x i64> poison, <2 x i32> zeroinitializer
// LLVM: ret <2 x i64> [[LANE]]
  return vld1q_dup_p64(a);
}

// ALL-LABEL: @test_vld1q_dup_p8(
poly8x16_t test_vld1q_dup_p8(poly8_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.vec.splat [[SCALAR]] : !s8i, !cir.vector<16 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[TMP1:%.*]] = insertelement <16 x i8> poison, i8 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <16 x i8> [[TMP1]], <16 x i8> poison, <16 x i32> zeroinitializer
// LLVM: ret <16 x i8> [[LANE]]
  return vld1q_dup_p8(a);
}

// ALL-LABEL: @test_vld1q_dup_s16(
int16x8_t test_vld1q_dup_s16(int16_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.vec.splat [[SCALAR]] : !s16i, !cir.vector<8 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[TMP1:%.*]] = insertelement <8 x i16> poison, i16 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <8 x i16> [[TMP1]], <8 x i16> poison, <8 x i32> zeroinitializer
// LLVM: ret <8 x i16> [[LANE]]
  return vld1q_dup_s16(a);
}

// ALL-LABEL: @test_vld1q_dup_s32(
int32x4_t test_vld1q_dup_s32(int32_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.vec.splat [[SCALAR]] : !s32i, !cir.vector<4 x !s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i32, ptr [[A]], align 4
// LLVM: [[TMP1:%.*]] = insertelement <4 x i32> poison, i32 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <4 x i32> zeroinitializer
// LLVM: ret <4 x i32> [[LANE]]
  return vld1q_dup_s32(a);
}

// ALL-LABEL: @test_vld1q_dup_s64(
int64x2_t test_vld1q_dup_s64(int64_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.vec.splat [[SCALAR]] : !s64i, !cir.vector<2 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[TMP1:%.*]] = insertelement <2 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <2 x i64> [[TMP1]], <2 x i64> poison, <2 x i32> zeroinitializer
// LLVM: ret <2 x i64> [[LANE]]
  return vld1q_dup_s64(a);
}

// ALL-LABEL: @test_vld1q_dup_s8(
int8x16_t test_vld1q_dup_s8(int8_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.vec.splat [[SCALAR]] : !s8i, !cir.vector<16 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[TMP1:%.*]] = insertelement <16 x i8> poison, i8 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <16 x i8> [[TMP1]], <16 x i8> poison, <16 x i32> zeroinitializer
// LLVM: ret <16 x i8> [[LANE]]
  return vld1q_dup_s8(a);
}

// ALL-LABEL: @test_vld1q_dup_u16(
uint16x8_t test_vld1q_dup_u16(uint16_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!u16i>, !u16i
// CIR: cir.vec.splat [[SCALAR]] : !u16i, !cir.vector<8 x !u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[TMP1:%.*]] = insertelement <8 x i16> poison, i16 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <8 x i16> [[TMP1]], <8 x i16> poison, <8 x i32> zeroinitializer
// LLVM: ret <8 x i16> [[LANE]]
  return vld1q_dup_u16(a);
}

// ALL-LABEL: @test_vld1q_dup_u32(
uint32x4_t test_vld1q_dup_u32(uint32_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.vec.splat [[SCALAR]] : !u32i, !cir.vector<4 x !u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i32, ptr [[A]], align 4
// LLVM: [[TMP1:%.*]] = insertelement <4 x i32> poison, i32 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <4 x i32> zeroinitializer
// LLVM: ret <4 x i32> [[LANE]]
  return vld1q_dup_u32(a);
}

// ALL-LABEL: @test_vld1q_dup_u64(
uint64x2_t test_vld1q_dup_u64(uint64_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!u64i>, !u64i
// CIR: cir.vec.splat [[SCALAR]] : !u64i, !cir.vector<2 x !u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[TMP1:%.*]] = insertelement <2 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <2 x i64> [[TMP1]], <2 x i64> poison, <2 x i32> zeroinitializer
// LLVM: ret <2 x i64> [[LANE]]
  return vld1q_dup_u64(a);
}

// ALL-LABEL: @test_vld1q_dup_u8(
uint8x16_t test_vld1q_dup_u8(uint8_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.vec.splat [[SCALAR]] : !u8i, !cir.vector<16 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[TMP1:%.*]] = insertelement <16 x i8> poison, i8 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <16 x i8> [[TMP1]], <16 x i8> poison, <16 x i32> zeroinitializer
// LLVM: ret <16 x i8> [[LANE]]
  return vld1q_dup_u8(a);
}

// ALL-LABEL: @test_vld1_dup_f16(
float16x4_t test_vld1_dup_f16(float16_t  *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<4 x !cir.f16>>, !cir.vector<4 x !cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load half, ptr [[A]], align 2
// LLVM: [[TMP1:%.*]] = insertelement <4 x half> poison, half [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <4 x half> [[TMP1]], <4 x half> poison, <4 x i32> zeroinitializer
// LLVM: ret <4 x half> [[LANE]]
  return vld1_dup_f16(a);
}

// ALL-LABEL: @test_vld1_dup_f32(
float32x2_t test_vld1_dup_f32(float32_t  *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<2 x !cir.float>>, !cir.vector<2 x !cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load float, ptr [[A]], align 4
// LLVM: [[TMP1:%.*]] = insertelement <2 x float> poison, float [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <2 x float> [[TMP1]], <2 x float> poison, <2 x i32> zeroinitializer
// LLVM: ret <2 x float> [[LANE]]
  return vld1_dup_f32(a);
}

// ALL-LABEL: @test_vld1_dup_f64(
float64x1_t test_vld1_dup_f64(float64_t  *a) {
// CIR: cir.load align(8) {{.*}} : !cir.ptr<!cir.vector<1 x !cir.double>>, !cir.vector<1 x !cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load double, ptr [[A]], align 8
// LLVM: [[TMP1:%.*]] = insertelement <1 x double> poison, double [[TMP0]], i64 0
// LLVM: ret <1 x double> [[TMP1]]
  return vld1_dup_f64(a);
}

// ALL-LABEL: @test_vld1_dup_mf8(
mfloat8x8_t test_vld1_dup_mf8(mfloat8_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.vec.splat [[SCALAR]] : !u8i, !cir.vector<8 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> poison, <8 x i32> zeroinitializer
// LLVM: ret <8 x i8> [[LANE]]
  return vld1_dup_mf8(a);
}

// ALL-LABEL: @test_vld1_dup_p16(
poly16x4_t test_vld1_dup_p16(poly16_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.vec.splat [[SCALAR]] : !s16i, !cir.vector<4 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <4 x i16> [[TMP1]], <4 x i16> poison, <4 x i32> zeroinitializer
// LLVM: ret <4 x i16> [[LANE]]
  return vld1_dup_p16(a);
}

// ALL-LABEL: @test_vld1_dup_p64(
poly64x1_t test_vld1_dup_p64(poly64_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.vec.splat [[SCALAR]] : !s64i, !cir.vector<1 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: ret <1 x i64> [[TMP1]]
  return vld1_dup_p64(a);
}

// ALL-LABEL: @test_vld1_dup_p8(
poly8x8_t test_vld1_dup_p8(poly8_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.vec.splat [[SCALAR]] : !s8i, !cir.vector<8 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> poison, <8 x i32> zeroinitializer
// LLVM: ret <8 x i8> [[LANE]]
  return vld1_dup_p8(a);
}

// ALL-LABEL: @test_vld1_dup_s16(
int16x4_t test_vld1_dup_s16(int16_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.vec.splat [[SCALAR]] : !s16i, !cir.vector<4 x !s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <4 x i16> [[TMP1]], <4 x i16> poison, <4 x i32> zeroinitializer
// LLVM: ret <4 x i16> [[LANE]]
  return vld1_dup_s16(a);
}

// ALL-LABEL: @test_vld1_dup_s32(
int32x2_t test_vld1_dup_s32(int32_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.vec.splat [[SCALAR]] : !s32i, !cir.vector<2 x !s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i32, ptr [[A]], align 4
// LLVM: [[TMP1:%.*]] = insertelement <2 x i32> poison, i32 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <2 x i32> [[TMP1]], <2 x i32> poison, <2 x i32> zeroinitializer
// LLVM: ret <2 x i32> [[LANE]]
  return vld1_dup_s32(a);
}

// ALL-LABEL: @test_vld1_dup_s64(
int64x1_t test_vld1_dup_s64(int64_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.vec.splat [[SCALAR]] : !s64i, !cir.vector<1 x !s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: ret <1 x i64> [[TMP1]]
  return vld1_dup_s64(a);
}

// ALL-LABEL: @test_vld1_dup_s8(
int8x8_t test_vld1_dup_s8(int8_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.vec.splat [[SCALAR]] : !s8i, !cir.vector<8 x !s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> poison, <8 x i32> zeroinitializer
// LLVM: ret <8 x i8> [[LANE]]
  return vld1_dup_s8(a);
}

// ALL-LABEL: @test_vld1_dup_u16(
uint16x4_t test_vld1_dup_u16(uint16_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(2) {{.*}} : !cir.ptr<!u16i>, !u16i
// CIR: cir.vec.splat [[SCALAR]] : !u16i, !cir.vector<4 x !u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i16, ptr [[A]], align 2
// LLVM: [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <4 x i16> [[TMP1]], <4 x i16> poison, <4 x i32> zeroinitializer
// LLVM: ret <4 x i16> [[LANE]]
  return vld1_dup_u16(a);
}

// ALL-LABEL: @test_vld1_dup_u32(
uint32x2_t test_vld1_dup_u32(uint32_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.vec.splat [[SCALAR]] : !u32i, !cir.vector<2 x !u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i32, ptr [[A]], align 4
// LLVM: [[TMP1:%.*]] = insertelement <2 x i32> poison, i32 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <2 x i32> [[TMP1]], <2 x i32> poison, <2 x i32> zeroinitializer
// LLVM: ret <2 x i32> [[LANE]]
  return vld1_dup_u32(a);
}

// ALL-LABEL: @test_vld1_dup_u64(
uint64x1_t test_vld1_dup_u64(uint64_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!u64i>, !u64i
// CIR: cir.vec.splat [[SCALAR]] : !u64i, !cir.vector<1 x !u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i64, ptr [[A]], align 8
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> poison, i64 [[TMP0]], i64 0
// LLVM: ret <1 x i64> [[TMP1]]
  return vld1_dup_u64(a);
}

// ALL-LABEL: @test_vld1_dup_u8(
uint8x8_t test_vld1_dup_u8(uint8_t  *a) {
// CIR: [[SCALAR:%.*]] = cir.load align(1) {{.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.vec.splat [[SCALAR]] : !u8i, !cir.vector<8 x !u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = load i8, ptr [[A]], align 1
// LLVM: [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 [[TMP0]], i64 0
// LLVM: [[LANE:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> poison, <8 x i32> zeroinitializer
// LLVM: ret <8 x i8> [[LANE]]
  return vld1_dup_u8(a);
}
