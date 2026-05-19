// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

//=============================================================================
// NOTES
//
// Tests for vector permutation intrinsics: zip and unzip elements.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#zip-elements
//
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.1.9.10.  Zip elements
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#zip-elements
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vzip1_s8(
// CIR-LABEL: @vzip1_s8(
int8x8_t test_vzip1_s8(int8x8_t a, int8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i] : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: ret <8 x i8> [[VZIP]]
return vzip1_s8(a, b);
}

// LLVM-LABEL: @test_vzip1q_s8(
// CIR-LABEL: @vzip1q_s8(
int8x16_t test_vzip1q_s8(int8x16_t a, int8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<0> : !s64i, #cir.int<16> : !s64i, #cir.int<1> : !s64i, #cir.int<17> : !s64i, #cir.int<2> : !s64i, #cir.int<18> : !s64i, #cir.int<3> : !s64i, #cir.int<19> : !s64i, #cir.int<4> : !s64i, #cir.int<20> : !s64i, #cir.int<5> : !s64i, #cir.int<21> : !s64i, #cir.int<6> : !s64i, #cir.int<22> : !s64i, #cir.int<7> : !s64i, #cir.int<23> : !s64i] : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// LLVM: ret <16 x i8> [[VZIP]]
return vzip1q_s8(a, b);
}

// LLVM-LABEL: @test_vzip1_s16(
// CIR-LABEL: @vzip1_s16(
int16x4_t test_vzip1_s16(int16x4_t a, int16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s16i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<1> : !s64i, #cir.int<5> : !s64i] : !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: ret <4 x i16> [[VZIP]]
return vzip1_s16(a, b);
}

// LLVM-LABEL: @test_vzip1q_s16(
// CIR-LABEL: @vzip1q_s16(
int16x8_t test_vzip1q_s16(int16x8_t a, int16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: ret <8 x i16> [[VZIP]]
return vzip1q_s16(a, b);
}

// LLVM-LABEL: @test_vzip1_s32(
// CIR-LABEL: @vzip1_s32(
int32x2_t test_vzip1_s32(int32x2_t a, int32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i32> [[VZIP]]
return vzip1_s32(a, b);
}

// LLVM-LABEL: @test_vzip1q_s32(
// CIR-LABEL: @vzip1q_s32(
int32x4_t test_vzip1q_s32(int32x4_t a, int32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<1> : !s64i, #cir.int<5> : !s64i] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: ret <4 x i32> [[VZIP]]
return vzip1q_s32(a, b);
}

// LLVM-LABEL: @test_vzip1q_s64(
// CIR-LABEL: @vzip1q_s64(
int64x2_t test_vzip1q_s64(int64x2_t a, int64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[VZIP]]
return vzip1q_s64(a, b);
}

// LLVM-LABEL: @test_vzip1_u8(
// CIR-LABEL: @vzip1_u8(
uint8x8_t test_vzip1_u8(uint8x8_t a, uint8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: ret <8 x i8> [[VZIP]]
return vzip1_u8(a, b);
}

// LLVM-LABEL: @test_vzip1q_u8(
// CIR-LABEL: @vzip1q_u8(
uint8x16_t test_vzip1q_u8(uint8x16_t a, uint8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<16> : !s64i, #cir.int<1> : !s64i, #cir.int<17> : !s64i, #cir.int<2> : !s64i, #cir.int<18> : !s64i, #cir.int<3> : !s64i, #cir.int<19> : !s64i, #cir.int<4> : !s64i, #cir.int<20> : !s64i, #cir.int<5> : !s64i, #cir.int<21> : !s64i, #cir.int<6> : !s64i, #cir.int<22> : !s64i, #cir.int<7> : !s64i, #cir.int<23> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// LLVM: ret <16 x i8> [[VZIP]]
return vzip1q_u8(a, b);
}

// LLVM-LABEL: @test_vzip1_u16(
// CIR-LABEL: @vzip1_u16(
uint16x4_t test_vzip1_u16(uint16x4_t a, uint16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<1> : !s64i, #cir.int<5> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: ret <4 x i16> [[VZIP]]
return vzip1_u16(a, b);
}

// LLVM-LABEL: @test_vzip1q_u16(
// CIR-LABEL: @vzip1q_u16(
uint16x8_t test_vzip1q_u16(uint16x8_t a, uint16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: ret <8 x i16> [[VZIP]]
return vzip1q_u16(a, b);
}

// LLVM-LABEL: @test_vzip1_u32(
// CIR-LABEL: @vzip1_u32(
uint32x2_t test_vzip1_u32(uint32x2_t a, uint32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u32i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i32> [[VZIP]]
return vzip1_u32(a, b);
}

// LLVM-LABEL: @test_vzip1q_u32(
// CIR-LABEL: @vzip1q_u32(
uint32x4_t test_vzip1q_u32(uint32x4_t a, uint32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u32i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<1> : !s64i, #cir.int<5> : !s64i] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: ret <4 x i32> [[VZIP]]
return vzip1q_u32(a, b);
}

// LLVM-LABEL: @test_vzip1q_u64(
// CIR-LABEL: @vzip1q_u64(
uint64x2_t test_vzip1q_u64(uint64x2_t a, uint64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[VZIP]]
return vzip1q_u64(a, b);
}

// LLVM-LABEL: @test_vzip1q_p64(
// CIR-LABEL: @vzip1q_p64(
poly64x2_t test_vzip1q_p64(poly64x2_t a, poly64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[VZIP]]
return vzip1q_p64(a, b);
}

// LLVM-LABEL: @test_vzip1_f32(
// CIR-LABEL: @vzip1_f32(
float32x2_t test_vzip1_f32(float32x2_t a, float32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x float> [[VZIP]]
return vzip1_f32(a, b);
}

// LLVM-LABEL: @test_vzip1q_f32(
// CIR-LABEL: @vzip1q_f32(
float32x4_t test_vzip1q_f32(float32x4_t a, float32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<1> : !s64i, #cir.int<5> : !s64i] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float>{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: ret <4 x float> [[VZIP]]
return vzip1q_f32(a, b);
}

// LLVM-LABEL: @test_vzip1q_f64(
// CIR-LABEL: @vzip1q_f64(
float64x2_t test_vzip1q_f64(float64x2_t a, float64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x double> [[A]], <2 x double> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x double> [[VZIP]]
return vzip1q_f64(a, b);
}

// LLVM-LABEL: @test_vzip1_p8(
// CIR-LABEL: @vzip1_p8(
poly8x8_t test_vzip1_p8(poly8x8_t a, poly8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: ret <8 x i8> [[VZIP]]
return vzip1_p8(a, b);
}

// LLVM-LABEL: @test_vzip1q_p8(
// CIR-LABEL: @vzip1q_p8(
poly8x16_t test_vzip1q_p8(poly8x16_t a, poly8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<16> : !s64i, #cir.int<1> : !s64i, #cir.int<17> : !s64i, #cir.int<2> : !s64i, #cir.int<18> : !s64i, #cir.int<3> : !s64i, #cir.int<19> : !s64i, #cir.int<4> : !s64i, #cir.int<20> : !s64i, #cir.int<5> : !s64i, #cir.int<21> : !s64i, #cir.int<6> : !s64i, #cir.int<22> : !s64i, #cir.int<7> : !s64i, #cir.int<23> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// LLVM: ret <16 x i8> [[VZIP]]
return vzip1q_p8(a, b);
}

// LLVM-LABEL: @test_vzip1_p16(
// CIR-LABEL: @vzip1_p16(
poly16x4_t test_vzip1_p16(poly16x4_t a, poly16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<1> : !s64i, #cir.int<5> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: ret <4 x i16> [[VZIP]]
return vzip1_p16(a, b);
}

// LLVM-LABEL: @test_vzip1q_p16(
// CIR-LABEL: @vzip1q_p16(
poly16x8_t test_vzip1q_p16(poly16x8_t a, poly16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: ret <8 x i16> [[VZIP]]
return vzip1q_p16(a, b);
}

// LLVM-LABEL: @test_vzip1_mf8(
// CIR-LABEL: @vzip1_mf8(
mfloat8x8_t test_vzip1_mf8(mfloat8x8_t a, mfloat8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: ret <8 x i8> [[VZIP]]
return vzip1_mf8(a, b);
}

// LLVM-LABEL: @test_vzip1q_mf8(
// CIR-LABEL: @vzip1q_mf8(
mfloat8x16_t test_vzip1q_mf8(mfloat8x16_t a, mfloat8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<16> : !s64i, #cir.int<1> : !s64i, #cir.int<17> : !s64i, #cir.int<2> : !s64i, #cir.int<18> : !s64i, #cir.int<3> : !s64i, #cir.int<19> : !s64i, #cir.int<4> : !s64i, #cir.int<20> : !s64i, #cir.int<5> : !s64i, #cir.int<21> : !s64i, #cir.int<6> : !s64i, #cir.int<22> : !s64i, #cir.int<7> : !s64i, #cir.int<23> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// LLVM: ret <16 x i8> [[VZIP]]
return vzip1q_mf8(a, b);
}

// LLVM-LABEL: @test_vzip2_s8(
// CIR-LABEL: @vzip2_s8(
int8x8_t test_vzip2_s8(int8x8_t a, int8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: ret <8 x i8> [[VZIP]]
return vzip2_s8(a, b);
}

// LLVM-LABEL: @test_vzip2q_s8(
// CIR-LABEL: @vzip2q_s8(
int8x16_t test_vzip2q_s8(int8x16_t a, int8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<8> : !s64i, #cir.int<24> : !s64i, #cir.int<9> : !s64i, #cir.int<25> : !s64i, #cir.int<10> : !s64i, #cir.int<26> : !s64i, #cir.int<11> : !s64i, #cir.int<27> : !s64i, #cir.int<12> : !s64i, #cir.int<28> : !s64i, #cir.int<13> : !s64i, #cir.int<29> : !s64i, #cir.int<14> : !s64i, #cir.int<30> : !s64i, #cir.int<15> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// LLVM: ret <16 x i8> [[VZIP]]
return vzip2q_s8(a, b);
}

// LLVM-LABEL: @test_vzip2_s16(
// CIR-LABEL: @vzip2_s16(
int16x4_t test_vzip2_s16(int16x4_t a, int16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s16i>) [#cir.int<2> : !s64i, #cir.int<6> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: ret <4 x i16> [[VZIP]]
return vzip2_s16(a, b);
}

// LLVM-LABEL: @test_vzip2q_s16(
// CIR-LABEL: @vzip2q_s16(
int16x8_t test_vzip2q_s16(int16x8_t a, int16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: ret <8 x i16> [[VZIP]]
return vzip2q_s16(a, b);
}

// LLVM-LABEL: @test_vzip2_s32(
// CIR-LABEL: @vzip2_s32(
int32x2_t test_vzip2_s32(int32x2_t a, int32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i32> [[VZIP]]
return vzip2_s32(a, b);
}

// LLVM-LABEL: @test_vzip2q_s32(
// CIR-LABEL: @vzip2q_s32(
int32x4_t test_vzip2q_s32(int32x4_t a, int32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<2> : !s64i, #cir.int<6> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: ret <4 x i32> [[VZIP]]
return vzip2q_s32(a, b);
}

// LLVM-LABEL: @test_vzip2q_s64(
// CIR-LABEL: @vzip2q_s64(
int64x2_t test_vzip2q_s64(int64x2_t a, int64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[VZIP]]
return vzip2q_s64(a, b);
}

// LLVM-LABEL: @test_vzip2_u8(
// CIR-LABEL: @vzip2_u8(
uint8x8_t test_vzip2_u8(uint8x8_t a, uint8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: ret <8 x i8> [[VZIP]]
return vzip2_u8(a, b);
}

// LLVM-LABEL: @test_vzip2q_u8(
// CIR-LABEL: @vzip2q_u8(
uint8x16_t test_vzip2q_u8(uint8x16_t a, uint8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<8> : !s64i, #cir.int<24> : !s64i, #cir.int<9> : !s64i, #cir.int<25> : !s64i, #cir.int<10> : !s64i, #cir.int<26> : !s64i, #cir.int<11> : !s64i, #cir.int<27> : !s64i, #cir.int<12> : !s64i, #cir.int<28> : !s64i, #cir.int<13> : !s64i, #cir.int<29> : !s64i, #cir.int<14> : !s64i, #cir.int<30> : !s64i, #cir.int<15> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// LLVM: ret <16 x i8> [[VZIP]]
return vzip2q_u8(a, b);
}

// LLVM-LABEL: @test_vzip2_u16(
// CIR-LABEL: @vzip2_u16(
uint16x4_t test_vzip2_u16(uint16x4_t a, uint16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<2> : !s64i, #cir.int<6> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: ret <4 x i16> [[VZIP]]
return vzip2_u16(a, b);
}

// LLVM-LABEL: @test_vzip2q_u16(
// CIR-LABEL: @vzip2q_u16(
uint16x8_t test_vzip2q_u16(uint16x8_t a, uint16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: ret <8 x i16> [[VZIP]]
return vzip2q_u16(a, b);
}

// LLVM-LABEL: @test_vzip2_u32(
// CIR-LABEL: @vzip2_u32(
uint32x2_t test_vzip2_u32(uint32x2_t a, uint32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u32i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i32> [[VZIP]]
return vzip2_u32(a, b);
}

// LLVM-LABEL: @test_vzip2q_u32(
// CIR-LABEL: @vzip2q_u32(
uint32x4_t test_vzip2q_u32(uint32x4_t a, uint32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u32i>) [#cir.int<2> : !s64i, #cir.int<6> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: ret <4 x i32> [[VZIP]]
return vzip2q_u32(a, b);
}

// LLVM-LABEL: @test_vzip2q_u64(
// CIR-LABEL: @vzip2q_u64(
uint64x2_t test_vzip2q_u64(uint64x2_t a, uint64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[VZIP]]
return vzip2q_u64(a, b);
}

// LLVM-LABEL: @test_vzip2q_p64(
// CIR-LABEL: @vzip2q_p64(
poly64x2_t test_vzip2q_p64(poly64x2_t a, poly64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[VZIP]]
return vzip2q_p64(a, b);
}

// LLVM-LABEL: @test_vzip2_f32(
// CIR-LABEL: @vzip2_f32(
float32x2_t test_vzip2_f32(float32x2_t a, float32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x float> [[VZIP]]
return vzip2_f32(a, b);
}

// LLVM-LABEL: @test_vzip2q_f32(
// CIR-LABEL: @vzip2q_f32(
float32x4_t test_vzip2q_f32(float32x4_t a, float32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<2> : !s64i, #cir.int<6> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float>{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: ret <4 x float> [[VZIP]]
return vzip2q_f32(a, b);
}

// LLVM-LABEL: @test_vzip2q_f64(
// CIR-LABEL: @vzip2q_f64(
float64x2_t test_vzip2q_f64(float64x2_t a, float64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <2 x double> [[A]], <2 x double> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x double> [[VZIP]]
return vzip2q_f64(a, b);
}

// LLVM-LABEL: @test_vzip2_p8(
// CIR-LABEL: @vzip2_p8(
poly8x8_t test_vzip2_p8(poly8x8_t a, poly8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: ret <8 x i8> [[VZIP]]
return vzip2_p8(a, b);
}

// LLVM-LABEL: @test_vzip2q_p8(
// CIR-LABEL: @vzip2q_p8(
poly8x16_t test_vzip2q_p8(poly8x16_t a, poly8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<8> : !s64i, #cir.int<24> : !s64i, #cir.int<9> : !s64i, #cir.int<25> : !s64i, #cir.int<10> : !s64i, #cir.int<26> : !s64i, #cir.int<11> : !s64i, #cir.int<27> : !s64i, #cir.int<12> : !s64i, #cir.int<28> : !s64i, #cir.int<13> : !s64i, #cir.int<29> : !s64i, #cir.int<14> : !s64i, #cir.int<30> : !s64i, #cir.int<15> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// LLVM: ret <16 x i8> [[VZIP]]
return vzip2q_p8(a, b);
}

// LLVM-LABEL: @test_vzip2_p16(
// CIR-LABEL: @vzip2_p16(
poly16x4_t test_vzip2_p16(poly16x4_t a, poly16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<2> : !s64i, #cir.int<6> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: ret <4 x i16> [[VZIP]]
return vzip2_p16(a, b);
}

// LLVM-LABEL: @test_vzip2q_p16(
// CIR-LABEL: @vzip2q_p16(
poly16x8_t test_vzip2q_p16(poly16x8_t a, poly16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: ret <8 x i16> [[VZIP]]
return vzip2q_p16(a, b);
}

// LLVM-LABEL: @test_vzip2_mf8(
// CIR-LABEL: @vzip2_mf8(
mfloat8x8_t test_vzip2_mf8(mfloat8x8_t a, mfloat8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: ret <8 x i8> [[VZIP]]
return vzip2_mf8(a, b);
}

// LLVM-LABEL: @test_vzip2q_mf8(
// CIR-LABEL: @vzip2q_mf8(
mfloat8x16_t test_vzip2q_mf8(mfloat8x16_t a, mfloat8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<8> : !s64i, #cir.int<24> : !s64i, #cir.int<9> : !s64i, #cir.int<25> : !s64i, #cir.int<10> : !s64i, #cir.int<26> : !s64i, #cir.int<11> : !s64i, #cir.int<27> : !s64i, #cir.int<12> : !s64i, #cir.int<28> : !s64i, #cir.int<13> : !s64i, #cir.int<29> : !s64i, #cir.int<14> : !s64i, #cir.int<30> : !s64i, #cir.int<15> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VZIP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// LLVM: ret <16 x i8> [[VZIP]]
return vzip2q_mf8(a, b);
}

// LLVM-LABEL: @test_vzip_s8(
// CIR-LABEL: @vzip_s8(
int8x8x2_t test_vzip_s8(int8x8_t a, int8x8_t b) {
// CIR: [[LO:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !s8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !s8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: insertvalue %struct.int8x8x2_t poison, <8 x i8> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.int8x8x2_t {{.*}}, <8 x i8> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.int8x8x2_t
return vzip_s8(a, b);
}

// LLVM-LABEL: @test_vzip_s16(
// CIR-LABEL: @vzip_s16(
int16x4x2_t test_vzip_s16(int16x4_t a, int16x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s16i>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<1> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s16i>) [#cir.int<2> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: insertvalue %struct.int16x4x2_t poison, <4 x i16> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.int16x4x2_t {{.*}}, <4 x i16> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.int16x4x2_t
return vzip_s16(a, b);
}

// LLVM-LABEL: @test_vzip_u8(
// CIR-LABEL: @vzip_u8(
uint8x8x2_t test_vzip_u8(uint8x8_t a, uint8x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u8i>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !u8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u8i>) [#cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !u8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: insertvalue %struct.uint8x8x2_t poison, <8 x i8> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.uint8x8x2_t {{.*}}, <8 x i8> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.uint8x8x2_t
return vzip_u8(a, b);
}

// LLVM-LABEL: @test_vzip_u16(
// CIR-LABEL: @vzip_u16(
uint16x4x2_t test_vzip_u16(uint16x4_t a, uint16x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !u16i>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<1> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !u16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !u16i>, !cir.ptr<!cir.vector<4 x !u16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !u16i>) [#cir.int<2> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !u16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !u16i>, !cir.ptr<!cir.vector<4 x !u16i>>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: insertvalue %struct.uint16x4x2_t poison, <4 x i16> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.uint16x4x2_t {{.*}}, <4 x i16> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.uint16x4x2_t
return vzip_u16(a, b);
}

// LLVM-LABEL: @test_vzip_p8(
// CIR-LABEL: @vzip_p8(
poly8x8x2_t test_vzip_p8(poly8x8_t a, poly8x8_t b) {
// CIR: [[LO:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !s8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !s8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: insertvalue %struct.poly8x8x2_t poison, <8 x i8> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.poly8x8x2_t {{.*}}, <8 x i8> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.poly8x8x2_t
return vzip_p8(a, b);
}

// LLVM-LABEL: @test_vzip_p16(
// CIR-LABEL: @vzip_p16(
poly16x4x2_t test_vzip_p16(poly16x4_t a, poly16x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s16i>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<1> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s16i>) [#cir.int<2> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: insertvalue %struct.poly16x4x2_t poison, <4 x i16> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.poly16x4x2_t {{.*}}, <4 x i16> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.poly16x4x2_t
return vzip_p16(a, b);
}

// LLVM-LABEL: @test_vzip_mf8(
// CIR-LABEL: @vzip_mf8(
mfloat8x8x2_t test_vzip_mf8(mfloat8x8_t a, mfloat8x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u8i>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !u8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u8i>) [#cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !u8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>

// LLVM-SAME: <8 x i8> {{.*}}[[A:%.*]], <8 x i8> {{.*}}[[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: insertvalue %struct.mfloat8x8x2_t poison, <8 x i8> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.mfloat8x8x2_t {{.*}}, <8 x i8> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.mfloat8x8x2_t
return vzip_mf8(a, b);
}

// LLVM-LABEL: @test_vzip_s32(
// CIR-LABEL: @vzip_s32(
int32x2x2_t test_vzip_s32(int32x2_t a, int32x2_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i] : !cir.vector<2 x !s32i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !s32i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !s32i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: insertvalue %struct.int32x2x2_t poison, <2 x i32> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.int32x2x2_t {{.*}}, <2 x i32> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.int32x2x2_t
return vzip_s32(a, b);
}

// LLVM-LABEL: @test_vzip_f32(
// CIR-LABEL: @vzip_f32(
float32x2x2_t test_vzip_f32(float32x2_t a, float32x2_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: insertvalue %struct.float32x2x2_t poison, <2 x float> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.float32x2x2_t {{.*}}, <2 x float> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.float32x2x2_t
return vzip_f32(a, b);
}

// LLVM-LABEL: @test_vzip_u32(
// CIR-LABEL: @vzip_u32(
uint32x2x2_t test_vzip_u32(uint32x2_t a, uint32x2_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !u32i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i] : !cir.vector<2 x !u32i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<2 x !u32i>, !cir.ptr<!cir.vector<2 x !u32i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !u32i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !u32i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<2 x !u32i>, !cir.ptr<!cir.vector<2 x !u32i>>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: insertvalue %struct.uint32x2x2_t poison, <2 x i32> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.uint32x2x2_t {{.*}}, <2 x i32> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.uint32x2x2_t
return vzip_u32(a, b);
}

// LLVM-LABEL: @test_vzipq_s8(
// CIR-LABEL: @vzipq_s8(
int8x16x2_t test_vzipq_s8(int8x16_t a, int8x16_t b) {
// CIR: [[LO:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<0> : !s32i, #cir.int<16> : !s32i, #cir.int<1> : !s32i, #cir.int<17> : !s32i, #cir.int<2> : !s32i, #cir.int<18> : !s32i, #cir.int<3> : !s32i, #cir.int<19> : !s32i, #cir.int<4> : !s32i, #cir.int<20> : !s32i, #cir.int<5> : !s32i, #cir.int<21> : !s32i, #cir.int<6> : !s32i, #cir.int<22> : !s32i, #cir.int<7> : !s32i, #cir.int<23> : !s32i] : !cir.vector<16 x !s8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<8> : !s32i, #cir.int<24> : !s32i, #cir.int<9> : !s32i, #cir.int<25> : !s32i, #cir.int<10> : !s32i, #cir.int<26> : !s32i, #cir.int<11> : !s32i, #cir.int<27> : !s32i, #cir.int<12> : !s32i, #cir.int<28> : !s32i, #cir.int<13> : !s32i, #cir.int<29> : !s32i, #cir.int<14> : !s32i, #cir.int<30> : !s32i, #cir.int<15> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !s8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// LLVM: insertvalue %struct.int8x16x2_t poison, <16 x i8> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.int8x16x2_t {{.*}}, <16 x i8> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.int8x16x2_t
return vzipq_s8(a, b);
}

// LLVM-LABEL: @test_vzipq_s16(
// CIR-LABEL: @vzipq_s16(
int16x8x2_t test_vzipq_s16(int16x8_t a, int16x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !s16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !s16i>) [#cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !s16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: insertvalue %struct.int16x8x2_t poison, <8 x i16> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.int16x8x2_t {{.*}}, <8 x i16> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.int16x8x2_t
return vzipq_s16(a, b);
}

// LLVM-LABEL: @test_vzipq_s32(
// CIR-LABEL: @vzipq_s32(
int32x4x2_t test_vzipq_s32(int32x4_t a, int32x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<1> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s32i>) [#cir.int<2> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: insertvalue %struct.int32x4x2_t poison, <4 x i32> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.int32x4x2_t {{.*}}, <4 x i32> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.int32x4x2_t
return vzipq_s32(a, b);
}

// LLVM-LABEL: @test_vzipq_f32(
// CIR-LABEL: @vzipq_f32(
float32x4x2_t test_vzipq_f32(float32x4_t a, float32x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<1> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !cir.float>) [#cir.int<2> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: insertvalue %struct.float32x4x2_t poison, <4 x float> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.float32x4x2_t {{.*}}, <4 x float> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.float32x4x2_t
return vzipq_f32(a, b);
}

// LLVM-LABEL: @test_vzipq_u8(
// CIR-LABEL: @vzipq_u8(
uint8x16x2_t test_vzipq_u8(uint8x16_t a, uint8x16_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<16 x !u8i>) [#cir.int<0> : !s32i, #cir.int<16> : !s32i, #cir.int<1> : !s32i, #cir.int<17> : !s32i, #cir.int<2> : !s32i, #cir.int<18> : !s32i, #cir.int<3> : !s32i, #cir.int<19> : !s32i, #cir.int<4> : !s32i, #cir.int<20> : !s32i, #cir.int<5> : !s32i, #cir.int<21> : !s32i, #cir.int<6> : !s32i, #cir.int<22> : !s32i, #cir.int<7> : !s32i, #cir.int<23> : !s32i] : !cir.vector<16 x !u8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<16 x !u8i>) [#cir.int<8> : !s32i, #cir.int<24> : !s32i, #cir.int<9> : !s32i, #cir.int<25> : !s32i, #cir.int<10> : !s32i, #cir.int<26> : !s32i, #cir.int<11> : !s32i, #cir.int<27> : !s32i, #cir.int<12> : !s32i, #cir.int<28> : !s32i, #cir.int<13> : !s32i, #cir.int<29> : !s32i, #cir.int<14> : !s32i, #cir.int<30> : !s32i, #cir.int<15> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !u8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// LLVM: insertvalue %struct.uint8x16x2_t poison, <16 x i8> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.uint8x16x2_t {{.*}}, <16 x i8> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.uint8x16x2_t
return vzipq_u8(a, b);
}

// LLVM-LABEL: @test_vzipq_u16(
// CIR-LABEL: @vzipq_u16(
uint16x8x2_t test_vzipq_u16(uint16x8_t a, uint16x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u16i>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !u16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !u16i>, !cir.ptr<!cir.vector<8 x !u16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u16i>) [#cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !u16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !u16i>, !cir.ptr<!cir.vector<8 x !u16i>>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: insertvalue %struct.uint16x8x2_t poison, <8 x i16> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.uint16x8x2_t {{.*}}, <8 x i16> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.uint16x8x2_t
return vzipq_u16(a, b);
}

// LLVM-LABEL: @test_vzipq_u32(
// CIR-LABEL: @vzipq_u32(
uint32x4x2_t test_vzipq_u32(uint32x4_t a, uint32x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !u32i>) [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<1> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !u32i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !u32i>, !cir.ptr<!cir.vector<4 x !u32i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !u32i>) [#cir.int<2> : !s32i, #cir.int<6> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !u32i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !u32i>, !cir.ptr<!cir.vector<4 x !u32i>>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// LLVM: insertvalue %struct.uint32x4x2_t poison, <4 x i32> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.uint32x4x2_t {{.*}}, <4 x i32> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.uint32x4x2_t
return vzipq_u32(a, b);
}

// LLVM-LABEL: @test_vzipq_p8(
// CIR-LABEL: @vzipq_p8(
poly8x16x2_t test_vzipq_p8(poly8x16_t a, poly8x16_t b) {
// CIR: [[LO:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<0> : !s32i, #cir.int<16> : !s32i, #cir.int<1> : !s32i, #cir.int<17> : !s32i, #cir.int<2> : !s32i, #cir.int<18> : !s32i, #cir.int<3> : !s32i, #cir.int<19> : !s32i, #cir.int<4> : !s32i, #cir.int<20> : !s32i, #cir.int<5> : !s32i, #cir.int<21> : !s32i, #cir.int<6> : !s32i, #cir.int<22> : !s32i, #cir.int<7> : !s32i, #cir.int<23> : !s32i] : !cir.vector<16 x !s8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<8> : !s32i, #cir.int<24> : !s32i, #cir.int<9> : !s32i, #cir.int<25> : !s32i, #cir.int<10> : !s32i, #cir.int<26> : !s32i, #cir.int<11> : !s32i, #cir.int<27> : !s32i, #cir.int<12> : !s32i, #cir.int<28> : !s32i, #cir.int<13> : !s32i, #cir.int<29> : !s32i, #cir.int<14> : !s32i, #cir.int<30> : !s32i, #cir.int<15> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !s8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// LLVM: insertvalue %struct.poly8x16x2_t poison, <16 x i8> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.poly8x16x2_t {{.*}}, <16 x i8> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.poly8x16x2_t
return vzipq_p8(a, b);
}

// LLVM-LABEL: @test_vzipq_p16(
// CIR-LABEL: @vzipq_p16(
poly16x8x2_t test_vzipq_p16(poly16x8_t a, poly16x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !s16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !s16i>) [#cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<6> : !s32i, #cir.int<14> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !s16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// LLVM: insertvalue %struct.poly16x8x2_t poison, <8 x i16> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.poly16x8x2_t {{.*}}, <8 x i16> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.poly16x8x2_t
return vzipq_p16(a, b);
}

// LLVM-LABEL: @test_vzipq_mf8(
// CIR-LABEL: @vzipq_mf8(
mfloat8x16x2_t test_vzipq_mf8(mfloat8x16_t a, mfloat8x16_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<16 x !u8i>) [#cir.int<0> : !s32i, #cir.int<16> : !s32i, #cir.int<1> : !s32i, #cir.int<17> : !s32i, #cir.int<2> : !s32i, #cir.int<18> : !s32i, #cir.int<3> : !s32i, #cir.int<19> : !s32i, #cir.int<4> : !s32i, #cir.int<20> : !s32i, #cir.int<5> : !s32i, #cir.int<21> : !s32i, #cir.int<6> : !s32i, #cir.int<22> : !s32i, #cir.int<7> : !s32i, #cir.int<23> : !s32i] : !cir.vector<16 x !u8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<16 x !u8i>) [#cir.int<8> : !s32i, #cir.int<24> : !s32i, #cir.int<9> : !s32i, #cir.int<25> : !s32i, #cir.int<10> : !s32i, #cir.int<26> : !s32i, #cir.int<11> : !s32i, #cir.int<27> : !s32i, #cir.int<12> : !s32i, #cir.int<28> : !s32i, #cir.int<13> : !s32i, #cir.int<29> : !s32i, #cir.int<14> : !s32i, #cir.int<30> : !s32i, #cir.int<15> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !u8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>

// LLVM-SAME: <16 x i8> {{.*}}[[A:%.*]], <16 x i8> {{.*}}[[B:%.*]]) {{.*}} {
// LLVM: [[VZIP_LO:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// LLVM: [[VZIP_HI:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// LLVM: insertvalue %struct.mfloat8x16x2_t poison, <16 x i8> [[VZIP_LO]], 0, 0
// LLVM: insertvalue %struct.mfloat8x16x2_t {{.*}}, <16 x i8> [[VZIP_HI]], 0, 1
// LLVM: ret %struct.mfloat8x16x2_t
return vzipq_mf8(a, b);
}

//===------------------------------------------------------===//
// 2.1.9.11.  Unzip elements
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#unzip-elements
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vuzp_s8(
// CIR-LABEL: @vuzp_s8(
int8x8x2_t test_vuzp_s8(int8x8_t a, int8x8_t b) {
// CIR: [[LO:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i] : !cir.vector<8 x !s8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !s8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: insertvalue %struct.int8x8x2_t poison, <8 x i8> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.int8x8x2_t {{.*}}, <8 x i8> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.int8x8x2_t
return vuzp_s8(a, b);
}

// LLVM-LABEL: @test_vuzp_s16(
// CIR-LABEL: @vuzp_s16(
int16x4x2_t test_vuzp_s16(int16x4_t a, int16x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s16i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s16i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: insertvalue %struct.int16x4x2_t poison, <4 x i16> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.int16x4x2_t {{.*}}, <4 x i16> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.int16x4x2_t
return vuzp_s16(a, b);
}

// LLVM-LABEL: @test_vuzp_u8(
// CIR-LABEL: @vuzp_u8(
uint8x8x2_t test_vuzp_u8(uint8x8_t a, uint8x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u8i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i] : !cir.vector<8 x !u8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u8i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !u8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: insertvalue %struct.uint8x8x2_t poison, <8 x i8> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.uint8x8x2_t {{.*}}, <8 x i8> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.uint8x8x2_t
return vuzp_u8(a, b);
}

// LLVM-LABEL: @test_vuzp_u16(
// CIR-LABEL: @vuzp_u16(
uint16x4x2_t test_vuzp_u16(uint16x4_t a, uint16x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !u16i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i] : !cir.vector<4 x !u16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !u16i>, !cir.ptr<!cir.vector<4 x !u16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !u16i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !u16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !u16i>, !cir.ptr<!cir.vector<4 x !u16i>>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: insertvalue %struct.uint16x4x2_t poison, <4 x i16> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.uint16x4x2_t {{.*}}, <4 x i16> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.uint16x4x2_t
return vuzp_u16(a, b);
}

// LLVM-LABEL: @test_vuzp_p8(
// CIR-LABEL: @vuzp_p8(
poly8x8x2_t test_vuzp_p8(poly8x8_t a, poly8x8_t b) {
// CIR: [[LO:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i] : !cir.vector<8 x !s8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !s8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: insertvalue %struct.poly8x8x2_t poison, <8 x i8> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.poly8x8x2_t {{.*}}, <8 x i8> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.poly8x8x2_t
return vuzp_p8(a, b);
}

// LLVM-LABEL: @test_vuzp_p16(
// CIR-LABEL: @vuzp_p16(
poly16x4x2_t test_vuzp_p16(poly16x4_t a, poly16x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s16i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s16i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: insertvalue %struct.poly16x4x2_t poison, <4 x i16> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.poly16x4x2_t {{.*}}, <4 x i16> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.poly16x4x2_t
return vuzp_p16(a, b);
}

// LLVM-LABEL: @test_vuzp_mf8(
// CIR-LABEL: @vuzp_mf8(
mfloat8x8x2_t test_vuzp_mf8(mfloat8x8_t a, mfloat8x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u8i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i] : !cir.vector<8 x !u8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u8i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !u8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>

// LLVM-SAME: <8 x i8> {{.*}}[[A:%.*]], <8 x i8> {{.*}}[[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: insertvalue %struct.mfloat8x8x2_t poison, <8 x i8> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.mfloat8x8x2_t {{.*}}, <8 x i8> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.mfloat8x8x2_t
return vuzp_mf8(a, b);
}

// LLVM-LABEL: @test_vuzp_s32(
// CIR-LABEL: @vuzp_s32(
int32x2x2_t test_vuzp_s32(int32x2_t a, int32x2_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !s32i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i] : !cir.vector<2 x !s32i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !s32i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !s32i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: insertvalue %struct.int32x2x2_t poison, <2 x i32> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.int32x2x2_t {{.*}}, <2 x i32> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.int32x2x2_t
return vuzp_s32(a, b);
}

// LLVM-LABEL: @test_vuzp_f32(
// CIR-LABEL: @vuzp_f32(
float32x2x2_t test_vuzp_f32(float32x2_t a, float32x2_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: insertvalue %struct.float32x2x2_t poison, <2 x float> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.float32x2x2_t {{.*}}, <2 x float> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.float32x2x2_t
return vuzp_f32(a, b);
}

// LLVM-LABEL: @test_vuzp_u32(
// CIR-LABEL: @vuzp_u32(
uint32x2x2_t test_vuzp_u32(uint32x2_t a, uint32x2_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !u32i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i] : !cir.vector<2 x !u32i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<2 x !u32i>, !cir.ptr<!cir.vector<2 x !u32i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<2 x !u32i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !u32i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<2 x !u32i>, !cir.ptr<!cir.vector<2 x !u32i>>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: insertvalue %struct.uint32x2x2_t poison, <2 x i32> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.uint32x2x2_t {{.*}}, <2 x i32> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.uint32x2x2_t
return vuzp_u32(a, b);
}

// LLVM-LABEL: @test_vuzpq_s8(
// CIR-LABEL: @vuzpq_s8(
int8x16x2_t test_vuzpq_s8(int8x16_t a, int8x16_t b) {
// CIR: [[LO:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i, #cir.int<16> : !s32i, #cir.int<18> : !s32i, #cir.int<20> : !s32i, #cir.int<22> : !s32i, #cir.int<24> : !s32i, #cir.int<26> : !s32i, #cir.int<28> : !s32i, #cir.int<30> : !s32i] : !cir.vector<16 x !s8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i, #cir.int<17> : !s32i, #cir.int<19> : !s32i, #cir.int<21> : !s32i, #cir.int<23> : !s32i, #cir.int<25> : !s32i, #cir.int<27> : !s32i, #cir.int<29> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !s8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// LLVM: insertvalue %struct.int8x16x2_t poison, <16 x i8> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.int8x16x2_t {{.*}}, <16 x i8> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.int8x16x2_t
return vuzpq_s8(a, b);
}

// LLVM-LABEL: @test_vuzpq_s16(
// CIR-LABEL: @vuzpq_s16(
int16x8x2_t test_vuzpq_s16(int16x8_t a, int16x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i] : !cir.vector<8 x !s16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !s16i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !s16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: insertvalue %struct.int16x8x2_t poison, <8 x i16> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.int16x8x2_t {{.*}}, <8 x i16> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.int16x8x2_t
return vuzpq_s16(a, b);
}

// LLVM-LABEL: @test_vuzpq_s32(
// CIR-LABEL: @vuzpq_s32(
int32x4x2_t test_vuzpq_s32(int32x4_t a, int32x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !s32i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !s32i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: insertvalue %struct.int32x4x2_t poison, <4 x i32> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.int32x4x2_t {{.*}}, <4 x i32> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.int32x4x2_t
return vuzpq_s32(a, b);
}

// LLVM-LABEL: @test_vuzpq_f32(
// CIR-LABEL: @vuzpq_f32(
float32x4x2_t test_vuzpq_f32(float32x4_t a, float32x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !cir.float>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: insertvalue %struct.float32x4x2_t poison, <4 x float> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.float32x4x2_t {{.*}}, <4 x float> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.float32x4x2_t
return vuzpq_f32(a, b);
}

// LLVM-LABEL: @test_vuzpq_u8(
// CIR-LABEL: @vuzpq_u8(
uint8x16x2_t test_vuzpq_u8(uint8x16_t a, uint8x16_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<16 x !u8i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i, #cir.int<16> : !s32i, #cir.int<18> : !s32i, #cir.int<20> : !s32i, #cir.int<22> : !s32i, #cir.int<24> : !s32i, #cir.int<26> : !s32i, #cir.int<28> : !s32i, #cir.int<30> : !s32i] : !cir.vector<16 x !u8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<16 x !u8i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i, #cir.int<17> : !s32i, #cir.int<19> : !s32i, #cir.int<21> : !s32i, #cir.int<23> : !s32i, #cir.int<25> : !s32i, #cir.int<27> : !s32i, #cir.int<29> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !u8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// LLVM: insertvalue %struct.uint8x16x2_t poison, <16 x i8> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.uint8x16x2_t {{.*}}, <16 x i8> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.uint8x16x2_t
return vuzpq_u8(a, b);
}

// LLVM-LABEL: @test_vuzpq_u32(
// CIR-LABEL: @vuzpq_u32(
uint32x4x2_t test_vuzpq_u32(uint32x4_t a, uint32x4_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !u32i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i] : !cir.vector<4 x !u32i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<4 x !u32i>, !cir.ptr<!cir.vector<4 x !u32i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<4 x !u32i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !u32i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<4 x !u32i>, !cir.ptr<!cir.vector<4 x !u32i>>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: insertvalue %struct.uint32x4x2_t poison, <4 x i32> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.uint32x4x2_t {{.*}}, <4 x i32> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.uint32x4x2_t
return vuzpq_u32(a, b);
}

// LLVM-LABEL: @test_vuzpq_p8(
// CIR-LABEL: @vuzpq_p8(
poly8x16x2_t test_vuzpq_p8(poly8x16_t a, poly8x16_t b) {
// CIR: [[LO:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i, #cir.int<16> : !s32i, #cir.int<18> : !s32i, #cir.int<20> : !s32i, #cir.int<22> : !s32i, #cir.int<24> : !s32i, #cir.int<26> : !s32i, #cir.int<28> : !s32i, #cir.int<30> : !s32i] : !cir.vector<16 x !s8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i, #cir.int<17> : !s32i, #cir.int<19> : !s32i, #cir.int<21> : !s32i, #cir.int<23> : !s32i, #cir.int<25> : !s32i, #cir.int<27> : !s32i, #cir.int<29> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !s8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// LLVM: insertvalue %struct.poly8x16x2_t poison, <16 x i8> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.poly8x16x2_t {{.*}}, <16 x i8> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.poly8x16x2_t
return vuzpq_p8(a, b);
}

// LLVM-LABEL: @test_vuzpq_p16(
// CIR-LABEL: @vuzpq_p16(
poly16x8x2_t test_vuzpq_p16(poly16x8_t a, poly16x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i] : !cir.vector<8 x !s16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !s16i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !s16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: insertvalue %struct.poly16x8x2_t poison, <8 x i16> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.poly16x8x2_t {{.*}}, <8 x i16> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.poly16x8x2_t
return vuzpq_p16(a, b);
}

// LLVM-LABEL: @test_vuzpq_mf8(
// CIR-LABEL: @vuzpq_mf8(
mfloat8x16x2_t test_vuzpq_mf8(mfloat8x16_t a, mfloat8x16_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<16 x !u8i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i, #cir.int<16> : !s32i, #cir.int<18> : !s32i, #cir.int<20> : !s32i, #cir.int<22> : !s32i, #cir.int<24> : !s32i, #cir.int<26> : !s32i, #cir.int<28> : !s32i, #cir.int<30> : !s32i] : !cir.vector<16 x !u8i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<16 x !u8i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i, #cir.int<17> : !s32i, #cir.int<19> : !s32i, #cir.int<21> : !s32i, #cir.int<23> : !s32i, #cir.int<25> : !s32i, #cir.int<27> : !s32i, #cir.int<29> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !u8i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>

// LLVM-SAME: <16 x i8> {{.*}}[[A:%.*]], <16 x i8> {{.*}}[[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// LLVM: insertvalue %struct.mfloat8x16x2_t poison, <16 x i8> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.mfloat8x16x2_t {{.*}}, <16 x i8> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.mfloat8x16x2_t
return vuzpq_mf8(a, b);
}

// LLVM-LABEL: @test_vuzp1_s8(
// CIR-LABEL: @vuzp1_s8(
int8x8_t test_vuzp1_s8(int8x8_t a, int8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: ret <8 x i8> [[VUZP]]
  return vuzp1_s8(a, b);
}

// LLVM-LABEL: @test_vuzp1q_s8(
// CIR-LABEL: @vuzp1q_s8(
int8x16_t test_vuzp1q_s8(int8x16_t a, int8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i, #cir.int<16> : !s64i, #cir.int<18> : !s64i, #cir.int<20> : !s64i, #cir.int<22> : !s64i, #cir.int<24> : !s64i, #cir.int<26> : !s64i, #cir.int<28> : !s64i, #cir.int<30> : !s64i] : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// LLVM: ret <16 x i8> [[VUZP]]
  return vuzp1q_s8(a, b);
}

// LLVM-LABEL: @test_vuzp1_s16(
// CIR-LABEL: @vuzp1_s16(
int16x4_t test_vuzp1_s16(int16x4_t a, int16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s16i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: ret <4 x i16> [[VUZP]]
  return vuzp1_s16(a, b);
}

// LLVM-LABEL: @test_vuzp1q_s16(
// CIR-LABEL: @vuzp1q_s16(
int16x8_t test_vuzp1q_s16(int16x8_t a, int16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: ret <8 x i16> [[VUZP]]
  return vuzp1q_s16(a, b);
}

// LLVM-LABEL: @test_vuzp1_s32(
// CIR-LABEL: @vuzp1_s32(
int32x2_t test_vuzp1_s32(int32x2_t a, int32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i32> [[VUZP]]
  return vuzp1_s32(a, b);
}

// LLVM-LABEL: @test_vuzp1q_s32(
// CIR-LABEL: @vuzp1q_s32(
int32x4_t test_vuzp1q_s32(int32x4_t a, int32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: ret <4 x i32> [[VUZP]]
  return vuzp1q_s32(a, b);
}

// LLVM-LABEL: @test_vuzp1q_s64(
// CIR-LABEL: @vuzp1q_s64(
int64x2_t test_vuzp1q_s64(int64x2_t a, int64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[VUZP]]
  return vuzp1q_s64(a, b);
}

// LLVM-LABEL: @test_vuzp1_u8(
// CIR-LABEL: @vuzp1_u8(
uint8x8_t test_vuzp1_u8(uint8x8_t a, uint8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: ret <8 x i8> [[VUZP]]
  return vuzp1_u8(a, b);
}

// LLVM-LABEL: @test_vuzp1q_u8(
// CIR-LABEL: @vuzp1q_u8(
uint8x16_t test_vuzp1q_u8(uint8x16_t a, uint8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i, #cir.int<16> : !s64i, #cir.int<18> : !s64i, #cir.int<20> : !s64i, #cir.int<22> : !s64i, #cir.int<24> : !s64i, #cir.int<26> : !s64i, #cir.int<28> : !s64i, #cir.int<30> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// LLVM: ret <16 x i8> [[VUZP]]
  return vuzp1q_u8(a, b);
}

// LLVM-LABEL: @test_vuzp1_u16(
// CIR-LABEL: @vuzp1_u16(
uint16x4_t test_vuzp1_u16(uint16x4_t a, uint16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: ret <4 x i16> [[VUZP]]
  return vuzp1_u16(a, b);
}

// LLVM-LABEL: @test_vuzp1q_u16(
// CIR-LABEL: @vuzp1q_u16(
uint16x8_t test_vuzp1q_u16(uint16x8_t a, uint16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: ret <8 x i16> [[VUZP]]
  return vuzp1q_u16(a, b);
}

// LLVM-LABEL: @test_vuzp1_u32(
// CIR-LABEL: @vuzp1_u32(
uint32x2_t test_vuzp1_u32(uint32x2_t a, uint32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u32i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i32> [[VUZP]]
  return vuzp1_u32(a, b);
}

// LLVM-LABEL: @test_vuzp1q_u32(
// CIR-LABEL: @vuzp1q_u32(
uint32x4_t test_vuzp1q_u32(uint32x4_t a, uint32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u32i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: ret <4 x i32> [[VUZP]]
  return vuzp1q_u32(a, b);
}

// LLVM-LABEL: @test_vuzp1q_u64(
// CIR-LABEL: @vuzp1q_u64(
uint64x2_t test_vuzp1q_u64(uint64x2_t a, uint64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[VUZP]]
  return vuzp1q_u64(a, b);
}

// LLVM-LABEL: @test_vuzp1q_p64(
// CIR-LABEL: @vuzp1q_p64(
poly64x2_t test_vuzp1q_p64(poly64x2_t a, poly64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[VUZP]]
  return vuzp1q_p64(a, b);
}

// LLVM-LABEL: @test_vuzp1_f32(
// CIR-LABEL: @vuzp1_f32(
float32x2_t test_vuzp1_f32(float32x2_t a, float32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x float> [[VUZP]]
  return vuzp1_f32(a, b);
}

// LLVM-LABEL: @test_vuzp1q_f32(
// CIR-LABEL: @vuzp1q_f32(
float32x4_t test_vuzp1q_f32(float32x4_t a, float32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float>{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: ret <4 x float> [[VUZP]]
  return vuzp1q_f32(a, b);
}

// LLVM-LABEL: @test_vuzp1q_f64(
// CIR-LABEL: @vuzp1q_f64(
float64x2_t test_vuzp1q_f64(float64x2_t a, float64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x double> [[A]], <2 x double> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x double> [[VUZP]]
  return vuzp1q_f64(a, b);
}

// LLVM-LABEL: @test_vuzp1_p8(
// CIR-LABEL: @vuzp1_p8(
poly8x8_t test_vuzp1_p8(poly8x8_t a, poly8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: ret <8 x i8> [[VUZP]]
  return vuzp1_p8(a, b);
}

// LLVM-LABEL: @test_vuzp1q_p8(
// CIR-LABEL: @vuzp1q_p8(
poly8x16_t test_vuzp1q_p8(poly8x16_t a, poly8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i, #cir.int<16> : !s64i, #cir.int<18> : !s64i, #cir.int<20> : !s64i, #cir.int<22> : !s64i, #cir.int<24> : !s64i, #cir.int<26> : !s64i, #cir.int<28> : !s64i, #cir.int<30> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// LLVM: ret <16 x i8> [[VUZP]]
  return vuzp1q_p8(a, b);
}

// LLVM-LABEL: @test_vuzp1_p16(
// CIR-LABEL: @vuzp1_p16(
poly16x4_t test_vuzp1_p16(poly16x4_t a, poly16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// LLVM: ret <4 x i16> [[VUZP]]
  return vuzp1_p16(a, b);
}

// LLVM-LABEL: @test_vuzp1q_p16(
// CIR-LABEL: @vuzp1q_p16(
poly16x8_t test_vuzp1q_p16(poly16x8_t a, poly16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: ret <8 x i16> [[VUZP]]
  return vuzp1q_p16(a, b);
}

// LLVM-LABEL: @test_vuzp1_mf8(
// CIR-LABEL: @vuzp1_mf8(
mfloat8x8_t test_vuzp1_mf8(mfloat8x8_t a, mfloat8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: ret <8 x i8> [[VUZP]]
  return vuzp1_mf8(a, b);
}

// LLVM-LABEL: @test_vuzp1q_mf8(
// CIR-LABEL: @vuzp1q_mf8(
mfloat8x16_t test_vuzp1q_mf8(mfloat8x16_t a, mfloat8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i, #cir.int<4> : !s64i, #cir.int<6> : !s64i, #cir.int<8> : !s64i, #cir.int<10> : !s64i, #cir.int<12> : !s64i, #cir.int<14> : !s64i, #cir.int<16> : !s64i, #cir.int<18> : !s64i, #cir.int<20> : !s64i, #cir.int<22> : !s64i, #cir.int<24> : !s64i, #cir.int<26> : !s64i, #cir.int<28> : !s64i, #cir.int<30> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// LLVM: ret <16 x i8> [[VUZP]]
  return vuzp1q_mf8(a, b);
}

// LLVM-LABEL: @test_vuzp2_s8(
// CIR-LABEL: @vuzp2_s8(
int8x8_t test_vuzp2_s8(int8x8_t a, int8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: ret <8 x i8> [[VUZP]]
  return vuzp2_s8(a, b);
}

// LLVM-LABEL: @test_vuzp2q_s8(
// CIR-LABEL: @vuzp2q_s8(
int8x16_t test_vuzp2q_s8(int8x16_t a, int8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i, #cir.int<17> : !s64i, #cir.int<19> : !s64i, #cir.int<21> : !s64i, #cir.int<23> : !s64i, #cir.int<25> : !s64i, #cir.int<27> : !s64i, #cir.int<29> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// LLVM: ret <16 x i8> [[VUZP]]
  return vuzp2q_s8(a, b);
}

// LLVM-LABEL: @test_vuzp2_s16(
// CIR-LABEL: @vuzp2_s16(
int16x4_t test_vuzp2_s16(int16x4_t a, int16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s16i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: ret <4 x i16> [[VUZP]]
  return vuzp2_s16(a, b);
}

// LLVM-LABEL: @test_vuzp2q_s16(
// CIR-LABEL: @vuzp2q_s16(
int16x8_t test_vuzp2q_s16(int16x8_t a, int16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: ret <8 x i16> [[VUZP]]
  return vuzp2q_s16(a, b);
}

// LLVM-LABEL: @test_vuzp2_s32(
// CIR-LABEL: @vuzp2_s32(
int32x2_t test_vuzp2_s32(int32x2_t a, int32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i32> [[VUZP]]
  return vuzp2_s32(a, b);
}

// LLVM-LABEL: @test_vuzp2q_s32(
// CIR-LABEL: @vuzp2q_s32(
int32x4_t test_vuzp2q_s32(int32x4_t a, int32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: ret <4 x i32> [[VUZP]]
  return vuzp2q_s32(a, b);
}

// LLVM-LABEL: @test_vuzp2q_s64(
// CIR-LABEL: @vuzp2q_s64(
int64x2_t test_vuzp2q_s64(int64x2_t a, int64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[VUZP]]
  return vuzp2q_s64(a, b);
}

// LLVM-LABEL: @test_vuzp2_u8(
// CIR-LABEL: @vuzp2_u8(
uint8x8_t test_vuzp2_u8(uint8x8_t a, uint8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: ret <8 x i8> [[VUZP]]
  return vuzp2_u8(a, b);
}

// LLVM-LABEL: @test_vuzp2q_u8(
// CIR-LABEL: @vuzp2q_u8(
uint8x16_t test_vuzp2q_u8(uint8x16_t a, uint8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i, #cir.int<17> : !s64i, #cir.int<19> : !s64i, #cir.int<21> : !s64i, #cir.int<23> : !s64i, #cir.int<25> : !s64i, #cir.int<27> : !s64i, #cir.int<29> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// LLVM: ret <16 x i8> [[VUZP]]
  return vuzp2q_u8(a, b);
}

// LLVM-LABEL: @test_vuzp2_u16(
// CIR-LABEL: @vuzp2_u16(
uint16x4_t test_vuzp2_u16(uint16x4_t a, uint16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: ret <4 x i16> [[VUZP]]
  return vuzp2_u16(a, b);
}

// LLVM-LABEL: @test_vuzp2q_u16(
// CIR-LABEL: @vuzp2q_u16(
uint16x8_t test_vuzp2q_u16(uint16x8_t a, uint16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: ret <8 x i16> [[VUZP]]
  return vuzp2q_u16(a, b);
}

// LLVM-LABEL: @test_vuzp2_u32(
// CIR-LABEL: @vuzp2_u32(
uint32x2_t test_vuzp2_u32(uint32x2_t a, uint32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u32i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i32> [[VUZP]]
  return vuzp2_u32(a, b);
}

// LLVM-LABEL: @test_vuzp2q_u32(
// CIR-LABEL: @vuzp2q_u32(
uint32x4_t test_vuzp2q_u32(uint32x4_t a, uint32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u32i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: ret <4 x i32> [[VUZP]]
  return vuzp2q_u32(a, b);
}

// LLVM-LABEL: @test_vuzp2q_u64(
// CIR-LABEL: @vuzp2q_u64(
uint64x2_t test_vuzp2q_u64(uint64x2_t a, uint64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[VUZP]]
  return vuzp2q_u64(a, b);
}

// LLVM-LABEL: @test_vuzp2q_p64(
// CIR-LABEL: @vuzp2q_p64(
poly64x2_t test_vuzp2q_p64(poly64x2_t a, poly64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[VUZP]]
  return vuzp2q_p64(a, b);
}

// LLVM-LABEL: @test_vuzp2_f32(
// CIR-LABEL: @vuzp2_f32(
float32x2_t test_vuzp2_f32(float32x2_t a, float32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x float> [[VUZP]]
  return vuzp2_f32(a, b);
}

// LLVM-LABEL: @test_vuzp2q_f32(
// CIR-LABEL: @vuzp2q_f32(
float32x4_t test_vuzp2q_f32(float32x4_t a, float32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float>{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: ret <4 x float> [[VUZP]]
  return vuzp2q_f32(a, b);
}

// LLVM-LABEL: @test_vuzp2q_f64(
// CIR-LABEL: @vuzp2q_f64(
float64x2_t test_vuzp2q_f64(float64x2_t a, float64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <2 x double> [[A]], <2 x double> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x double> [[VUZP]]
  return vuzp2q_f64(a, b);
}

// LLVM-LABEL: @test_vuzp2_p8(
// CIR-LABEL: @vuzp2_p8(
poly8x8_t test_vuzp2_p8(poly8x8_t a, poly8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: ret <8 x i8> [[VUZP]]
  return vuzp2_p8(a, b);
}

// LLVM-LABEL: @test_vuzp2q_p8(
// CIR-LABEL: @vuzp2q_p8(
poly8x16_t test_vuzp2q_p8(poly8x16_t a, poly8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i, #cir.int<17> : !s64i, #cir.int<19> : !s64i, #cir.int<21> : !s64i, #cir.int<23> : !s64i, #cir.int<25> : !s64i, #cir.int<27> : !s64i, #cir.int<29> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// LLVM: ret <16 x i8> [[VUZP]]
  return vuzp2q_p8(a, b);
}

// LLVM-LABEL: @test_vuzp2_p16(
// CIR-LABEL: @vuzp2_p16(
poly16x4_t test_vuzp2_p16(poly16x4_t a, poly16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// LLVM: ret <4 x i16> [[VUZP]]
  return vuzp2_p16(a, b);
}

// LLVM-LABEL: @test_vuzp2q_p16(
// CIR-LABEL: @vuzp2q_p16(
poly16x8_t test_vuzp2q_p16(poly16x8_t a, poly16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: ret <8 x i16> [[VUZP]]
  return vuzp2q_p16(a, b);
}

// LLVM-LABEL: @test_vuzp2_mf8(
// CIR-LABEL: @vuzp2_mf8(
mfloat8x8_t test_vuzp2_mf8(mfloat8x8_t a, mfloat8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: ret <8 x i8> [[VUZP]]
  return vuzp2_mf8(a, b);
}

// LLVM-LABEL: @test_vuzp2q_mf8(
// CIR-LABEL: @vuzp2q_mf8(
mfloat8x16_t test_vuzp2q_mf8(mfloat8x16_t a, mfloat8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i, #cir.int<5> : !s64i, #cir.int<7> : !s64i, #cir.int<9> : !s64i, #cir.int<11> : !s64i, #cir.int<13> : !s64i, #cir.int<15> : !s64i, #cir.int<17> : !s64i, #cir.int<19> : !s64i, #cir.int<21> : !s64i, #cir.int<23> : !s64i, #cir.int<25> : !s64i, #cir.int<27> : !s64i, #cir.int<29> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[VUZP:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// LLVM: ret <16 x i8> [[VUZP]]
  return vuzp2q_mf8(a, b);
}

//===------------------------------------------------------===//
// 2.1.9.14.  Unzip elements`
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#unzip-elements-1
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vuzpq_u16(
// CIR-LABEL: @vuzpq_u16(
uint16x8x2_t test_vuzpq_u16(uint16x8_t a, uint16x8_t b) {
// CIR: [[A_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
// CIR: [[B_CAST:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
// CIR: [[LO:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u16i>) [#cir.int<0> : !s32i, #cir.int<2> : !s32i, #cir.int<4> : !s32i, #cir.int<6> : !s32i, #cir.int<8> : !s32i, #cir.int<10> : !s32i, #cir.int<12> : !s32i, #cir.int<14> : !s32i] : !cir.vector<8 x !u16i>
// CIR: cir.store [[LO]], %{{.*}} : !cir.vector<8 x !u16i>, !cir.ptr<!cir.vector<8 x !u16i>>
// CIR: [[HI:%.*]] = cir.vec.shuffle([[A_CAST]], [[B_CAST]] : !cir.vector<8 x !u16i>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<15> : !s32i] : !cir.vector<8 x !u16i>
// CIR: cir.store [[HI]], %{{.*}} : !cir.vector<8 x !u16i>, !cir.ptr<!cir.vector<8 x !u16i>>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM: [[VUZP_LO:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// LLVM: [[VUZP_HI:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// LLVM: insertvalue %struct.uint16x8x2_t poison, <8 x i16> [[VUZP_LO]], 0, 0
// LLVM: insertvalue %struct.uint16x8x2_t {{.*}}, <8 x i16> [[VUZP_HI]], 0, 1
// LLVM: ret %struct.uint16x8x2_t
return vuzpq_u16(a, b);
}

//===------------------------------------------------------===//
// 2.1.9.12.  Transpose elements
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#transpose-elements
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vtrn1_s8(
// CIR-LABEL: @vtrn1_s8(
int8x8_t test_vtrn1_s8(int8x8_t a, int8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// LLVM: ret <8 x i8> [[SHUFFLE]]
  return vtrn1_s8(a, b);
}

// LLVM-LABEL: @test_vtrn1q_s8(
// CIR-LABEL: @vtrn1q_s8(
int8x16_t test_vtrn1q_s8(int8x16_t a, int8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<0> : !s64i, #cir.int<16> : !s64i, #cir.int<2> : !s64i, #cir.int<18> : !s64i, #cir.int<4> : !s64i, #cir.int<20> : !s64i, #cir.int<6> : !s64i, #cir.int<22> : !s64i, #cir.int<8> : !s64i, #cir.int<24> : !s64i, #cir.int<10> : !s64i, #cir.int<26> : !s64i, #cir.int<12> : !s64i, #cir.int<28> : !s64i, #cir.int<14> : !s64i, #cir.int<30> : !s64i] : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// LLVM: ret <16 x i8> [[SHUFFLE]]
  return vtrn1q_s8(a, b);
}

// LLVM-LABEL: @test_vtrn1_s16(
// CIR-LABEL: @vtrn1_s16(
int16x4_t test_vtrn1_s16(int16x4_t a, int16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s16i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<2> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// LLVM: ret <4 x i16> [[SHUFFLE]]
  return vtrn1_s16(a, b);
}

// LLVM-LABEL: @test_vtrn1q_s16(
// CIR-LABEL: @vtrn1q_s16(
int16x8_t test_vtrn1q_s16(int16x8_t a, int16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// LLVM: ret <8 x i16> [[SHUFFLE]]
  return vtrn1q_s16(a, b);
}

// LLVM-LABEL: @test_vtrn1_s32(
// CIR-LABEL: @vtrn1_s32(
int32x2_t test_vtrn1_s32(int32x2_t a, int32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i32> [[SHUFFLE]]
  return vtrn1_s32(a, b);
}

// LLVM-LABEL: @test_vtrn1q_s32(
// CIR-LABEL: @vtrn1q_s32(
int32x4_t test_vtrn1q_s32(int32x4_t a, int32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<2> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// LLVM: ret <4 x i32> [[SHUFFLE]]
  return vtrn1q_s32(a, b);
}

// LLVM-LABEL: @test_vtrn1q_s64(
// CIR-LABEL: @vtrn1q_s64(
int64x2_t test_vtrn1q_s64(int64x2_t a, int64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[SHUFFLE]]
  return vtrn1q_s64(a, b);
}

// LLVM-LABEL: @test_vtrn1_u8(
// CIR-LABEL: @vtrn1_u8(
uint8x8_t test_vtrn1_u8(uint8x8_t a, uint8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// LLVM: ret <8 x i8> [[SHUFFLE]]
  return vtrn1_u8(a, b);
}

// LLVM-LABEL: @test_vtrn1q_u8(
// CIR-LABEL: @vtrn1q_u8(
uint8x16_t test_vtrn1q_u8(uint8x16_t a, uint8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<16> : !s64i, #cir.int<2> : !s64i, #cir.int<18> : !s64i, #cir.int<4> : !s64i, #cir.int<20> : !s64i, #cir.int<6> : !s64i, #cir.int<22> : !s64i, #cir.int<8> : !s64i, #cir.int<24> : !s64i, #cir.int<10> : !s64i, #cir.int<26> : !s64i, #cir.int<12> : !s64i, #cir.int<28> : !s64i, #cir.int<14> : !s64i, #cir.int<30> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// LLVM: ret <16 x i8> [[SHUFFLE]]
  return vtrn1q_u8(a, b);
}

// LLVM-LABEL: @test_vtrn1_u16(
// CIR-LABEL: @vtrn1_u16(
uint16x4_t test_vtrn1_u16(uint16x4_t a, uint16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<2> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// LLVM: ret <4 x i16> [[SHUFFLE]]
  return vtrn1_u16(a, b);
}

// LLVM-LABEL: @test_vtrn1q_u16(
// CIR-LABEL: @vtrn1q_u16(
uint16x8_t test_vtrn1q_u16(uint16x8_t a, uint16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// LLVM: ret <8 x i16> [[SHUFFLE]]
  return vtrn1q_u16(a, b);
}

// LLVM-LABEL: @test_vtrn1_u32(
// CIR-LABEL: @vtrn1_u32(
uint32x2_t test_vtrn1_u32(uint32x2_t a, uint32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u32i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i32> [[SHUFFLE]]
  return vtrn1_u32(a, b);
}

// LLVM-LABEL: @test_vtrn1q_u32(
// CIR-LABEL: @vtrn1q_u32(
uint32x4_t test_vtrn1q_u32(uint32x4_t a, uint32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u32i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<2> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// LLVM: ret <4 x i32> [[SHUFFLE]]
  return vtrn1q_u32(a, b);
}

// LLVM-LABEL: @test_vtrn1q_u64(
// CIR-LABEL: @vtrn1q_u64(
uint64x2_t test_vtrn1q_u64(uint64x2_t a, uint64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[SHUFFLE]]
  return vtrn1q_u64(a, b);
}

// LLVM-LABEL: @test_vtrn1q_p64(
// CIR-LABEL: @vtrn1q_p64(
poly64x2_t test_vtrn1q_p64(poly64x2_t a, poly64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x i64> [[SHUFFLE]]
  return vtrn1q_p64(a, b);
}

// LLVM-LABEL: @test_vtrn1_f32(
// CIR-LABEL: @vtrn1_f32(
float32x2_t test_vtrn1_f32(float32x2_t a, float32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x float> [[SHUFFLE]]
  return vtrn1_f32(a, b);
}

// LLVM-LABEL: @test_vtrn1q_f32(
// CIR-LABEL: @vtrn1q_f32(
float32x4_t test_vtrn1q_f32(float32x4_t a, float32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<2> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float>{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// LLVM: ret <4 x float> [[SHUFFLE]]
  return vtrn1q_f32(a, b);
}

// LLVM-LABEL: @test_vtrn1q_f64(
// CIR-LABEL: @vtrn1q_f64(
float64x2_t test_vtrn1q_f64(float64x2_t a, float64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s64i, #cir.int<2> : !s64i] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x double> [[A]], <2 x double> [[B]], <2 x i32> <i32 0, i32 2>
// LLVM: ret <2 x double> [[SHUFFLE]]
  return vtrn1q_f64(a, b);
}

// LLVM-LABEL: @test_vtrn1_p8(
// CIR-LABEL: @vtrn1_p8(
poly8x8_t test_vtrn1_p8(poly8x8_t a, poly8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// LLVM: ret <8 x i8> [[SHUFFLE]]
  return vtrn1_p8(a, b);
}

// LLVM-LABEL: @test_vtrn1q_p8(
// CIR-LABEL: @vtrn1q_p8(
poly8x16_t test_vtrn1q_p8(poly8x16_t a, poly8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<16> : !s64i, #cir.int<2> : !s64i, #cir.int<18> : !s64i, #cir.int<4> : !s64i, #cir.int<20> : !s64i, #cir.int<6> : !s64i, #cir.int<22> : !s64i, #cir.int<8> : !s64i, #cir.int<24> : !s64i, #cir.int<10> : !s64i, #cir.int<26> : !s64i, #cir.int<12> : !s64i, #cir.int<28> : !s64i, #cir.int<14> : !s64i, #cir.int<30> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// LLVM: ret <16 x i8> [[SHUFFLE]]
  return vtrn1q_p8(a, b);
}

// LLVM-LABEL: @test_vtrn1_p16(
// CIR-LABEL: @vtrn1_p16(
poly16x4_t test_vtrn1_p16(poly16x4_t a, poly16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<0> : !s64i, #cir.int<4> : !s64i, #cir.int<2> : !s64i, #cir.int<6> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// LLVM: ret <4 x i16> [[SHUFFLE]]
  return vtrn1_p16(a, b);
}

// LLVM-LABEL: @test_vtrn1q_p16(
// CIR-LABEL: @vtrn1q_p16(
poly16x8_t test_vtrn1q_p16(poly16x8_t a, poly16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// LLVM: ret <8 x i16> [[SHUFFLE]]
  return vtrn1q_p16(a, b);
}

// LLVM-LABEL: @test_vtrn1_mf8(
// CIR-LABEL: @vtrn1_mf8(
mfloat8x8_t test_vtrn1_mf8(mfloat8x8_t a, mfloat8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<0> : !s64i, #cir.int<8> : !s64i, #cir.int<2> : !s64i, #cir.int<10> : !s64i, #cir.int<4> : !s64i, #cir.int<12> : !s64i, #cir.int<6> : !s64i, #cir.int<14> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// LLVM: ret <8 x i8> [[SHUFFLE]]
  return vtrn1_mf8(a, b);
}

// LLVM-LABEL: @test_vtrn1q_mf8(
// CIR-LABEL: @vtrn1q_mf8(
mfloat8x16_t test_vtrn1q_mf8(mfloat8x16_t a, mfloat8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<0> : !s64i, #cir.int<16> : !s64i, #cir.int<2> : !s64i, #cir.int<18> : !s64i, #cir.int<4> : !s64i, #cir.int<20> : !s64i, #cir.int<6> : !s64i, #cir.int<22> : !s64i, #cir.int<8> : !s64i, #cir.int<24> : !s64i, #cir.int<10> : !s64i, #cir.int<26> : !s64i, #cir.int<12> : !s64i, #cir.int<28> : !s64i, #cir.int<14> : !s64i, #cir.int<30> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// LLVM: ret <16 x i8> [[SHUFFLE]]
  return vtrn1q_mf8(a, b);
}

// LLVM-LABEL: @test_vtrn2_s8(
// CIR-LABEL: @vtrn2_s8(
int8x8_t test_vtrn2_s8(int8x8_t a, int8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>) [#cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// LLVM: ret <8 x i8> [[SHUFFLE]]
  return vtrn2_s8(a, b);
}

// LLVM-LABEL: @test_vtrn2q_s8(
// CIR-LABEL: @vtrn2q_s8(
int8x16_t test_vtrn2q_s8(int8x16_t a, int8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>) [#cir.int<1> : !s64i, #cir.int<17> : !s64i, #cir.int<3> : !s64i, #cir.int<19> : !s64i, #cir.int<5> : !s64i, #cir.int<21> : !s64i, #cir.int<7> : !s64i, #cir.int<23> : !s64i, #cir.int<9> : !s64i, #cir.int<25> : !s64i, #cir.int<11> : !s64i, #cir.int<27> : !s64i, #cir.int<13> : !s64i, #cir.int<29> : !s64i, #cir.int<15> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// LLVM: ret <16 x i8> [[SHUFFLE]]
  return vtrn2q_s8(a, b);
}

// LLVM-LABEL: @test_vtrn2_s16(
// CIR-LABEL: @vtrn2_s16(
int16x4_t test_vtrn2_s16(int16x4_t a, int16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s16i>) [#cir.int<1> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// LLVM: ret <4 x i16> [[SHUFFLE]]
  return vtrn2_s16(a, b);
}

// LLVM-LABEL: @test_vtrn2q_s16(
// CIR-LABEL: @vtrn2q_s16(
int16x8_t test_vtrn2q_s16(int16x8_t a, int16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// LLVM: ret <8 x i16> [[SHUFFLE]]
  return vtrn2q_s16(a, b);
}

// LLVM-LABEL: @test_vtrn2_s32(
// CIR-LABEL: @vtrn2_s32(
int32x2_t test_vtrn2_s32(int32x2_t a, int32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s32i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i32> [[SHUFFLE]]
  return vtrn2_s32(a, b);
}

// LLVM-LABEL: @test_vtrn2q_s32(
// CIR-LABEL: @vtrn2q_s32(
int32x4_t test_vtrn2q_s32(int32x4_t a, int32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<1> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// LLVM: ret <4 x i32> [[SHUFFLE]]
  return vtrn2q_s32(a, b);
}

// LLVM-LABEL: @test_vtrn2q_s64(
// CIR-LABEL: @vtrn2q_s64(
int64x2_t test_vtrn2q_s64(int64x2_t a, int64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !s64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[SHUFFLE]]
  return vtrn2q_s64(a, b);
}

// LLVM-LABEL: @test_vtrn2_u8(
// CIR-LABEL: @vtrn2_u8(
uint8x8_t test_vtrn2_u8(uint8x8_t a, uint8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// LLVM: ret <8 x i8> [[SHUFFLE]]
  return vtrn2_u8(a, b);
}

// LLVM-LABEL: @test_vtrn2q_u8(
// CIR-LABEL: @vtrn2q_u8(
uint8x16_t test_vtrn2q_u8(uint8x16_t a, uint8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<1> : !s64i, #cir.int<17> : !s64i, #cir.int<3> : !s64i, #cir.int<19> : !s64i, #cir.int<5> : !s64i, #cir.int<21> : !s64i, #cir.int<7> : !s64i, #cir.int<23> : !s64i, #cir.int<9> : !s64i, #cir.int<25> : !s64i, #cir.int<11> : !s64i, #cir.int<27> : !s64i, #cir.int<13> : !s64i, #cir.int<29> : !s64i, #cir.int<15> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// LLVM: ret <16 x i8> [[SHUFFLE]]
  return vtrn2q_u8(a, b);
}

// LLVM-LABEL: @test_vtrn2_u16(
// CIR-LABEL: @vtrn2_u16(
uint16x4_t test_vtrn2_u16(uint16x4_t a, uint16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<1> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// LLVM: ret <4 x i16> [[SHUFFLE]]
  return vtrn2_u16(a, b);
}

// LLVM-LABEL: @test_vtrn2q_u16(
// CIR-LABEL: @vtrn2q_u16(
uint16x8_t test_vtrn2q_u16(uint16x8_t a, uint16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// LLVM: ret <8 x i16> [[SHUFFLE]]
  return vtrn2q_u16(a, b);
}

// LLVM-LABEL: @test_vtrn2_u32(
// CIR-LABEL: @vtrn2_u32(
uint32x2_t test_vtrn2_u32(uint32x2_t a, uint32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u32i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i32> [[SHUFFLE]]
  return vtrn2_u32(a, b);
}

// LLVM-LABEL: @test_vtrn2q_u32(
// CIR-LABEL: @vtrn2q_u32(
uint32x4_t test_vtrn2q_u32(uint32x4_t a, uint32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u32i>) [#cir.int<1> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// LLVM: ret <4 x i32> [[SHUFFLE]]
  return vtrn2q_u32(a, b);
}

// LLVM-LABEL: @test_vtrn2q_u64(
// CIR-LABEL: @vtrn2q_u64(
uint64x2_t test_vtrn2q_u64(uint64x2_t a, uint64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[SHUFFLE]]
  return vtrn2q_u64(a, b);
}

// LLVM-LABEL: @test_vtrn2q_p64(
// CIR-LABEL: @vtrn2q_p64(
poly64x2_t test_vtrn2q_p64(poly64x2_t a, poly64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !u64i>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x i64> [[SHUFFLE]]
  return vtrn2q_p64(a, b);
}

// LLVM-LABEL: @test_vtrn2_f32(
// CIR-LABEL: @vtrn2_f32(
float32x2_t test_vtrn2_f32(float32x2_t a, float32x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x float> [[SHUFFLE]]
  return vtrn2_f32(a, b);
}

// LLVM-LABEL: @test_vtrn2q_f32(
// CIR-LABEL: @vtrn2q_f32(
float32x4_t test_vtrn2q_f32(float32x4_t a, float32x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<1> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float>{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// LLVM: ret <4 x float> [[SHUFFLE]]
  return vtrn2q_f32(a, b);
}

// LLVM-LABEL: @test_vtrn2q_f64(
// CIR-LABEL: @vtrn2q_f64(
float64x2_t test_vtrn2q_f64(float64x2_t a, float64x2_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<1> : !s64i, #cir.int<3> : !s64i] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <2 x double> [[A]], <2 x double> [[B]], <2 x i32> <i32 1, i32 3>
// LLVM: ret <2 x double> [[SHUFFLE]]
  return vtrn2q_f64(a, b);
}

// LLVM-LABEL: @test_vtrn2_p8(
// CIR-LABEL: @vtrn2_p8(
poly8x8_t test_vtrn2_p8(poly8x8_t a, poly8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// LLVM: ret <8 x i8> [[SHUFFLE]]
  return vtrn2_p8(a, b);
}

// LLVM-LABEL: @test_vtrn2q_p8(
// CIR-LABEL: @vtrn2q_p8(
poly8x16_t test_vtrn2q_p8(poly8x16_t a, poly8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<1> : !s64i, #cir.int<17> : !s64i, #cir.int<3> : !s64i, #cir.int<19> : !s64i, #cir.int<5> : !s64i, #cir.int<21> : !s64i, #cir.int<7> : !s64i, #cir.int<23> : !s64i, #cir.int<9> : !s64i, #cir.int<25> : !s64i, #cir.int<11> : !s64i, #cir.int<27> : !s64i, #cir.int<13> : !s64i, #cir.int<29> : !s64i, #cir.int<15> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// LLVM: ret <16 x i8> [[SHUFFLE]]
  return vtrn2q_p8(a, b);
}

// LLVM-LABEL: @test_vtrn2_p16(
// CIR-LABEL: @vtrn2_p16(
poly16x4_t test_vtrn2_p16(poly16x4_t a, poly16x4_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !u16i>) [#cir.int<1> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<7> : !s64i] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// LLVM: ret <4 x i16> [[SHUFFLE]]
  return vtrn2_p16(a, b);
}

// LLVM-LABEL: @test_vtrn2q_p16(
// CIR-LABEL: @vtrn2q_p16(
poly16x8_t test_vtrn2q_p16(poly16x8_t a, poly16x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u16i>) [#cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// LLVM: ret <8 x i16> [[SHUFFLE]]
  return vtrn2q_p16(a, b);
}

// LLVM-LABEL: @test_vtrn2_mf8(
// CIR-LABEL: @vtrn2_mf8(
mfloat8x8_t test_vtrn2_mf8(mfloat8x8_t a, mfloat8x8_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !u8i>) [#cir.int<1> : !s64i, #cir.int<9> : !s64i, #cir.int<3> : !s64i, #cir.int<11> : !s64i, #cir.int<5> : !s64i, #cir.int<13> : !s64i, #cir.int<7> : !s64i, #cir.int<15> : !s64i] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// LLVM: ret <8 x i8> [[SHUFFLE]]
  return vtrn2_mf8(a, b);
}

// LLVM-LABEL: @test_vtrn2q_mf8(
// CIR-LABEL: @vtrn2q_mf8(
mfloat8x16_t test_vtrn2q_mf8(mfloat8x16_t a, mfloat8x16_t b) {
// CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<16 x !u8i>) [#cir.int<1> : !s64i, #cir.int<17> : !s64i, #cir.int<3> : !s64i, #cir.int<19> : !s64i, #cir.int<5> : !s64i, #cir.int<21> : !s64i, #cir.int<7> : !s64i, #cir.int<23> : !s64i, #cir.int<9> : !s64i, #cir.int<25> : !s64i, #cir.int<11> : !s64i, #cir.int<27> : !s64i, #cir.int<13> : !s64i, #cir.int<29> : !s64i, #cir.int<15> : !s64i, #cir.int<31> : !s64i] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[SHUFFLE:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// LLVM: ret <16 x i8> [[SHUFFLE]]
  return vtrn2q_mf8(a, b);
}
