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
// TODO: Migrate the unzip elements intrinsics in https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#unzip-elements
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
