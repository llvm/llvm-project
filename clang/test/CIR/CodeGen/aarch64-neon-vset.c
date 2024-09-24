// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -emit-llvm -target-feature +neon %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test file is similar to but not the same as 
// clang/test/CodeGen/aarch64-neon-vget.c 
// The difference is that this file only tests uses vset intrinsics, as we feel
// it would be proper to have a separate test file testing vget intrinsics 
// with the file name aarch64-neon-vget.c 
// Also, for each integer type, we only test signed or unsigned, not both. 
// This is because integer types of the same size just use same intrinsic.

// REQUIRES: aarch64-registered-target || arm-registered-target
#include <arm_neon.h>

uint8x8_t test_vset_lane_u8(uint8_t a, uint8x8_t b) {
  return vset_lane_u8(a, b, 7);
}

// CIR-LABEL: test_vset_lane_u8
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i loc(#loc7)
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s8i x 8>

// LLVM: define dso_local <8 x i8> @test_vset_lane_u8(i8 [[A:%.*]], <8 x i8> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca i8, i64 1, align 1
// LLVM: [[B_ADR:%.*]] = alloca <8 x i8>, i64 1, align 8
// LLVM: store i8 [[A]], ptr [[A_ADR]], align 1
// LLVM: store <8 x i8> [[B]], ptr [[B_ADR]], align 8
// LLVM: [[TMP_A0:%.*]] = load i8, ptr [[A_ADR]], align 1
// LLVM: store i8 [[TMP_A0]], ptr [[S0:%.*]], align 1
// LLVM: [[TMP_B0:%.*]] = load <8 x i8>, ptr [[B_ADR]], align 8
// LLVM: store <8 x i8> [[TMP_B0]], ptr [[S1:%.*]], align 8
// LLVM: [[INTRN_ARG0:%.*]] = load i8, ptr [[S0]], align 1
// LLVM: [[INTRN_ARG1:%.*]] = load <8 x i8>, ptr [[S1]], align 8
// LLVM: [[INTRN_RES:%.*]] = insertelement <8 x i8> [[INTRN_ARG1]], i8 [[INTRN_ARG0]], i32 7
// LLVM: ret <8 x i8> {{%.*}}

uint16x4_t test_vset_lane_u16(uint16_t a, uint16x4_t b) {
  return vset_lane_u16(a, b, 3);
}

// CIR-LABEL: test_vset_lane_u16
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s16i x 4>

// LLVM: define dso_local <4 x i16> @test_vset_lane_u16(i16 [[A:%.*]], <4 x i16> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca i16, i64 1, align 2
// LLVM: [[B_ADR:%.*]] = alloca <4 x i16>, i64 1, align 8
// LLVM: store i16 [[A]], ptr [[A_ADR]], align 2
// LLVM: store <4 x i16> [[B]], ptr [[B_ADR]], align 8
// LLVM: [[TMP_A0:%.*]] = load i16, ptr [[A_ADR]], align 2
// LLVM: store i16 [[TMP_A0]], ptr [[S0:%.*]], align 2
// LLVM: [[TMP_B0:%.*]] = load <4 x i16>, ptr [[B_ADR]], align 8
// LLVM: store <4 x i16> [[TMP_B0]], ptr [[S1:%.*]], align 8
// LLVM: [[INTRN_ARG0:%.*]] = load i16, ptr [[S0]], align 2
// LLVM: [[INTRN_ARG1:%.*]] = load <4 x i16>, ptr [[S1]], align 8
// LLVM: [[INTRN_RES:%.*]] = insertelement <4 x i16> [[INTRN_ARG1]], i16 [[INTRN_ARG0]], i32 3
// LLVM: ret <4 x i16> {{%.*}}

uint32x2_t test_vset_lane_u32(uint32_t a, uint32x2_t b) {
  return vset_lane_u32(a, b, 1);
}

// CIR-LABEL: test_vset_lane_u32
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s32i x 2>

// LLVM: define dso_local <2 x i32> @test_vset_lane_u32(i32 [[A:%.*]], <2 x i32> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca i32, i64 1, align 4
// LLVM: [[B_ADR:%.*]] = alloca <2 x i32>, i64 1, align 8
// LLVM: store i32 [[A]], ptr [[A_ADR]], align 4
// LLVM: store <2 x i32> [[B]], ptr [[B_ADR]], align 8
// LLVM: [[TMP_A0:%.*]] = load i32, ptr [[A_ADR]], align 4
// LLVM: store i32 [[TMP_A0]], ptr [[S0:%.*]], align 4
// LLVM: [[TMP_B0:%.*]] = load <2 x i32>, ptr [[B_ADR]], align 8
// LLVM: store <2 x i32> [[TMP_B0]], ptr [[S1:%.*]], align 8
// LLVM: [[INTRN_ARG0:%.*]] = load i32, ptr [[S0]], align 4
// LLVM: [[INTRN_ARG1:%.*]] = load <2 x i32>, ptr [[S1]], align 8
// LLVM: [[INTRN_RES:%.*]] = insertelement <2 x i32> [[INTRN_ARG1]], i32 [[INTRN_ARG0]], i32 1
// LLVM: ret <2 x i32> {{%.*}}


int64x1_t test_vset_lane_u64(int64_t a, int64x1_t b) {
  return vset_lane_u64(a, b, 0);
}

// CIR-LABEL: test_vset_lane_u64
// CIR: [[IDX:%.*]] = cir.const #cir.int<0> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s64i x 1>

// LLVM: define dso_local <1 x i64> @test_vset_lane_u64(i64 [[A:%.*]], <1 x i64> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca i64, i64 1, align 8
// LLVM: [[B_ADR:%.*]] = alloca <1 x i64>, i64 1, align 8
// LLVM: store i64 [[A]], ptr [[A_ADR]], align 8
// LLVM: store <1 x i64> [[B]], ptr [[B_ADR]], align 8
// LLVM: [[TMP_A0:%.*]] = load i64, ptr [[A_ADR]], align 8
// LLVM: store i64 [[TMP_A0]], ptr [[S0:%.*]], align 8
// LLVM: [[TMP_B0:%.*]] = load <1 x i64>, ptr [[B_ADR]], align 8
// LLVM: store <1 x i64> [[TMP_B0]], ptr [[S1:%.*]], align 8
// LLVM: [[INTRN_ARG0:%.*]] = load i64, ptr [[S0]], align 8
// LLVM: [[INTRN_ARG1:%.*]] = load <1 x i64>, ptr [[S1]], align 8
// LLVM: [[INTRN_RES:%.*]] = insertelement <1 x i64> [[INTRN_ARG1]], i64 [[INTRN_ARG0]], i32 0
// LLVM: ret <1 x i64> {{%.*}}

float32x2_t test_vset_lane_f32(float32_t a, float32x2_t b) {
  return vset_lane_f32(a, b, 1);
}

// CIR-LABEL: test_vset_lane_f32
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.float x 2>

// LLVM: define dso_local <2 x float> @test_vset_lane_f32(float [[A:%.*]], <2 x float> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca float, i64 1, align 4
// LLVM: [[B_ADR:%.*]] = alloca <2 x float>, i64 1, align 8
// LLVM: store float [[A]], ptr [[A_ADR]], align 4
// LLVM: store <2 x float> [[B]], ptr [[B_ADR]], align 8
// LLVM: [[TMP_A0:%.*]] = load float, ptr [[A_ADR]], align 4
// LLVM: store float [[TMP_A0]], ptr [[S0:%.*]], align 4
// LLVM: [[TMP_B0:%.*]] = load <2 x float>, ptr [[B_ADR]], align 8
// LLVM: store <2 x float> [[TMP_B0]], ptr [[S1:%.*]], align 8
// LLVM: [[INTRN_ARG0:%.*]] = load float, ptr [[S0]], align 4
// LLVM: [[INTRN_ARG1:%.*]] = load <2 x float>, ptr [[S1]], align 8
// LLVM: [[INTRN_RES:%.*]] = insertelement <2 x float> [[INTRN_ARG1]], float [[INTRN_ARG0]], i32 1
// LLVM: ret <2 x float> {{%.*}}

uint8x16_t test_vsetq_lane_u8(uint8_t a, uint8x16_t b) {
  return vsetq_lane_u8(a, b, 15);
}

// CIR-LABEL: test_vsetq_lane_u8
// CIR: [[IDX:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s8i x 16>

// LLVM: define dso_local <16 x i8> @test_vsetq_lane_u8(i8 [[A:%.*]], <16 x i8> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca i8, i64 1, align 1
// LLVM: [[B_ADR:%.*]] = alloca <16 x i8>, i64 1, align 16
// LLVM: store i8 [[A]], ptr [[A_ADR]], align 1
// LLVM: store <16 x i8> [[B]], ptr [[B_ADR]], align 16
// LLVM: [[TMP_A0:%.*]] = load i8, ptr [[A_ADR]], align 1
// LLVM: store i8 [[TMP_A0]], ptr [[S0:%.*]], align 1
// LLVM: [[TMP_B0:%.*]] = load <16 x i8>, ptr [[B_ADR]], align 16
// LLVM: store <16 x i8> [[TMP_B0]], ptr [[S1:%.*]], align 16
// LLVM: [[INTRN_ARG0:%.*]] = load i8, ptr [[S0]], align 1
// LLVM: [[INTRN_ARG1:%.*]] = load <16 x i8>, ptr [[S1]], align 16
// LLVM: [[INTRN_RES:%.*]] = insertelement <16 x i8> [[INTRN_ARG1]], i8 [[INTRN_ARG0]], i32 15
// LLVM: ret <16 x i8> {{%.*}}

uint16x8_t test_vsetq_lane_u16(uint16_t a, uint16x8_t b) {
  return vsetq_lane_u16(a, b, 7);
}

// CIR-LABEL: test_vsetq_lane_u16
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s16i x 8>

// LLVM: define dso_local <8 x i16> @test_vsetq_lane_u16(i16 [[A:%.*]], <8 x i16> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca i16, i64 1, align 2
// LLVM: [[B_ADR:%.*]] = alloca <8 x i16>, i64 1, align 16
// LLVM: store i16 [[A]], ptr [[A_ADR]], align 2
// LLVM: store <8 x i16> [[B]], ptr [[B_ADR]], align 16
// LLVM: [[TMP_A0:%.*]] = load i16, ptr [[A_ADR]], align 2
// LLVM: store i16 [[TMP_A0]], ptr [[S0:%.*]], align 2
// LLVM: [[TMP_B0:%.*]] = load <8 x i16>, ptr [[B_ADR]], align 16
// LLVM: store <8 x i16> [[TMP_B0]], ptr [[S1:%.*]], align 16
// LLVM: [[INTRN_ARG0:%.*]] = load i16, ptr [[S0]], align 2
// LLVM: [[INTRN_ARG1:%.*]] = load <8 x i16>, ptr [[S1]], align 16
// LLVM: [[INTRN_RES:%.*]] = insertelement <8 x i16> [[INTRN_ARG1]], i16 [[INTRN_ARG0]], i32 7
// LLVM: ret <8 x i16> {{%.*}}

uint32x4_t test_vsetq_lane_u32(uint32_t a, uint32x4_t b) {
  return vsetq_lane_u32(a, b, 3);
}

// CIR-LABEL: test_vsetq_lane_u32
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s32i x 4>

// LLVM: define dso_local <4 x i32> @test_vsetq_lane_u32(i32 [[A:%.*]], <4 x i32> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca i32, i64 1, align 4
// LLVM: [[B_ADR:%.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store i32 [[A]], ptr [[A_ADR]], align 4
// LLVM: store <4 x i32> [[B]], ptr [[B_ADR]], align 16
// LLVM: [[TMP_A0:%.*]] = load i32, ptr [[A_ADR]], align 4
// LLVM: store i32 [[TMP_A0]], ptr [[S0:%.*]], align 4
// LLVM: [[TMP_B0:%.*]] = load <4 x i32>, ptr [[B_ADR]], align 16
// LLVM: store <4 x i32> [[TMP_B0]], ptr [[S1:%.*]], align 16
// LLVM: [[INTRN_ARG0:%.*]] = load i32, ptr [[S0]], align 4
// LLVM: [[INTRN_ARG1:%.*]] = load <4 x i32>, ptr [[S1]], align 16
// LLVM: [[INTRN_RES:%.*]] = insertelement <4 x i32> [[INTRN_ARG1]], i32 [[INTRN_ARG0]], i32 3
// LLVM: ret <4 x i32> {{%.*}}

int64x2_t test_vsetq_lane_s64(int64_t a, int64x2_t b) {
  return vsetq_lane_s64(a, b, 1);
}

// CIR-LABEL: test_vsetq_lane_s64
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s64i x 2>

// LLVM: define dso_local <2 x i64> @test_vsetq_lane_s64(i64 [[A:%.*]], <2 x i64> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca i64, i64 1, align 8
// LLVM: [[B_ADR:%.*]] = alloca <2 x i64>, i64 1, align 16
// LLVM: store i64 [[A]], ptr [[A_ADR]], align 8
// LLVM: store <2 x i64> [[B]], ptr [[B_ADR]], align 16
// LLVM: [[TMP_A0:%.*]] = load i64, ptr [[A_ADR]], align 8
// LLVM: store i64 [[TMP_A0]], ptr [[S0:%.*]], align 8
// LLVM: [[TMP_B0:%.*]] = load <2 x i64>, ptr [[B_ADR]], align 16
// LLVM: store <2 x i64> [[TMP_B0]], ptr [[S1:%.*]], align 16
// LLVM: [[INTRN_ARG0:%.*]] = load i64, ptr [[S0]], align 8
// LLVM: [[INTRN_ARG1:%.*]] = load <2 x i64>, ptr [[S1]], align 16
// LLVM: [[INTRN_RES:%.*]] = insertelement <2 x i64> [[INTRN_ARG1]], i64 [[INTRN_ARG0]], i32 1
// LLVM: ret <2 x i64> {{%.*}}

float32x4_t test_vsetq_lane_f32(float32_t a, float32x4_t b) {
  return vsetq_lane_f32(a, b, 3);
}

// CIR-LABEL: test_vsetq_lane_f32
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.float x 4>

// LLVM: define dso_local <4 x float> @test_vsetq_lane_f32(float [[A:%.*]], <4 x float> [[B:%.*]])
// LLVM: [[A_ADR:%.*]] = alloca float, i64 1, align 4
// LLVM: [[B_ADR:%.*]] = alloca <4 x float>, i64 1, align 16
// LLVM: store float [[A]], ptr [[A_ADR]], align 4
// LLVM: store <4 x float> [[B]], ptr [[B_ADR]], align 16
// LLVM: [[TMP_A0:%.*]] = load float, ptr [[A_ADR]], align 4
// LLVM: store float [[TMP_A0]], ptr [[S0:%.*]], align 4
// LLVM: [[TMP_B0:%.*]] = load <4 x float>, ptr [[B_ADR]], align 16
// LLVM: store <4 x float> [[TMP_B0]], ptr [[S1:%.*]], align 16
// LLVM: [[INTRN_ARG0:%.*]] = load float, ptr [[S0]], align 4
// LLVM: [[INTRN_ARG1:%.*]] = load <4 x float>, ptr [[S1]], align 16
// LLVM: [[INTRN_RES:%.*]] = insertelement <4 x float> [[INTRN_ARG1]], float [[INTRN_ARG0]], i32 3
// LLVM: ret <4 x float> {{%.*}}
