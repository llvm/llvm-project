// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-llvm -target-feature +neon %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// This test file contains tests for the AArch64 NEON load/store intrinsics.

#include <arm_neon.h>

int8x8_t test_vld1_lane_s8(int8_t const * ptr, int8x8_t src) {
    return vld1_lane_s8(ptr, src, 7);
}

// CIR-LABEL: test_vld1_lane_s8
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!s8i>
// CIR: [[VAL:%.*]] = cir.load align(1) [[PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vld1_lane_s8(ptr{{.*}}[[PTR:%.*]], <8 x i8>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <8 x i8> [[SRC]], ptr [[SRC_ADDR:%.*]], align 8
// LLVM: [[SRC_VAL:%.*]] = load <8 x i8>, ptr [[SRC_ADDR]], align 8
// LLVM:  store <8 x i8> [[SRC_VAL]], ptr [[S1:%.*]], align 8
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <8 x i8>, ptr [[S1]], align 8
// LLVM: [[INTRN_VAL:%.*]] = load i8, ptr [[PTR_VAL]], align 1
// LLVM: {{.*}} = insertelement <8 x i8> [[INTRN_VEC]], i8 [[INTRN_VAL]], i32 7
// LLVM: ret <8 x i8> {{.*}}

int8x16_t test_vld1q_lane_s8(int8_t const * ptr, int8x16_t src) {
    return vld1q_lane_s8(ptr, src, 15);
}

// CIR-LABEL: test_vld1q_lane_s8
// CIR: [[IDX:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!s8i>
// CIR: [[VAL:%.*]] = cir.load align(1) [[PTR]] : !cir.ptr<!s8i>, !s8i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vld1q_lane_s8(ptr{{.*}}[[PTR:%.*]], <16 x i8>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <16 x i8> [[SRC]], ptr [[SRC_ADDR:%.*]], align 16
// LLVM: [[SRC_VAL:%.*]] = load <16 x i8>, ptr [[SRC_ADDR]], align 16
// LLVM:  store <16 x i8> [[SRC_VAL]], ptr [[S1:%.*]], align 16
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <16 x i8>, ptr [[S1]], align 16
// LLVM: [[INTRN_VAL:%.*]] = load i8, ptr [[PTR_VAL]], align 1
// LLVM: {{.*}} = insertelement <16 x i8> [[INTRN_VEC]], i8 [[INTRN_VAL]], i32 15
// LLVM: ret <16 x i8> {{.*}}

uint8x16_t test_vld1q_lane_u8(uint8_t const * ptr, uint8x16_t src) {
    return vld1q_lane_u8(ptr, src, 15);
}

// CIR-LABEL: test_vld1q_lane_u8
// CIR: [[IDX:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!u8i>
// CIR: [[VAL:%.*]] = cir.load align(1) [[PTR]] : !cir.ptr<!u8i>, !u8i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u8i x 16>

// LLVM: {{.*}}test_vld1q_lane_u8(ptr{{.*}}[[PTR:%.*]], <16 x i8>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <16 x i8> [[SRC]], ptr [[SRC_ADDR:%.*]], align 16
// LLVM: [[SRC_VAL:%.*]] = load <16 x i8>, ptr [[SRC_ADDR]], align 16
// LLVM:  store <16 x i8> [[SRC_VAL]], ptr [[S1:%.*]], align 16
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <16 x i8>, ptr [[S1]], align 16
// LLVM: [[INTRN_VAL:%.*]] = load i8, ptr [[PTR_VAL]], align 1
// LLVM: {{.*}} = insertelement <16 x i8> [[INTRN_VEC]], i8 [[INTRN_VAL]], i32 15
// LLVM: ret <16 x i8> {{.*}}


uint8x8_t test_vld1_lane_u8(uint8_t const * ptr, uint8x8_t src) {
    return vld1_lane_u8(ptr, src, 7);
}

// CIR-LABEL: test_vld1_lane_u8
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!u8i>
// CIR: [[VAL:%.*]] = cir.load align(1) [[PTR]] : !cir.ptr<!u8i>, !u8i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u8i x 8>

// LLVM: {{.*}}test_vld1_lane_u8(ptr{{.*}}[[PTR:%.*]], <8 x i8>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <8 x i8> [[SRC]], ptr [[SRC_ADDR:%.*]], align 8
// LLVM: [[SRC_VAL:%.*]] = load <8 x i8>, ptr [[SRC_ADDR]], align 8
// LLVM:  store <8 x i8> [[SRC_VAL]], ptr [[S1:%.*]], align 8
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <8 x i8>, ptr [[S1]], align 8
// LLVM: [[INTRN_VAL:%.*]] = load i8, ptr [[PTR_VAL]], align 1
// LLVM: {{.*}} = insertelement <8 x i8> [[INTRN_VEC]], i8 [[INTRN_VAL]], i32 7
// LLVM: ret <8 x i8> {{.*}}


int16x4_t test_vld1_lane_s16(int16_t const * ptr, int16x4_t src) {
    return vld1_lane_s16(ptr, src, 3);
}

// CIR-LABEL: test_vld1_lane_s16
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!s16i>
// CIR: [[VAL:%.*]] = cir.load align(2) [[PTR]] : !cir.ptr<!s16i>, !s16i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s16i x 4>

// LLVM: {{.*}}test_vld1_lane_s16(ptr{{.*}}[[PTR:%.*]], <4 x i16>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <4 x i16> [[SRC]], ptr [[SRC_ADDR:%.*]], align 8
// LLVM: [[SRC_VAL:%.*]] = load <4 x i16>, ptr [[SRC_ADDR]], align 8
// LLVM:  store <4 x i16> [[SRC_VAL]], ptr [[S1:%.*]], align 8
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <4 x i16>, ptr [[S1]], align 8
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <4 x i16> [[INTRN_VEC]] to <8 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <8 x i8> [[INTRN_VEC_CAST0]] to <4 x i16>
// LLVM: [[INTRN_VAL:%.*]] = load i16, ptr [[PTR_VAL]], align 2
// LLVM: {{.*}} = insertelement <4 x i16> [[INTRN_VEC_CAST1]], i16 [[INTRN_VAL]], i32 3
// LLVM: ret <4 x i16> {{.*}}

uint16x4_t test_vld1_lane_u16(uint16_t const * ptr, uint16x4_t src) {
    return vld1_lane_u16(ptr, src, 3);
}

// CIR-LABEL: test_vld1_lane_u16
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!u16i>
// CIR: [[VAL:%.*]] = cir.load align(2) [[PTR]] : !cir.ptr<!u16i>, !u16i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u16i x 4>

// LLVM: {{.*}}test_vld1_lane_u16(ptr{{.*}}[[PTR:%.*]], <4 x i16>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <4 x i16> [[SRC]], ptr [[SRC_ADDR:%.*]], align 8
// LLVM: [[SRC_VAL:%.*]] = load <4 x i16>, ptr [[SRC_ADDR]], align 8
// LLVM:  store <4 x i16> [[SRC_VAL]], ptr [[S1:%.*]], align 8
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <4 x i16>, ptr [[S1]], align 8
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <4 x i16> [[INTRN_VEC]] to <8 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <8 x i8> [[INTRN_VEC_CAST0]] to <4 x i16>
// LLVM: [[INTRN_VAL:%.*]] = load i16, ptr [[PTR_VAL]], align 2
// LLVM: {{.*}} = insertelement <4 x i16> [[INTRN_VEC_CAST1]], i16 [[INTRN_VAL]], i32 3
// LLVM: ret <4 x i16> {{.*}}

int16x8_t test_vld1q_lane_s16(int16_t const * ptr, int16x8_t src) {
    return vld1q_lane_s16(ptr, src, 7);
}

// CIR-LABEL: test_vld1q_lane_s16
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!s16i>
// CIR: [[VAL:%.*]] = cir.load align(2) [[PTR]] : !cir.ptr<!s16i>, !s16i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s16i x 8>

// LLVM: {{.*}}test_vld1q_lane_s16(ptr{{.*}}[[PTR:%.*]], <8 x i16>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <8 x i16> [[SRC]], ptr [[SRC_ADDR:%.*]], align 16
// LLVM: [[SRC_VAL:%.*]] = load <8 x i16>, ptr [[SRC_ADDR]], align 16
// LLVM:  store <8 x i16> [[SRC_VAL]], ptr [[S1:%.*]], align 16
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <8 x i16>, ptr [[S1]], align 16
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <8 x i16> [[INTRN_VEC]] to <16 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <16 x i8> [[INTRN_VEC_CAST0]] to <8 x i16>
// LLVM: [[INTRN_VAL:%.*]] = load i16, ptr [[PTR_VAL]], align 2
// LLVM: {{.*}} = insertelement <8 x i16> [[INTRN_VEC_CAST1]], i16 [[INTRN_VAL]], i32 7
// LLVM: ret <8 x i16> {{.*}}

uint16x8_t test_vld1q_lane_u16(uint16_t const * ptr, uint16x8_t src) {
    return vld1q_lane_u16(ptr, src, 7);
}

// CIR-LABEL: test_vld1q_lane_u16
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!u16i>
// CIR: [[VAL:%.*]] = cir.load align(2) [[PTR]] : !cir.ptr<!u16i>, !u16i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u16i x 8>

// LLVM: {{.*}}test_vld1q_lane_u16(ptr{{.*}}[[PTR:%.*]], <8 x i16>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <8 x i16> [[SRC]], ptr [[SRC_ADDR:%.*]], align 16
// LLVM: [[SRC_VAL:%.*]] = load <8 x i16>, ptr [[SRC_ADDR]], align 16
// LLVM:  store <8 x i16> [[SRC_VAL]], ptr [[S1:%.*]], align 16
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <8 x i16>, ptr [[S1]], align 16
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <8 x i16> [[INTRN_VEC]] to <16 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <16 x i8> [[INTRN_VEC_CAST0]] to <8 x i16>
// LLVM: [[INTRN_VAL:%.*]] = load i16, ptr [[PTR_VAL]], align 2
// LLVM: {{.*}} = insertelement <8 x i16> [[INTRN_VEC_CAST1]], i16 [[INTRN_VAL]], i32 7
// LLVM: ret <8 x i16> {{.*}}




int32x2_t test_vld1_lane_s32(int32_t const * ptr, int32x2_t src) {
    return vld1_lane_s32(ptr, src, 1);
}

// CIR-LABEL: test_vld1_lane_s32
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!s32i>
// CIR: [[VAL:%.*]] = cir.load align(4) [[PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s32i x 2>

// LLVM: {{.*}}test_vld1_lane_s32(ptr{{.*}}[[PTR:%.*]], <2 x i32>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <2 x i32> [[SRC]], ptr [[SRC_ADDR:%.*]], align 8
// LLVM: [[SRC_VAL:%.*]] = load <2 x i32>, ptr [[SRC_ADDR]], align 8
// LLVM:  store <2 x i32> [[SRC_VAL]], ptr [[S1:%.*]], align 8
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <2 x i32>, ptr [[S1]], align 8
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <2 x i32> [[INTRN_VEC]] to <8 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <8 x i8> [[INTRN_VEC_CAST0]] to <2 x i32>
// LLVM: [[INTRN_VAL:%.*]] = load i32, ptr [[PTR_VAL]], align 4
// LLVM: {{.*}} = insertelement <2 x i32> [[INTRN_VEC_CAST1]], i32 [[INTRN_VAL]], i32 1
// LLVM: ret <2 x i32> {{.*}}

uint32x2_t test_vld1_lane_u32(uint32_t const * ptr, uint32x2_t src) {
    return vld1_lane_u32(ptr, src, 1);
}

// CIR-LABEL: test_vld1_lane_u32
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!u32i>
// CIR: [[VAL:%.*]] = cir.load align(4) [[PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u32i x 2>

// LLVM: {{.*}}test_vld1_lane_u32(ptr{{.*}}[[PTR:%.*]], <2 x i32>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <2 x i32> [[SRC]], ptr [[SRC_ADDR:%.*]], align 8
// LLVM: [[SRC_VAL:%.*]] = load <2 x i32>, ptr [[SRC_ADDR]], align 8
// LLVM:  store <2 x i32> [[SRC_VAL]], ptr [[S1:%.*]], align 8
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <2 x i32>, ptr [[S1]], align 8
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <2 x i32> [[INTRN_VEC]] to <8 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <8 x i8> [[INTRN_VEC_CAST0]] to <2 x i32>
// LLVM: [[INTRN_VAL:%.*]] = load i32, ptr [[PTR_VAL]], align 4
// LLVM: {{.*}} = insertelement <2 x i32> [[INTRN_VEC_CAST1]], i32 [[INTRN_VAL]], i32 1
// LLVM: ret <2 x i32> {{.*}}


int32x4_t test_vld1q_lane_s32(int32_t const * ptr, int32x4_t src) {
    return vld1q_lane_s32(ptr, src, 3);
}

// CIR-LABEL: test_vld1q_lane_s32
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!s32i>
// CIR: [[VAL:%.*]] = cir.load align(4) [[PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s32i x 4>

// LLVM: {{.*}}test_vld1q_lane_s32(ptr{{.*}}[[PTR:%.*]], <4 x i32>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <4 x i32> [[SRC]], ptr [[SRC_ADDR:%.*]], align 16
// LLVM: [[SRC_VAL:%.*]] = load <4 x i32>, ptr [[SRC_ADDR]], align 16
// LLVM:  store <4 x i32> [[SRC_VAL]], ptr [[S1:%.*]], align 16
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <4 x i32>, ptr [[S1]], align 16
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <4 x i32> [[INTRN_VEC]] to <16 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <16 x i8> [[INTRN_VEC_CAST0]] to <4 x i32>
// LLVM: [[INTRN_VAL:%.*]] = load i32, ptr [[PTR_VAL]], align 4
// LLVM: {{.*}} = insertelement <4 x i32> [[INTRN_VEC_CAST1]], i32 [[INTRN_VAL]], i32 3
// LLVM: ret <4 x i32> {{.*}}


uint32x4_t test_vld1q_lane_u32(uint32_t const * ptr, uint32x4_t src) {
    return vld1q_lane_u32(ptr, src, 3);
}

// CIR-LABEL: test_vld1q_lane_u32
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!u32i>
// CIR: [[VAL:%.*]] = cir.load align(4) [[PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u32i x 4>

// LLVM: {{.*}}test_vld1q_lane_u32(ptr{{.*}}[[PTR:%.*]], <4 x i32>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <4 x i32> [[SRC]], ptr [[SRC_ADDR:%.*]], align 16
// LLVM: [[SRC_VAL:%.*]] = load <4 x i32>, ptr [[SRC_ADDR]], align 16
// LLVM:  store <4 x i32> [[SRC_VAL]], ptr [[S1:%.*]], align 16
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <4 x i32>, ptr [[S1]], align 16
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <4 x i32> [[INTRN_VEC]] to <16 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <16 x i8> [[INTRN_VEC_CAST0]] to <4 x i32>
// LLVM: [[INTRN_VAL:%.*]] = load i32, ptr [[PTR_VAL]], align 4
// LLVM: {{.*}} = insertelement <4 x i32> [[INTRN_VEC_CAST1]], i32 [[INTRN_VAL]], i32 3
// LLVM: ret <4 x i32> {{.*}}

int64x1_t test_vld1_lane_s64(int64_t const * ptr, int64x1_t src) {
    return vld1_lane_s64(ptr, src, 0);
}

// CIR-LABEL: test_vld1_lane_s64
// CIR: [[IDX:%.*]] = cir.const #cir.int<0> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!s64i>
// CIR: [[VAL:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!s64i>, !s64i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s64i x 1>

// LLVM: {{.*}}test_vld1_lane_s64(ptr{{.*}}[[PTR:%.*]], <1 x i64>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <1 x i64> [[SRC]], ptr [[SRC_ADDR:%.*]], align 8
// LLVM: [[SRC_VAL:%.*]] = load <1 x i64>, ptr [[SRC_ADDR]], align 8
// LLVM:  store <1 x i64> [[SRC_VAL]], ptr [[S1:%.*]], align 8
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <1 x i64>, ptr [[S1]], align 8
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <1 x i64> [[INTRN_VEC]] to <8 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <8 x i8> [[INTRN_VEC_CAST0]] to <1 x i64>
// LLVM: [[INTRN_VAL:%.*]] = load i64, ptr [[PTR_VAL]], align 8
// LLVM: {{.*}} = insertelement <1 x i64> [[INTRN_VEC_CAST1]], i64 [[INTRN_VAL]], i32 0
// LLVM: ret <1 x i64> {{.*}}

uint64x1_t test_vld1_lane_u64(uint64_t const * ptr, uint64x1_t src) {
    return vld1_lane_u64(ptr, src, 0);
}

// CIR-LABEL: test_vld1_lane_u64
// CIR: [[IDX:%.*]] = cir.const #cir.int<0> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!u64i>
// CIR: [[VAL:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!u64i>, !u64i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u64i x 1>

// LLVM: {{.*}}test_vld1_lane_u64(ptr{{.*}}[[PTR:%.*]], <1 x i64>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <1 x i64> [[SRC]], ptr [[SRC_ADDR:%.*]], align 8
// LLVM: [[SRC_VAL:%.*]] = load <1 x i64>, ptr [[SRC_ADDR]], align 8
// LLVM:  store <1 x i64> [[SRC_VAL]], ptr [[S1:%.*]], align 8
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <1 x i64>, ptr [[S1]], align 8
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <1 x i64> [[INTRN_VEC]] to <8 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <8 x i8> [[INTRN_VEC_CAST0]] to <1 x i64>
// LLVM: [[INTRN_VAL:%.*]] = load i64, ptr [[PTR_VAL]], align 8
// LLVM: {{.*}} = insertelement <1 x i64> [[INTRN_VEC_CAST1]], i64 [[INTRN_VAL]], i32 0
// LLVM: ret <1 x i64> {{.*}}

int64x2_t test_vld1q_lane_s64(int64_t const * ptr, int64x2_t src) {
    return vld1q_lane_s64(ptr, src, 1);
}

// CIR-LABEL: test_vld1q_lane_s64
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!s64i>
// CIR: [[VAL:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!s64i>, !s64i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s64i x 2>

// LLVM: {{.*}}test_vld1q_lane_s64(ptr{{.*}}[[PTR:%.*]], <2 x i64>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <2 x i64> [[SRC]], ptr [[SRC_ADDR:%.*]], align 16
// LLVM: [[SRC_VAL:%.*]] = load <2 x i64>, ptr [[SRC_ADDR]], align 16
// LLVM:  store <2 x i64> [[SRC_VAL]], ptr [[S1:%.*]], align 16
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <2 x i64>, ptr [[S1]], align 16
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <2 x i64> [[INTRN_VEC]] to <16 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <16 x i8> [[INTRN_VEC_CAST0]] to <2 x i64>
// LLVM: [[INTRN_VAL:%.*]] = load i64, ptr [[PTR_VAL]], align 8
// LLVM: {{.*}} = insertelement <2 x i64> [[INTRN_VEC_CAST1]], i64 [[INTRN_VAL]], i32 1
// LLVM: ret <2 x i64> {{.*}}

uint64x2_t test_vld1q_lane_u64(uint64_t const * ptr, uint64x2_t src) {
    return vld1q_lane_u64(ptr, src, 1);
}

// CIR-LABEL: test_vld1q_lane_u64
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!u64i>
// CIR: [[VAL:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!u64i>, !u64i
// CIR: {{%.*}} = cir.vec.insert [[VAL]], {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u64i x 2>

// LLVM: {{.*}}test_vld1q_lane_u64(ptr{{.*}}[[PTR:%.*]], <2 x i64>{{.*}}[[SRC:%.*]])
// LLVM: store ptr [[PTR]], ptr [[PTR_ADDR:%.*]], align 8
// LLVM: store <2 x i64> [[SRC]], ptr [[SRC_ADDR:%.*]], align 16
// LLVM: [[SRC_VAL:%.*]] = load <2 x i64>, ptr [[SRC_ADDR]], align 16
// LLVM:  store <2 x i64> [[SRC_VAL]], ptr [[S1:%.*]], align 16
// LLVM: [[PTR_VAL:%.*]] = load ptr, ptr [[PTR_ADDR]], align 8
// LLVM: [[INTRN_VEC:%.*]] = load <2 x i64>, ptr [[S1]], align 16
// LLVM: [[INTRN_VEC_CAST0:%.*]] = bitcast <2 x i64> [[INTRN_VEC]] to <16 x i8>
// LLVM: [[INTRN_VEC_CAST1:%.*]] = bitcast <16 x i8> [[INTRN_VEC_CAST0]] to <2 x i64>
// LLVM: [[INTRN_VAL:%.*]] = load i64, ptr [[PTR_VAL]], align 8
// LLVM: {{.*}} = insertelement <2 x i64> [[INTRN_VEC_CAST1]], i64 [[INTRN_VAL]], i32 1
// LLVM: ret <2 x i64> {{.*}}
