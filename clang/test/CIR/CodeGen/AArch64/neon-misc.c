
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -fno-clangir-call-conv-lowering -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// This test file contains test cases for the intrinsics that are not covered
// by the other neon test files.

#include <arm_neon.h>

uint8x8_t test_vset_lane_u8(uint8_t a, uint8x8_t b) {
  return vset_lane_u8(a, b, 7);
}

// CIR-LABEL: test_vset_lane_u8
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i loc(#loc7)
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vset_lane_u8(i8{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <8 x i8> [[B]], i8 [[A]], i32 7
// LLVM: ret <8 x i8> [[INTRN_RES]]

uint16x4_t test_vset_lane_u16(uint16_t a, uint16x4_t b) {
  return vset_lane_u16(a, b, 3);
}

// CIR-LABEL: test_vset_lane_u16
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s16i x 4>

// LLVM: {{.*}}test_vset_lane_u16(i16{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <4 x i16> [[B]], i16 [[A]], i32 3
// LLVM: ret <4 x i16> [[INTRN_RES]]

uint32x2_t test_vset_lane_u32(uint32_t a, uint32x2_t b) {
  return vset_lane_u32(a, b, 1);
}

// CIR-LABEL: test_vset_lane_u32
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s32i x 2>

// LLVM: {{.*}}test_vset_lane_u32(i32{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <2 x i32> [[B]], i32 [[A]], i32 1
// LLVM: ret <2 x i32> [[INTRN_RES]]

uint64x1_t test_vset_lane_u64(uint64_t a, uint64x1_t b) {
  return vset_lane_u64(a, b, 0);
}

// CIR-LABEL: test_vset_lane_u64
// CIR: [[IDX:%.*]] = cir.const #cir.int<0> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s64i x 1>

// LLVM: {{.*}}test_vset_lane_u64(i64{{.*}}[[A:%.*]], <1 x i64>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <1 x i64> [[B]], i64 [[A]], i32 0
// LLVM: ret <1 x i64> [[INTRN_RES]]

float32x2_t test_vset_lane_f32(float32_t a, float32x2_t b) {
  return vset_lane_f32(a, b, 1);
}

// CIR-LABEL: test_vset_lane_f32
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.float x 2>

// LLVM: {{.*}}test_vset_lane_f32(float{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <2 x float> [[B]], float [[A]], i32 1
// LLVM: ret <2 x float> [[INTRN_RES]]

uint8x16_t test_vsetq_lane_u8(uint8_t a, uint8x16_t b) {
  return vsetq_lane_u8(a, b, 15);
}

// CIR-LABEL: test_vsetq_lane_u8
// CIR: [[IDX:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vsetq_lane_u8(i8{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <16 x i8> [[B]], i8 [[A]], i32 15
// LLVM: ret <16 x i8> [[INTRN_RES]]

uint16x8_t test_vsetq_lane_u16(uint16_t a, uint16x8_t b) {
  return vsetq_lane_u16(a, b, 7);
}

// CIR-LABEL: test_vsetq_lane_u16
// CIR: [[IDX:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s16i x 8>

// LLVM: {{.*}}test_vsetq_lane_u16(i16{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <8 x i16> [[B]], i16 [[A]], i32 7
// LLVM: ret <8 x i16> [[INTRN_RES]]

uint32x4_t test_vsetq_lane_u32(uint32_t a, uint32x4_t b) {
  return vsetq_lane_u32(a, b, 3);
}

// CIR-LABEL: test_vsetq_lane_u32
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s32i x 4>

// LLVM: {{.*}}test_vsetq_lane_u32(i32{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <4 x i32> [[B]], i32 [[A]], i32 3
// LLVM: ret <4 x i32> [[INTRN_RES]]

int64x2_t test_vsetq_lane_s64(int64_t a, int64x2_t b) {
  return vsetq_lane_s64(a, b, 1);
}

// CIR-LABEL: test_vsetq_lane_s64
// CIR: [[IDX:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!s64i x 2>

// LLVM: {{.*}}test_vsetq_lane_s64(i64{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <2 x i64> [[B]], i64 [[A]], i32 1
// LLVM: ret <2 x i64> [[INTRN_RES]]

float32x4_t test_vsetq_lane_f32(float32_t a, float32x4_t b) {
  return vsetq_lane_f32(a, b, 3);
}

// CIR-LABEL: test_vsetq_lane_f32
// CIR: [[IDX:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.float x 4>

// LLVM: {{.*}}test_vsetq_lane_f32(float{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
// LLVM: [[INTRN_RES:%.*]] = insertelement <4 x float> [[B]], float [[A]], i32 3
// LLVM: ret <4 x float> [[INTRN_RES]]

uint8_t test_vget_lane_u8(uint8x8_t a) {
  return vget_lane_u8(a, 7);
}

// CIR-LABEL: test_vget_lane_u8
// CIR: [[IDX:%.*]]  = cir.const #cir.int<7> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u8i x 8>

// LLVM: {{.*}}test_vget_lane_u8(<8 x i8>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <8 x i8> [[ARG]], i32 7
// LLVM: ret i8 [[RES]]

uint8_t test_vgetq_lane_u8(uint8x16_t a) {
  return vgetq_lane_u8(a, 15);
}

// CIR-LABEL: test_vgetq_lane_u8
// CIR: [[IDX:%.*]]  = cir.const #cir.int<15> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u8i x 16>

// LLVM: {{.*}}test_vgetq_lane_u8(<16 x i8>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <16 x i8> [[ARG]], i32 15
// LLVM: ret i8 [[RES]]

uint16_t test_vget_lane_u16(uint16x4_t a) {
  return vget_lane_u16(a, 3);
}

// CIR-LABEL: test_vget_lane_u16
// CIR: [[IDX:%.*]]  = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u16i x 4>

// LLVM: {{.*}}test_vget_lane_u16(<4 x i16>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <4 x i16> [[ARG]], i32 3
// LLVM: ret i16 [[RES]]

uint16_t test_vgetq_lane_u16(uint16x8_t a) {
  return vgetq_lane_u16(a, 7);
}

// CIR-LABEL: test_vgetq_lane_u16
// CIR: [[IDX:%.*]]  = cir.const #cir.int<7> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u16i x 8>

// LLVM: {{.*}}test_vgetq_lane_u16(<8 x i16>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <8 x i16> [[ARG]], i32 7
// LLVM: ret i16 [[RES]]

uint32_t test_vget_lane_u32(uint32x2_t a) {
  return vget_lane_u32(a, 1);
}

// CIR-LABEL: test_vget_lane_u32
// CIR: [[IDX:%.*]]  = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u32i x 2>

// LLVM: {{.*}}test_vget_lane_u32(<2 x i32>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <2 x i32> [[ARG]], i32 1
// LLVM: ret i32 [[RES]]

uint32_t test_vgetq_lane_u32(uint32x4_t a) {
  return vgetq_lane_u32(a, 3);
}

// CIR-LABEL: test_vgetq_lane_u32
// CIR: [[IDX:%.*]]  = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u32i x 4>

// LLVM: {{.*}}test_vgetq_lane_u32(<4 x i32>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <4 x i32> [[ARG]], i32 3
// LLVM: ret i32 [[RES]]

uint64_t test_vget_lane_u64(uint64x1_t a) {
  return vget_lane_u64(a, 0);
}

// CIR-LABEL: test_vget_lane_u64
// CIR: [[IDX:%.*]]  = cir.const #cir.int<0> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u64i x 1>

// LLVM: {{.*}}test_vget_lane_u64(<1 x i64>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <1 x i64> [[ARG]], i32 0
// LLVM: ret i64 [[RES]]

uint64_t test_vgetq_lane_u64(uint64x2_t a) {
  return vgetq_lane_u64(a, 1);
}

// CIR-LABEL: test_vgetq_lane_u64
// CIR: [[IDX:%.*]]  = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u64i x 2>

// LLVM: {{.*}}test_vgetq_lane_u64(<2 x i64>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <2 x i64> [[ARG]], i32 1
// LLVM: ret i64 [[RES]]

float32_t test_vget_lane_f32(float32x2_t a) {
  return vget_lane_f32(a, 1);
}

// CIR-LABEL: test_vget_lane_f32
// CIR: [[IDX:%.*]]  = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.float x 2>

// LLVM: {{.*}}test_vget_lane_f32(<2 x float>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <2 x float> [[ARG]], i32 1
// LLVM: ret float [[RES]]

float64_t test_vget_lane_f64(float64x1_t a) {
  return vget_lane_f64(a, 0);
}

// CIR-LABEL: test_vget_lane_f64
// CIR: [[IDX:%.*]]  = cir.const #cir.int<0> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.double x 1>

// LLVM: {{.*}}test_vget_lane_f64(<1 x double>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <1 x double> [[ARG]], i32 0
// LLVM: ret double [[RES]]

float32_t test_vgetq_lane_f32(float32x4_t a) {
  return vgetq_lane_f32(a, 3);
}

// CIR-LABEL: test_vgetq_lane_f32
// CIR: [[IDX:%.*]]  = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.float x 4>

// LLVM: {{.*}}test_vgetq_lane_f32(<4 x float>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <4 x float> [[ARG]], i32 3
// LLVM: ret float [[RES]]

float64_t test_vgetq_lane_f64(float64x2_t a) {
  return vgetq_lane_f64(a, 1);
}

// CIR-LABEL: test_vgetq_lane_f64
// CIR: [[IDX:%.*]]  = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.double x 2>

// LLVM: {{.*}}test_vgetq_lane_f64(<2 x double>{{.*}}[[ARG:%.*]])
// LLVM: [[RES:%.*]] = extractelement <2 x double> [[ARG]], i32 1
// LLVM: ret double [[RES]]

uint8x8x2_t test_vtrn_u8(uint8x8_t a, uint8x8_t b) {
  return vtrn_u8(a, b);

  // CIR-LABEL: vtrn_u8
  // CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u8i x 8>>
  // CIR: [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
  // CIR: [[ADDR:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!u8i x 8>>, [[ZERO]] : !s32i), !cir.ptr<!cir.vector<!u8i x 8>>
  // CIR: [[RES:%.*]] = cir.vec.shuffle([[INP1:%.*]], [[INP2:%.*]] : !cir.vector<!u8i x 8>) 
  // CIR-SAME: [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, 
  // CIR-SAME: #cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<6> : !s32i,
  // CIR-SAME: #cir.int<14> : !s32i] : !cir.vector<!u8i x 8>
  // CIR:  cir.store [[RES]], [[ADDR]] : !cir.vector<!u8i x 8>, !cir.ptr<!cir.vector<!u8i x 8>>
  // CIR: [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
  // CIR: [[ADDR1:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!u8i x 8>>, [[ONE]] : !s32i), !cir.ptr<!cir.vector<!u8i x 8>>
  // CIR: [[RES1:%.*]] = cir.vec.shuffle([[INP1]], [[INP2]] : !cir.vector<!u8i x 8>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i, 
  // CIR-SAME: #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : 
  // CIR-SAME: !cir.vector<!u8i x 8>
  // CIR:  cir.store [[RES1]], [[ADDR1]] : !cir.vector<!u8i x 8>, !cir.ptr<!cir.vector<!u8i x 8>>

  // LLVM: {{.*}}test_vtrn_u8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]]) 
  // LLVM: [[VTRN:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], 
  // LLVM-SAME: <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  // LLVM: store <8 x i8> [[VTRN]], ptr [[RES:%.*]], align 8
  // LLVM: [[RES1:%.*]] = getelementptr {{.*}}<8 x i8>, ptr [[RES]], i64 1
  // LLVM: [[VTRN1:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  // LLVM: store <8 x i8> [[VTRN1]], ptr [[RES1]], align 8
  // LLVM: ret %struct.uint8x8x2_t {{.*}}
}

uint16x4x2_t test_vtrn_u16(uint16x4_t a, uint16x4_t b) {
  return vtrn_u16(a, b);

  // CIR-LABEL: vtrn_u16
  // CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u16i x 4>>
  // CIR: [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
  // CIR: [[ADDR:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!u16i x 4>>, [[ZERO]] : !s32i), !cir.ptr<!cir.vector<!u16i x 4>>
  // CIR: [[RES:%.*]] = cir.vec.shuffle([[INP1:%.*]], [[INP2:%.*]] : !cir.vector<!u16i x 4>) 
  // CIR-SAME: [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<2> : !s32i, #cir.int<6> : !s32i] : !cir.vector<!u16i x 4>
  // CIR:  cir.store [[RES]], [[ADDR]] : !cir.vector<!u16i x 4>, !cir.ptr<!cir.vector<!u16i x 4>>
  // CIR: [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
  // CIR: [[ADDR1:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!u16i x 4>>, [[ONE]] : !s32i), !cir.ptr<!cir.vector<!u16i x 4>>
  // CIR: [[RES1:%.*]] = cir.vec.shuffle([[INP1]], [[INP2]] : !cir.vector<!u16i x 4>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<5> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i] :
  // CIR-SAME: !cir.vector<!u16i x 4>
  // CIR:  cir.store [[RES1]], [[ADDR1]] : !cir.vector<!u16i x 4>, !cir.ptr<!cir.vector<!u16i x 4>>

  // LLVM: {{.*}}test_vtrn_u16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]]) 
  // LLVM: [[VTRN:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], 
  // LLVM-SAME: <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // LLVM: store <4 x i16> [[VTRN]], ptr [[RES:%.*]], align 8
  // LLVM: [[RES1:%.*]] = getelementptr {{.*}}<4 x i16>, ptr [[RES]], i64 1
  // LLVM: [[VTRN1:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], 
  // LLVM-SAME: <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  // LLVM: store <4 x i16> [[VTRN1]], ptr [[RES1]], align 8
  // LLVM: ret %struct.uint16x4x2_t {{.*}}
}

int32x2x2_t test_vtrn_s32(int32x2_t a, int32x2_t b) {
  return vtrn_s32(a, b);

  // CIR-LABEL: vtrn_s32
  // CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s32i x 2>>
  // CIR: [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
  // CIR: [[ADDR:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!s32i x 2>>, [[ZERO]] : !s32i), !cir.ptr<!cir.vector<!s32i x 2>>
  // CIR: [[RES:%.*]] = cir.vec.shuffle([[INP1:%.*]], [[INP2:%.*]] : !cir.vector<!s32i x 2>) 
  // CIR-SAME: [#cir.int<0> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!s32i x 2>
  // CIR:  cir.store [[RES]], [[ADDR]] : !cir.vector<!s32i x 2>, !cir.ptr<!cir.vector<!s32i x 2>>
  // CIR: [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
  // CIR: [[ADDR1:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!s32i x 2>>, [[ONE]] : !s32i), !cir.ptr<!cir.vector<!s32i x 2>>
  // CIR: [[RES1:%.*]] = cir.vec.shuffle([[INP1]], [[INP2]] : !cir.vector<!s32i x 2>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<3> : !s32i] :
  // CIR-SAME: !cir.vector<!s32i x 2>
  // CIR:  cir.store [[RES1]], [[ADDR1]] : !cir.vector<!s32i x 2>, !cir.ptr<!cir.vector<!s32i x 2>>

  // LLVM: {{.*}}test_vtrn_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])   
  // LLVM: [[VTRN:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], 
  // LLVM-SAME: <2 x i32> <i32 0, i32 2>
  // LLVM: store <2 x i32> [[VTRN]], ptr [[RES:%.*]], align 8
  // LLVM: [[RES1:%.*]] = getelementptr {{.*}}<2 x i32>, ptr [[RES]], i64 1
  // LLVM: [[VTRN1:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], 
  // LLVM-SAME: <2 x i32> <i32 1, i32 3>
  // LLVM: store <2 x i32> [[VTRN1]], ptr [[RES1]], align 8
  // LLVM: ret %struct.int32x2x2_t {{.*}}
}

uint8x16x2_t test_vtrnq_u8(uint8x16_t a, uint8x16_t b) {
  return vtrnq_u8(a, b);

  // CIR-LABEL: vtrnq_u8
  // CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u8i x 16>>
  // CIR: [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
  // CIR: [[ADDR:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!u8i x 16>>, [[ZERO]] : !s32i), !cir.ptr<!cir.vector<!u8i x 16>>
  // CIR: [[RES:%.*]] = cir.vec.shuffle([[INP1:%.*]], [[INP2:%.*]] : !cir.vector<!u8i x 16>) 
  // CIR-SAME: [#cir.int<0> : !s32i, #cir.int<16> : !s32i, #cir.int<2> : !s32i, #cir.int<18> : !s32i, 
  // CIR-SAME: #cir.int<4> : !s32i, #cir.int<20> : !s32i, #cir.int<6> : !s32i, #cir.int<22> : !s32i,
  // CIR-SAME: #cir.int<8> : !s32i, #cir.int<24> : !s32i, #cir.int<10> : !s32i, #cir.int<26> : !s32i,
  // CIR-SAME: #cir.int<12> : !s32i, #cir.int<28> : !s32i, #cir.int<14> : !s32i, #cir.int<30> : !s32i] : !cir.vector<!u8i x 16>
  // CIR:  cir.store [[RES]], [[ADDR]] : !cir.vector<!u8i x 16>, !cir.ptr<!cir.vector<!u8i x 16>>
  // CIR: [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
  // CIR: [[ADDR1:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!u8i x 16>>, [[ONE]] : !s32i), !cir.ptr<!cir.vector<!u8i x 16>>
  // CIR: [[RES1:%.*]] = cir.vec.shuffle([[INP1]], [[INP2]] : !cir.vector<!u8i x 16>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<17> : !s32i, #cir.int<3> : !s32i, #cir.int<19> : !s32i, 
  // CIR-SAME: #cir.int<5> : !s32i, #cir.int<21> : !s32i, #cir.int<7> : !s32i, #cir.int<23> : !s32i,
  // CIR-SAME: #cir.int<9> : !s32i, #cir.int<25> : !s32i, #cir.int<11> : !s32i, #cir.int<27> : !s32i,
  // CIR-SAME: #cir.int<13> : !s32i, #cir.int<29> : !s32i, #cir.int<15> : !s32i, #cir.int<31> : !s32i] : !cir.vector<!u8i x 16>
  // CIR:  cir.store [[RES1]], [[ADDR1]] : !cir.vector<!u8i x 16>, !cir.ptr<!cir.vector<!u8i x 16>>

  // LLVM: {{.*}}test_vtrnq_u8(<16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[VTRN:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], 
  // LLVM-SAME: <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, 
  // LLVM-SAME: i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  // LLVM: store <16 x i8> [[VTRN]], ptr [[RES:%.*]], align 16
  // LLVM: [[RES1:%.*]] = getelementptr {{.*}}<16 x i8>, ptr [[RES]], i64 1
  // LLVM: [[VTRN1:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], 
  // LLVM-SAME: <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23,
  // LLVM-SAME: i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31> 
  // LLVM: store <16 x i8> [[VTRN1]], ptr [[RES1]], align 16
  // LLVM: ret %struct.uint8x16x2_t {{.*}}
}

int16x8x2_t test_vtrnq_s16(int16x8_t a, int16x8_t b) {
  return vtrnq_s16(a, b);

  // CIR-LABEL: vtrnq_s16
  // CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s16i x 8>>
  // CIR: [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
  // CIR: [[ADDR:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!s16i x 8>>, [[ZERO]] : !s32i), !cir.ptr<!cir.vector<!s16i x 8>>
  // CIR: [[RES:%.*]] = cir.vec.shuffle([[INP1:%.*]], [[INP2:%.*]] : !cir.vector<!s16i x 8>) 
  // CIR-SAME: [#cir.int<0> : !s32i, #cir.int<8> : !s32i, #cir.int<2> : !s32i, #cir.int<10> : !s32i, 
  // CIR-SAME: #cir.int<4> : !s32i, #cir.int<12> : !s32i, #cir.int<6> : !s32i,
  // CIR-SAME: #cir.int<14> : !s32i] : !cir.vector<!s16i x 8>
  // CIR:  cir.store [[RES]], [[ADDR]] : !cir.vector<!s16i x 8>, !cir.ptr<!cir.vector<!s16i x 8>>
  // CIR: [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
  // CIR: [[ADDR1:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!s16i x 8>>, [[ONE]] : !s32i), !cir.ptr<!cir.vector<!s16i x 8>>
  // CIR: [[RES1:%.*]] = cir.vec.shuffle([[INP1]], [[INP2]] : !cir.vector<!s16i x 8>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<3> : !s32i, #cir.int<11> : !s32i, 
  // CIR-SAME: #cir.int<5> : !s32i, #cir.int<13> : !s32i, #cir.int<7> : !s32i, #cir.int<15> : !s32i] : 
  // CIR-SAME: !cir.vector<!s16i x 8>
  // CIR:  cir.store [[RES1]], [[ADDR1]] : !cir.vector<!s16i x 8>, !cir.ptr<!cir.vector<!s16i x 8>>

  // LLVM: {{.*}}test_vtrnq_s16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[VTRN:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], 
  // LLVM-SAME: <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  // LLVM: store <8 x i16> [[VTRN]], ptr [[RES:%.*]], align 16
  // LLVM: [[RES1:%.*]] = getelementptr {{.*}}<8 x i16>, ptr [[RES]], i64 1
  // LLVM: [[VTRN1:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  // LLVM: store <8 x i16> [[VTRN1]], ptr [[RES1]], align 16
  // LLVM: ret %struct.int16x8x2_t {{.*}}
}

uint32x4x2_t test_vtrnq_u32(uint32x4_t a, uint32x4_t b) {
  return vtrnq_u32(a, b);

  // CIR-LABEL: vtrnq_u32
  // CIR: [[PTR:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u32i x 4>>
  // CIR: [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
  // CIR: [[ADDR:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!u32i x 4>>, [[ZERO]] : !s32i), !cir.ptr<!cir.vector<!u32i x 4>>
  // CIR: [[RES:%.*]] = cir.vec.shuffle([[INP1:%.*]], [[INP2:%.*]] : !cir.vector<!u32i x 4>) 
  // CIR-SAME: [#cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<2> : !s32i, #cir.int<6> : !s32i] :
  // CIR-SAME: !cir.vector<!u32i x 4>
  // CIR:  cir.store [[RES]], [[ADDR]] : !cir.vector<!u32i x 4>, !cir.ptr<!cir.vector<!u32i x 4>>
  // CIR: [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
  // CIR: [[ADDR1:%.*]] = cir.ptr_stride([[PTR]] : !cir.ptr<!cir.vector<!u32i x 4>>, [[ONE]] : !s32i), !cir.ptr<!cir.vector<!u32i x 4>>
  // CIR: [[RES1:%.*]] = cir.vec.shuffle([[INP1]], [[INP2]] : !cir.vector<!u32i x 4>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<5> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i] :
  // CIR-SAME: !cir.vector<!u32i x 4>
  // CIR:  cir.store [[RES1]], [[ADDR1]] : !cir.vector<!u32i x 4>, !cir.ptr<!cir.vector<!u32i x 4>>
  // LLVM: ret %struct.uint32x4x2_t {{.*}}
}

uint8x8_t test_vqmovun_s16(int16x8_t a) {
  return vqmovun_s16(a);

  // CIR-LABEL: vqmovun_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqxtun" {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>) -> !cir.vector<!u8i x 8>
  
  // LLVM: {{.*}}test_vqmovun_s16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM: [[VQMOVUN_V1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtun.v8i8(<8 x i16> [[A]])
  // LLVM: ret <8 x i8> [[VQMOVUN_V1_I]]
}

uint16x4_t test_vqmovun_s32(int32x4_t a) {
  return vqmovun_s32(a);

  // CIR-LABEL: vqmovun_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqxtun" {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}test_vqmovun_s32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM: [[VQMOVUN_V1_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32> [[A]])
  // LLVM: [[VQMOVUN_V2_I:%.*]] = bitcast <4 x i16> [[VQMOVUN_V1_I]] to <8 x i8>
  // LLVM: ret <4 x i16> [[VQMOVUN_V1_I]]
}

uint32x2_t test_vqmovun_s64(int64x2_t a) {
  return vqmovun_s64(a);

  // CIR-LABEL: vqmovun_s64
  // CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqxtun" {{%.*}} :
  // CIR-SAME: (!cir.vector<!s64i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}test_vqmovun_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM: [[VQMOVUN_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqxtun.v2i32(<2 x i64> [[A]])
  // LLVM: [[VQMOVUN_V2_I:%.*]] = bitcast <2 x i32> [[VQMOVUN_V1_I]] to <8 x i8>
  // LLVM: ret <2 x i32> [[VQMOVUN_V1_I]]
}
