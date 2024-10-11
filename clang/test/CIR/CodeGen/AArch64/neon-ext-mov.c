// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone  -fno-clangir-call-conv-lowering \
// RUN:  -flax-vector-conversions=none -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone -fno-clangir-call-conv-lowering \
// RUN:  -flax-vector-conversions=none -emit-llvm -o - %s \
// RUN: | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// This test file contains test cases for the intrinsics that move data between
// registers and vectors, such as mov, get, set, and ext. We dedicate this file 
// to them becuase they are many. The file neon.c covers some such intrinsics 
// that are not in this file.  

#include <arm_neon.h>

int8x8_t test_vext_s8(int8x8_t a, int8x8_t b) {
  return vext_s8(a, b, 2);

  // CIR-LABEL: vext_s8
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!s8i x 8>) 
  // CIR-SAME: [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, 
  // CIR-SAME:  #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, 
  // CIR-SAME:  #cir.int<8> : !s32i, #cir.int<9> : !s32i] : !cir.vector<!s8i x 8>

  // LLVM: {{.*}}test_vext_s8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <8 x i8> [[A]], <8 x i8> [[B]], 
  // LLVM-SAME: <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  // LLVM: ret <8 x i8> [[RES]]
}

int8x16_t test_vextq_s8(int8x16_t a, int8x16_t b) {
  return vextq_s8(a, b, 2);

  // CIR-LABEL: vextq_s8
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!s8i x 16>) 
  // CIR-SAME: [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, 
  // CIR-SAME:  #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, 
  // CIR-SAME:  #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, 
  // CIR-SAME:  #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i,
  // CIR-SAME:  #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i,  
  // CIR-SAME:  #cir.int<17> : !s32i] : !cir.vector<!s8i x 16>

  // LLVM: {{.*}}test_vextq_s8(<16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[B]], 
  // LLVM-SAME: <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9
  // LLVM-SAME: i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // LLVM: ret <16 x i8> [[RES]]
}

int16x4_t test_vext_s16(int16x4_t a, int16x4_t b) {
  return vext_s16(a, b, 3);

  // CIR-LABEL: vext_s16
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!s16i x 4>) 
  // CIR-SAME: [#cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i,
  // CIR-SAME:  #cir.int<6> : !s32i] : !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vext_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], 
  // LLVM-SAME: <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  // LLVM: ret <4 x i16> [[RES]]
}

int16x8_t test_vextq_s16(int16x8_t a, int16x8_t b) {
  return vextq_s16(a, b, 3);

  // CIR-LABEL: vextq_s16
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!s16i x 8>) 
  // CIR-SAME: [#cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, 
  // CIR-SAME:  #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, 
  // CIR-SAME:  #cir.int<9> : !s32i, #cir.int<10> : !s32i] : !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vextq_s16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], 
  // LLVM-SAME: <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  // LLVM: ret <8 x i16> [[RES]]
}


uint16x4_t test_vext_u16(uint16x4_t a, uint16x4_t b) {
  return vext_u16(a, b, 3);

  // CIR-LABEL: vext_u16
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!u16i x 4>) 
  // CIR-SAME: [#cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i,
  // CIR-SAME:  #cir.int<6> : !s32i] : !cir.vector<!u16i x 4>

  // LLVM: {{.*}}test_vext_u16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <4 x i16> [[A]], <4 x i16> [[B]], 
  // LLVM-SAME: <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  // LLVM: ret <4 x i16> [[RES]]
}

uint16x8_t test_vextq_u16(uint16x8_t a, uint16x8_t b) {
  return vextq_u16(a, b, 3);

  // CIR-LABEL: vextq_u16
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!u16i x 8>) 
  // CIR-SAME: [#cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, 
  // CIR-SAME:  #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, 
  // CIR-SAME:  #cir.int<9> : !s32i, #cir.int<10> : !s32i] : !cir.vector<!u16i x 8>

  // LLVM: {{.*}}test_vextq_u16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[B]], 
  // LLVM-SAME: <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  // LLVM: ret <8 x i16> [[RES]]
}

int32x2_t test_vext_s32(int32x2_t a, int32x2_t b) {
  return vext_s32(a, b, 1);

  // CIR-LABEL: vext_s32
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!s32i x 2>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vext_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[B]], 
  // LLVM-SAME: <2 x i32> <i32 1, i32 2>
  // LLVM: ret <2 x i32> [[RES]]
}

int32x4_t test_vextq_s32(int32x4_t a, int32x4_t b) {
  return vextq_s32(a, b, 1);

  // CIR-LABEL: vextq_s32
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!s32i x 4>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<2> : !s32i,
  // CIR-SAME:  #cir.int<3> : !s32i, #cir.int<4> : !s32i] : !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vextq_s32(<4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[B]], 
  // LLVM-SAME: <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  // LLVM: ret <4 x i32> [[RES]]
}

int64x1_t test_vext_s64(int64x1_t a, int64x1_t b) {
  return vext_s64(a, b, 0);
  
  // CIR-LABEL: vext_s64
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!s64i x 1>) 
  // CIR-SAME: [#cir.int<0> : !s32i] : !cir.vector<!s64i x 1>

  // LLVM: {{.*}}test_vext_s64(<1 x i64>{{.*}}[[A:%.*]], <1 x i64>{{.*}}[[B:%.*]])
  // LLVM: ret <1 x i64> [[A]]
}

int64x2_t test_vextq_s64(int64x2_t a, int64x2_t b) {
  return vextq_s64(a, b, 1);

  // CIR-LABEL: vextq_s64
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!s64i x 2>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vextq_s64(<2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <2 x i64> [[A]], <2 x i64> [[B]], 
  // LLVM-SAME: <2 x i32> <i32 1, i32 2>
  // LLVM: ret <2 x i64> [[RES]]
}

float32x2_t test_vext_f32(float32x2_t a, float32x2_t b) {
  return vext_f32(a, b, 1);

  // CIR-LABEL: vext_f32
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!cir.float x 2>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!cir.float x 2>

  // LLVM: {{.*}}test_vext_f32(<2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <2 x float> [[A]], <2 x float> [[B]], 
  // LLVM-SAME: <2 x i32> <i32 1, i32 2>
  // LLVM: ret <2 x float> [[RES]]
}

float32x4_t test_vextq_f32(float32x4_t a, float32x4_t b) {
  return vextq_f32(a, b, 1);

  // CIR-LABEL: vextq_f32
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!cir.float x 4>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, 
  // CIR-SAME:  #cir.int<4> : !s32i] : !cir.vector<!cir.float x 4>

  // LLVM: {{.*}}test_vextq_f32(<4 x float>{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <4 x float> [[A]], <4 x float> [[B]], 
  // LLVM-SAME: <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  // LLVM: ret <4 x float> [[RES]]
}


float64x1_t test_vext_f64(float64x1_t a, float64x1_t b) {
  return vext_f64(a, b, 0);
  
  // CIR-LABEL: vext_f64
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!cir.double x 1>) 
  // CIR-SAME: [#cir.int<0> : !s32i] : !cir.vector<!cir.double x 1>

  // LLVM: {{.*}}test_vext_f64(<1 x double>{{.*}}[[A:%.*]], <1 x double>{{.*}}[[B:%.*]])
  // LLVM: ret <1 x double> [[A]]
}

float64x2_t test_vextq_f64(float64x2_t a, float64x2_t b) {
  return vextq_f64(a, b, 1);

  // CIR-LABEL: vextq_f64
  // CIR: {{%.*}}= cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<!cir.double x 2>) 
  // CIR-SAME: [#cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!cir.double x 2>

  // LLVM: {{.*}}test_vextq_f64(<2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = shufflevector <2 x double> [[A]], <2 x double> [[B]], 
  // LLVM-SAME: <2 x i32> <i32 1, i32 2>
  // LLVM: ret <2 x double> [[RES]]
}
