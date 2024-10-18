// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -fno-clangir-call-conv-lowering \
// RUN:   -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -fno-clangir-call-conv-lowering \
// RUN:  -emit-llvm -o - %s \
// RUN: | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target
#include <arm_neon.h>

// This test file contains tests for aarch64 NEON arithmetic intrinsics 
// that are not vector type related.

float32_t test_vrndns_f32(float32_t a) {
  return vrndns_f32(a);
}
// CIR: cir.func internal private  @vrndns_f32(%arg0: !cir.float {{.*}}) -> !cir.float
// CIR: cir.store %arg0, [[ARG_SAVE:%.*]] : !cir.float, !cir.ptr<!cir.float> 
// CIR: [[INTRIN_ARG:%.*]] = cir.load [[ARG_SAVE]] : !cir.ptr<!cir.float>, !cir.float 
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.roundeven.f32" [[INTRIN_ARG]] : (!cir.float)
// CIR: cir.return {{%.*}} : !cir.float

// CIR-LABEL: test_vrndns_f32
// CIR: cir.store %arg0, [[ARG_SAVE0:%.*]] : !cir.float, !cir.ptr<!cir.float> 
// CIR: [[FUNC_ARG:%.*]] = cir.load [[ARG_SAVE]] : !cir.ptr<!cir.float>, !cir.float 
// CIR: [[FUNC_RES:%.*]] = cir.call @vrndns_f32([[FUNC_ARG]]) : (!cir.float) -> !cir.float
// CIR: cir.store [[FUNC_RES]], [[RET_P:%.*]] : !cir.float, !cir.ptr<!cir.float>
// CIR: [[RET_VAL:%.*]] = cir.load [[RET_P]] : !cir.ptr<!cir.float>, !cir.float
// CIR: cir.return [[RET_VAL]] : !cir.float loc

// LLVM: {{.*}}test_vrndns_f32(float{{.*}}[[ARG:%.*]])
// LLVM: [[INTRIN_RES:%.*]] = call float @llvm.roundeven.f32(float [[ARG]])
// LLVM: ret float [[INTRIN_RES]]

float32x2_t test_vrnda_f32(float32x2_t a) {
  return vrnda_f32(a);
}

// CIR: cir.func internal private  @vrnda_f32(%arg0: !cir.vector<!cir.float x 2>
// CIR: cir.store %arg0, [[ARG_SAVE:%.*]] : !cir.vector<!cir.float x 2>, !cir.ptr<!cir.vector<!cir.float x 2>>
// CIR: [[INTRIN_ARG:%.*]] = cir.load [[ARG_SAVE]] : !cir.ptr<!cir.vector<!cir.float x 2>>, !cir.vector<!cir.float x 2>
// CIR: [[INTRIN_ARG_CAST:%.*]] = cir.cast(bitcast, [[INTRIN_ARG]] : !cir.vector<!cir.float x 2>), !cir.vector<!s8i x 8>
// CIR: [[INTRIN_ARG_BACK:%.*]] = cir.cast(bitcast, [[INTRIN_ARG_CAST]] : !cir.vector<!s8i x 8>), !cir.vector<!cir.float x 2>
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.round" [[INTRIN_ARG_BACK]] : (!cir.vector<!cir.float x 2>) -> !cir.vector<!cir.float x 2>
// CIR: cir.return {{%.*}} : !cir.vector<!cir.float x 2>

// CIR-LABEL: test_vrnda_f32
// CIR: cir.store %arg0, [[ARG_SAVE0:%.*]] :  !cir.vector<!cir.float x 2>, !cir.ptr<!cir.vector<!cir.float x 2>> 
// CIR: [[FUNC_ARG:%.*]] = cir.load [[ARG_SAVE]] : !cir.ptr<!cir.vector<!cir.float x 2>>, !cir.vector<!cir.float x 2> 
// CIR: [[FUNC_RES:%.*]] = cir.call @vrnda_f32([[FUNC_ARG]]) : (!cir.vector<!cir.float x 2>) -> !cir.vector<!cir.float x 2>
// CIR: cir.store [[FUNC_RES]], [[RET_P:%.*]] : !cir.vector<!cir.float x 2>, !cir.ptr<!cir.vector<!cir.float x 2>>
// CIR: [[RET_VAL:%.*]] = cir.load [[RET_P]] : !cir.ptr<!cir.vector<!cir.float x 2>>, !cir.vector<!cir.float x 2>
// CIR: cir.return [[RET_VAL]] : !cir.vector<!cir.float x 2>

// LLVM: {{.*}}test_vrnda_f32(<2 x float>{{.*}}[[ARG:%.*]])
// LLVM: [[INTRIN_RES:%.*]] = call <2 x float> @llvm.round.v2f32(<2 x float> [[ARG]])
// LLVM: ret <2 x float> [[INTRIN_RES]]

float32x4_t test_vrndaq_f32(float32x4_t a) {
  return vrndaq_f32(a);
}

// CIR: cir.func internal private  @vrndaq_f32(%arg0: !cir.vector<!cir.float x 4>
// CIR: cir.store %arg0, [[ARG_SAVE:%.*]] : !cir.vector<!cir.float x 4>, !cir.ptr<!cir.vector<!cir.float x 4>>
// CIR: [[INTRIN_ARG:%.*]] = cir.load [[ARG_SAVE]] : !cir.ptr<!cir.vector<!cir.float x 4>>, !cir.vector<!cir.float x 4>
// CIR: [[INTRIN_ARG_CAST:%.*]] = cir.cast(bitcast, [[INTRIN_ARG]] : !cir.vector<!cir.float x 4>), !cir.vector<!s8i x 16>
// CIR: [[INTRIN_ARG_BACK:%.*]] = cir.cast(bitcast, [[INTRIN_ARG_CAST]] : !cir.vector<!s8i x 16>), !cir.vector<!cir.float x 4>
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.round" [[INTRIN_ARG_BACK]] : (!cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>
// CIR: cir.return {{%.*}} : !cir.vector<!cir.float x 4>

// LLVM: {{.*}}test_vrndaq_f32(<4 x float>{{.*}}[[ARG:%.*]])
// LLVM: [[INTRIN_RES:%.*]] = call <4 x float> @llvm.round.v4f32(<4 x float> [[ARG]])
// LLVM: ret <4 x float> [[INTRIN_RES]]

int8x8_t test_vpadd_s8(int8x8_t a, int8x8_t b) {
  return vpadd_s8(a, b);
}

// CIR-LABEL: vpadd_s8
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vpadd_s8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8> [[A]], <8 x i8> [[B]])
// LLVM: ret <8 x i8> [[RES]]


int8x16_t test_vpaddq_s8(int8x16_t a, int8x16_t b) {
  return vpaddq_s8(a, b);
}

// CIR-LABEL: vpaddq_s8
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vpaddq_s8(<16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> [[A]], <16 x i8> [[B]])
// LLVM: ret <16 x i8> [[RES]]

uint8x8_t test_vpadd_u8(uint8x8_t a, uint8x8_t b) {
  return vpadd_u8(a, b);
}

// CIR-LABEL: vpadd_u8
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

// LLVM: {{.*}}test_vpadd_u8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8> [[A]], <8 x i8> [[B]])
// LLVM: ret <8 x i8> [[RES]]

int16x4_t test_vpadd_s16(int16x4_t a, int16x4_t b) {
  return vpadd_s16(a, b);
}

// CIR-LABEL: vpadd_s16
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>
// CIR: {{%.*}} = cir.cast(bitcast, [[RES]] : !cir.vector<!s16i x 4>), !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vpadd_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16> [[A]], <4 x i16> [[B]])
// LLVM: ret <4 x i16> [[RES]]

int16x8_t test_vpaddq_s16(int16x8_t a, int16x8_t b) {
  return vpaddq_s16(a, b);
}

// CIR-LABEL: vpaddq_s16
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>
// CIR: {{%.*}} = cir.cast(bitcast, [[RES]] : !cir.vector<!s16i x 8>), !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vpaddq_s16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16> [[A]], <8 x i16> [[B]])
// LLVM: ret <8 x i16> [[RES]]

uint16x4_t test_vpadd_u16(uint16x4_t a, uint16x4_t b) {
  return vpadd_u16(a, b);
}

// CIR-LABEL: vpadd_u16
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>
// CIR: {{%.*}} = cir.cast(bitcast, [[RES]] : !cir.vector<!u16i x 4>), !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vpadd_u16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16> [[A]], <4 x i16> [[B]])
// LLVM: ret <4 x i16> [[RES]]

int32x2_t test_vpadd_s32(int32x2_t a, int32x2_t b) {
  return vpadd_s32(a, b);
}

// CIR-LABEL: vpadd_s32
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>
// CIR: {{%.*}} = cir.cast(bitcast, [[RES]] : !cir.vector<!s32i x 2>), !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vpadd_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32> [[A]], <2 x i32> [[B]])
// LLVM: ret <2 x i32> [[RES]]

int32x4_t test_vpaddq_s32(int32x4_t a, int32x4_t b) {
  return vpaddq_s32(a, b);
}

// CIR-LABEL: vpaddq_s32
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>
// CIR: {{%.*}} = cir.cast(bitcast, [[RES]] : !cir.vector<!s32i x 4>), !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vpaddq_s32(<4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32> [[A]], <4 x i32> [[B]])
// LLVM: ret <4 x i32> [[RES]]

float32x2_t test_vpadd_f32(float32x2_t a, float32x2_t b) {
  return vpadd_f32(a, b);
}

// CIR-LABEL: vpadd_f32
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!cir.float x 2>, !cir.vector<!cir.float x 2>) -> !cir.vector<!cir.float x 2>
// CIR: {{%.*}} = cir.cast(bitcast, [[RES]] : !cir.vector<!cir.float x 2>), !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vpadd_f32(<2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float> [[A]], <2 x float> [[B]])
// LLVM: ret <2 x float> [[RES]]

float32x4_t test_vpaddq_f32(float32x4_t a, float32x4_t b) {
  return vpaddq_f32(a, b);
}

// CIR-LABEL: vpaddq_f32
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!cir.float x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>
// CIR: {{%.*}} = cir.cast(bitcast, [[RES]] : !cir.vector<!cir.float x 4>), !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vpaddq_f32(<4 x float>{{.*}}[[A:%.*]], <4 x float>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x float> @llvm.aarch64.neon.faddp.v4f32(<4 x float> [[A]], <4 x float> [[B]])
// LLVM: ret <4 x float> [[RES]]

float64x2_t test_vpaddq_f64(float64x2_t a, float64x2_t b) {
  return vpaddq_f64(a, b);
}

// CIR-LABEL: vpaddq_f64
// CIR: [[RES:%.*]] = cir.llvm.intrinsic "llvm.aarch64.neon.addp" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>
// CIR: {{%.*}} = cir.cast(bitcast, [[RES]] : !cir.vector<!cir.double x 2>), !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vpaddq_f64(<2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x double> @llvm.aarch64.neon.faddp.v2f64(<2 x double> [[A]], <2 x double> [[B]])
// LLVM: ret <2 x double> [[RES]]

int16x4_t test_vqdmulh_lane_s16(int16x4_t a, int16x4_t v) {
  return vqdmulh_lane_s16(a, v, 3);
}

// CIR-LABEL: vqdmulh_lane_s16
// CIR: [[LANE:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqdmulh.lane" {{%.*}}, {{%.*}}, [[LANE]] : 
// CIR: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>, !s32i) -> !cir.vector<!s16i x 4>

// LLVM: {{.*}}test_vqdmulh_lane_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[V:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.lane.v4i16.v4i16
// LLVM-SAME: (<4 x i16> [[A]], <4 x i16> [[V]], i32 3)
// LLVM:  ret <4 x i16> [[RES]]


int32x2_t test_vqdmulh_lane_s32(int32x2_t a, int32x2_t v) {
  return vqdmulh_lane_s32(a, v, 1);
}

// CIR-LABEL: vqdmulh_lane_s32
// CIR: [[LANE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqdmulh.lane" {{%.*}}, {{%.*}}, [[LANE]] : 
// CIR: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>, !s32i) -> !cir.vector<!s32i x 2>

// LLVM: {{.*}}test_vqdmulh_lane_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[V:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqdmulh.lane.v2i32.v2i32
// LLVM-SAME: (<2 x i32> [[A]], <2 x i32> [[V]], i32 1)
// LLVM:  ret <2 x i32> [[RES]]

int16x8_t test_vqdmulhq_lane_s16(int16x8_t a, int16x4_t v) {
  return vqdmulhq_lane_s16(a, v, 3);
}

// CIR-LABEL: vqdmulhq_lane_s16
// CIR: [[LANE:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqdmulh.lane" {{%.*}}, {{%.*}}, [[LANE]] : 
// CIR: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 4>, !s32i) -> !cir.vector<!s16i x 8>

// LLVM: {{.*}}test_vqdmulhq_lane_s16(<8 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[V:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqdmulh.lane.v8i16.v4i16
// LLVM-SAME: (<8 x i16> [[A]], <4 x i16> [[V]], i32 3)
// LLVM:  ret <8 x i16> [[RES]]

int32x4_t test_vqdmulhq_lane_s32(int32x4_t a, int32x2_t v) {
  return vqdmulhq_lane_s32(a, v, 1);
}

// CIR-LABEL: vqdmulhq_lane_s32
// CIR: [[LANE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqdmulh.lane" {{%.*}}, {{%.*}}, [[LANE]] : 
// CIR: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 2>, !s32i) -> !cir.vector<!s32i x 4>

// LLVM: {{.*}}test_vqdmulhq_lane_s32(<4 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[V:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmulh.lane.v4i32.v2i32
// LLVM-SAME: (<4 x i32> [[A]], <2 x i32> [[V]], i32 1)
// LLVM:  ret <4 x i32> [[RES]]

int16x4_t test_vqrdmulh_lane_s16(int16x4_t a, int16x4_t v) {
  return vqrdmulh_lane_s16(a, v, 3);
}

// CIR-LABEL: vqrdmulh_lane_s16
// CIR: [[LANE:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrdmulh.lane" {{%.*}}, {{%.*}}, [[LANE]] : 
// CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>, !s32i) -> !cir.vector<!s16i x 4>

// LLVM: {{.*}}test_vqrdmulh_lane_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[V:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.lane.v4i16.v4i16
// LLVM-SAME: (<4 x i16> [[A]], <4 x i16> [[V]], i32 3)
// LLVM:  ret <4 x i16> [[RES]]

int16x8_t test_vqrdmulhq_lane_s16(int16x8_t a, int16x4_t v) {
  return vqrdmulhq_lane_s16(a, v, 3);
}

// CIR-LABEL: vqrdmulhq_lane_s16
// CIR: [[LANE:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrdmulh.lane" {{%.*}}, {{%.*}}, [[LANE]] : 
// CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 4>, !s32i) -> !cir.vector<!s16i x 8>

// LLVM: {{.*}}test_vqrdmulhq_lane_s16(<8 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[V:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.lane.v8i16.v4i16
// LLVM-SAME: (<8 x i16> [[A]], <4 x i16> [[V]], i32 3)
// LLVM:  ret <8 x i16> [[RES]]

int32x2_t test_vqrdmulh_lane_s32(int32x2_t a, int32x2_t v) {
  return vqrdmulh_lane_s32(a, v, 1);
}

// CIR-LABEL: vqrdmulh_lane_s32
// CIR: [[LANE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrdmulh.lane" {{%.*}}, {{%.*}}, [[LANE]] : 
// CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>, !s32i) -> !cir.vector<!s32i x 2>

// LLVM: {{.*}}test_vqrdmulh_lane_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[V:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.lane.v2i32.v2i32
// LLVM-SAME: (<2 x i32> [[A]], <2 x i32> [[V]], i32 1)
// LLVM:  ret <2 x i32> [[RES]]

int32x4_t test_vqrdmulhq_lane_s32(int32x4_t a, int32x2_t v) {
  return vqrdmulhq_lane_s32(a, v, 1);
}

// CIR-LABEL: vqrdmulhq_lane_s32
// CIR: [[LANE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrdmulh.lane" {{%.*}}, {{%.*}}, [[LANE]] :
// CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 2>, !s32i) -> !cir.vector<!s32i x 4>

// LLVM: {{.*}}test_vqrdmulhq_lane_s32(<4 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[V:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.lane.v4i32.v2i32
// LLVM-SAME: (<4 x i32> [[A]], <2 x i32> [[V]], i32 1)
// LLVM:  ret <4 x i32> [[RES]]

int8x16_t test_vqaddq_s8(int8x16_t a, int8x16_t b) {
  return vqaddq_s8(a, b);
}

// CIR-LABEL: vqaddq_s8
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vqaddq_s8(<16 x i8>{{.*}} [[A:%.*]], <16 x i8>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqadd.v16i8(<16 x i8> [[A]], <16 x i8> [[B]])
// LLVM: ret <16 x i8> [[RES]]

uint8x16_t test_vqaddq_u8(uint8x16_t a, uint8x16_t b) {
  return vqaddq_u8(a, b);
}

// CIR-LABEL: vqaddq_u8
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

// LLVM: {{.*}}test_vqaddq_u8(<16 x i8>{{.*}} [[A:%.*]], <16 x i8>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqadd.v16i8(<16 x i8> [[A]], <16 x i8> [[B]])
// LLVM: ret <16 x i8> [[RES]]

int16x8_t test_vqaddq_s16(int16x8_t a, int16x8_t b) {
  return vqaddq_s16(a, b);
}

// CIR-LABEL: vqaddq_s16
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

// LLVM: {{.*}}test_vqaddq_s16(<8 x i16>{{.*}} [[A:%.*]], <8 x i16>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> [[A]], <8 x i16> [[B]])
// LLVM: ret <8 x i16> [[RES]]

uint16x8_t test_vqaddq_u16(uint16x8_t a, uint16x8_t b) {
  return vqaddq_u16(a, b);
}

// CIR-LABEL: vqaddq_u16
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

// LLVM: {{.*}}test_vqaddq_u16(<8 x i16>{{.*}} [[A:%.*]], <8 x i16>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqadd.v8i16(<8 x i16> [[A]], <8 x i16> [[B]])
// LLVM: ret <8 x i16> [[RES]]

int32x4_t test_vqaddq_s32(int32x4_t a, int32x4_t b) {
  return vqaddq_s32(a, b);
}

// CIR-LABEL: vqaddq_s32
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

// LLVM: {{.*}}test_vqaddq_s32(<4 x i32>{{.*}} [[A:%.*]], <4 x i32>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> [[A]], <4 x i32> [[B]])
// LLVM: ret <4 x i32> [[RES]]

int64x2_t test_vqaddq_s64(int64x2_t a, int64x2_t b) {
  return vqaddq_s64(a, b);
}

// CIR-LABEL: vqaddq_s64
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

// LLVM: {{.*}}test_vqaddq_s64(<2 x i64>{{.*}} [[A:%.*]], <2 x i64>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> [[A]], <2 x i64> [[B]])
// LLVM: ret <2 x i64> [[RES]]

uint64x2_t test_vqaddq_u64(uint64x2_t a, uint64x2_t b) {
  return vqaddq_u64(a, b);
}

// CIR-LABEL: vqaddq_u64
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u64i x 2>, !cir.vector<!u64i x 2>) -> !cir.vector<!u64i x 2>

// LLVM: {{.*}}test_vqaddq_u64(<2 x i64>{{.*}} [[A:%.*]], <2 x i64>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqadd.v2i64(<2 x i64> [[A]], <2 x i64> [[B]])
// LLVM: ret <2 x i64> [[RES]]

int8x8_t test_vqsub_s8(int8x8_t a, int8x8_t b) {
  return vqsub_s8(a, b);
}

// CIR-LABEL: vqsub_s8
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vqsub_s8(<8 x i8>{{.*}} [[A:%.*]], <8 x i8>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqsub.v8i8(<8 x i8> [[A]], <8 x i8> [[B]])
// LLVM: ret <8 x i8> [[RES]]

uint8x8_t test_vqsub_u8(uint8x8_t a, uint8x8_t b) {
  return vqsub_u8(a, b);
}

// CIR-LABEL: vqsub_u8
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

// LLVM: {{.*}}test_vqsub_u8(<8 x i8>{{.*}} [[A:%.*]], <8 x i8>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqsub.v8i8(<8 x i8> [[A]], <8 x i8> [[B]])
// LLVM: ret <8 x i8> [[RES]]

int16x4_t test_vqsub_s16(int16x4_t a, int16x4_t b) {
  return vqsub_s16(a, b);
}

// CIR-LABEL: vqsub_s16
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

// LLVM: {{.*}}test_vqsub_s16(<4 x i16>{{.*}} [[A:%.*]], <4 x i16>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> [[A]], <4 x i16> [[B]])
// LLVM: ret <4 x i16> [[RES]]

uint16x4_t test_vqsub_u16(uint16x4_t a, uint16x4_t b) {
  return vqsub_u16(a, b);
}

// CIR-LABEL: vqsub_u16
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

// LLVM: {{.*}}test_vqsub_u16(<4 x i16>{{.*}} [[A:%.*]], <4 x i16>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqsub.v4i16(<4 x i16> [[A]], <4 x i16> [[B]])
// LLVM: ret <4 x i16> [[RES]]

int32x2_t test_vqsub_s32(int32x2_t a, int32x2_t b) {
  return vqsub_s32(a, b);
}

// CIR-LABEL: vqsub_s32
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

// LLVM: {{.*}}test_vqsub_s32(<2 x i32>{{.*}} [[A:%.*]], <2 x i32>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> [[A]], <2 x i32> [[B]])
// LLVM: ret <2 x i32> [[RES]]

uint32x2_t test_vqsub_u32(uint32x2_t a, uint32x2_t b) {
  return vqsub_u32(a, b);
}

// CIR-LABEL: vqsub_u32
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

// LLVM: {{.*}}test_vqsub_u32(<2 x i32>{{.*}} [[A:%.*]], <2 x i32>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqsub.v2i32(<2 x i32> [[A]], <2 x i32> [[B]])
// LLVM: ret <2 x i32> [[RES]]

int64x1_t test_vqsub_s64(int64x1_t a, int64x1_t b) {
  return vqsub_s64(a, b);
}

// CIR-LABEL: vqsub_s64
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s64i x 1>, !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>

// LLVM: {{.*}}test_vqsub_s64(<1 x i64>{{.*}} [[A:%.*]], <1 x i64>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqsub.v1i64(<1 x i64> [[A]], <1 x i64> [[B]])
// LLVM: ret <1 x i64> [[RES]]

uint64x1_t test_vqsub_u64(uint64x1_t a, uint64x1_t b) {
  return vqsub_u64(a, b);
}

// CIR-LABEL: vqsub_u64
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u64i x 1>, !cir.vector<!u64i x 1>) -> !cir.vector<!u64i x 1>

// LLVM: {{.*}}test_vqsub_u64(<1 x i64>{{.*}} [[A:%.*]], <1 x i64>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqsub.v1i64(<1 x i64> [[A]], <1 x i64> [[B]])
// LLVM: ret <1 x i64> [[RES]]

int8x16_t test_vqsubq_s8(int8x16_t a, int8x16_t b) {
  return vqsubq_s8(a, b);
}

// CIR-LABEL: vqsubq_s8
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vqsubq_s8(<16 x i8>{{.*}} [[A:%.*]], <16 x i8>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqsub.v16i8(<16 x i8> [[A]], <16 x i8> [[B]])
// LLVM: ret <16 x i8> [[RES]]

uint8x16_t test_vqsubq_u8(uint8x16_t a, uint8x16_t b) {
  return vqsubq_u8(a, b);
}

// CIR-LABEL: vqsubq_u8
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

// LLVM: {{.*}}test_vqsubq_u8(<16 x i8>{{.*}} [[A:%.*]], <16 x i8>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqsub.v16i8(<16 x i8> [[A]], <16 x i8> [[B]])
// LLVM: ret <16 x i8> [[RES]]

int16x8_t test_vqsubq_s16(int16x8_t a, int16x8_t b) { 
  return vqsubq_s16(a, b);
}

// CIR-LABEL: vqsubq_s16
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

// LLVM: {{.*}}test_vqsubq_s16(<8 x i16>{{.*}} [[A:%.*]], <8 x i16>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> [[A]], <8 x i16> [[B]])
// LLVM: ret <8 x i16> [[RES]]

uint16x8_t test_vqsubq_u16(uint16x8_t a, uint16x8_t b) {
  return vqsubq_u16(a, b);
}

// CIR-LABEL: vqsubq_u16
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

// LLVM: {{.*}}test_vqsubq_u16(<8 x i16>{{.*}} [[A:%.*]], <8 x i16>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqsub.v8i16(<8 x i16> [[A]], <8 x i16> [[B]])
// LLVM: ret <8 x i16> [[RES]]

int32x4_t test_vqsubq_s32(int32x4_t a, int32x4_t b) {
  return vqsubq_s32(a, b);
}

// CIR-LABEL: vqsubq_s32
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

// LLVM: {{.*}}test_vqsubq_s32(<4 x i32>{{.*}} [[A:%.*]], <4 x i32>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> [[A]], <4 x i32> [[B]])
// LLVM: ret <4 x i32> [[RES]]

uint32x4_t test_vqsubq_u32(uint32x4_t a, uint32x4_t b) {
  return vqsubq_u32(a, b);
}

// CIR-LABEL: vqsubq_u32
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u32i x 4>, !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

// LLVM: {{.*}}test_vqsubq_u32(<4 x i32>{{.*}} [[A:%.*]], <4 x i32>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqsub.v4i32(<4 x i32> [[A]], <4 x i32> [[B]])
// LLVM: ret <4 x i32> [[RES]]

int64x2_t test_vqsubq_s64(int64x2_t a, int64x2_t b) {
  return vqsubq_s64(a, b);
}

// CIR-LABEL: vqsubq_s64
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

// LLVM: {{.*}}test_vqsubq_s64(<2 x i64>{{.*}} [[A:%.*]], <2 x i64>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> [[A]], <2 x i64> [[B]])
// LLVM: ret <2 x i64> [[RES]]

uint64x2_t test_vqsubq_u64(uint64x2_t a, uint64x2_t b) {
  return vqsubq_u64(a, b);
}

// CIR-LABEL: vqsubq_u64
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqsub" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u64i x 2>, !cir.vector<!u64i x 2>) -> !cir.vector<!u64i x 2>

// LLVM: {{.*}}test_vqsubq_u64(<2 x i64>{{.*}} [[A:%.*]], <2 x i64>{{.*}} [[B:%.*]])
// LLVM: [[RES:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqsub.v2i64(<2 x i64> [[A]], <2 x i64> [[B]])
// LLVM: ret <2 x i64> [[RES]]
