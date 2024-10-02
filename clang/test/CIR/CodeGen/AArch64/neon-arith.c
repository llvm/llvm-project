// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-llvm -target-feature +neon %s -o %t.ll
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

// LLVM: define dso_local float @test_vrndns_f32(float [[ARG:%.*]])
// LLVM: store float [[ARG]], ptr [[ARG_SAVE:%.*]], align 4
// LLVM: [[P0:%.*]] = load float, ptr [[ARG_SAVE]], align 4,
// LLVM: store float [[P0]], ptr [[P0_SAVE:%.*]], align 4,
// LLVM: [[INTRIN_ARG:%.*]] = load float, ptr [[P0_SAVE]], align 4,
// LLVM: [[INTRIN_RES:%.*]] = call float @llvm.roundeven.f32(float [[INTRIN_ARG]])
// LLVM: store float [[INTRIN_RES]], ptr [[RES_SAVE0:%.*]], align 4, 
// LLVM: [[RES_COPY0:%.*]] = load float, ptr [[RES_SAVE0]], align 4,
// LLVM: store float [[RES_COPY0]], ptr [[RES_SAVE1:%.*]], align 4,
// LLVM: [[RES_COPY1:%.*]] = load float, ptr [[RES_SAVE1]], align 4,
// LLVM: store float [[RES_COPY1]], ptr [[RET_P:%.*]], align 4,
// LLVM: [[RET_VAL:%.*]] = load float, ptr [[RET_P]], align 4,
// LLVM: ret float [[RET_VAL]]

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

// LLVM: define dso_local <2 x float> @test_vrnda_f32(<2 x float> [[ARG:%.*]])
// LLVM: store <2 x float> [[ARG]], ptr [[ARG_SAVE:%.*]], align 8
// LLVM: [[P0:%.*]] = load <2 x float>, ptr [[ARG_SAVE]], align 8,
// LLVM: store <2 x float> [[P0]], ptr [[P0_SAVE:%.*]], align 8,
// LLVM: [[INTRIN_ARG:%.*]] = load <2 x float>, ptr [[P0_SAVE]], align 8,
// LLVM: [[INTRIN_RES:%.*]] = call <2 x float> @llvm.round.v2f32(<2 x float> [[INTRIN_ARG]])
// LLVM: store <2 x float> [[INTRIN_RES]], ptr [[RES_SAVE0:%.*]], align 8,
// LLVM: [[RES_COPY0:%.*]] = load <2 x float>, ptr [[RES_SAVE0]], align 8,
// LLVM: store <2 x float> [[RES_COPY0]], ptr [[RES_SAVE1:%.*]], align 8,
// LLVM: [[RES_COPY1:%.*]] = load <2 x float>, ptr [[RES_SAVE1]], align 8,
// LLVM: store <2 x float> [[RES_COPY1]], ptr [[RET_P:%.*]], align 8,
// LLVM: [[RET_VAL:%.*]] = load <2 x float>, ptr [[RET_P]], align 8,
// LLVM: ret <2 x float> [[RET_VAL]]

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

// LLVM: define dso_local <4 x float> @test_vrndaq_f32(<4 x float> [[ARG:%.*]])
// LLVM: store <4 x float> [[ARG]], ptr [[ARG_SAVE:%.*]], align 16
// LLVM: [[P0:%.*]] = load <4 x float>, ptr [[ARG_SAVE]], align 16,
// LLVM: store <4 x float> [[P0]], ptr [[P0_SAVE:%.*]], align 16,
// LLVM: [[INTRIN_ARG:%.*]] = load <4 x float>, ptr [[P0_SAVE]], align 16,
// LLVM: [[INTRIN_RES:%.*]] = call <4 x float> @llvm.round.v4f32(<4 x float> [[INTRIN_ARG]])
// LLVM: store <4 x float> [[INTRIN_RES]], ptr [[RES_SAVE0:%.*]], align 16,
// LLVM: [[RES_COPY0:%.*]] = load <4 x float>, ptr [[RES_SAVE0]], align 16,
// LLVM: store <4 x float> [[RES_COPY0]], ptr [[RES_SAVE1:%.*]], align 16,
// LLVM: [[RES_COPY1:%.*]] = load <4 x float>, ptr [[RES_SAVE1]], align 16,
// LLVM: store <4 x float> [[RES_COPY1]], ptr [[RET_P:%.*]], align 16,
// LLVM: [[RET_VAL:%.*]] = load <4 x float>, ptr [[RET_P]], align 16,
// LLVM: ret <4 x float> [[RET_VAL]]
