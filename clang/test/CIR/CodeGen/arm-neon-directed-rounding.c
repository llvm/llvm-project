// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-llvm -target-feature +neon %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target
#include <arm_neon.h>

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
