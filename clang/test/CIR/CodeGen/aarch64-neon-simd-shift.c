// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-llvm -target-feature +neon %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target
#include <arm_neon.h>

uint8x8_t test_vqrshrun_n_s16(int16x8_t a) {
  return vqrshrun_n_s16(a, 3);
}

// CIR-LABEL: test_vqrshrun_n_s16
// CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s16i x 8> 
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
// CIR-SAME: (!cir.vector<!s16i x 8>, !s32i) -> !cir.vector<!u8i x 8>

// LLVM: {{.*}}test_vqrshrun_n_s16(<8 x i16>{{.*}} [[A:%.*]])
// LLVM: store <8 x i16> [[A]], ptr [[A_ADDR:%.*]], align 16
// LLVM: [[A_VAL:%.*]] = load <8 x i16>, ptr [[A_ADDR]], align 16
// LLVM: store <8 x i16> [[A_VAL]], ptr [[S0:%.*]], align 16
// LLVM: [[S0_VAL:%.*]] = load <8 x i16>, ptr [[S0]], align 16
// LLVM: [[S0_VAL_CAST:%.*]] = bitcast <8 x i16> [[S0_VAL]] to <16 x i8>
// LLVM: [[INTRN_ARG:%.*]] = bitcast <16 x i8> [[S0_VAL_CAST]] to <8 x i16>
// LLVM: {{%.*}} = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> [[INTRN_ARG]], i32 3)
// LLVM: ret <8 x i8> {{%.*}}

uint16x4_t test_vqrshrun_n_s32(int32x4_t a) {
  return vqrshrun_n_s32(a, 7);
}

// CIR-LABEL: test_vqrshrun_n_s32
// CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4> 
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
// CIR-SAME: (!cir.vector<!s32i x 4>, !s32i) -> !cir.vector<!u16i x 4>

// LLVM: {{.*}}test_vqrshrun_n_s32(<4 x i32>{{.*}} [[A:%.*]])
// LLVM: store <4 x i32> [[A]], ptr [[A_ADDR:%.*]], align 16
// LLVM: [[A_VAL:%.*]] = load <4 x i32>, ptr [[A_ADDR]], align 16
// LLVM: store <4 x i32> [[A_VAL]], ptr [[S0:%.*]], align 16
// LLVM: [[S0_VAL:%.*]] = load <4 x i32>, ptr [[S0]], align 16
// LLVM: [[S0_VAL_CAST:%.*]] = bitcast <4 x i32> [[S0_VAL]] to <16 x i8>
// LLVM: [[INTRN_ARG:%.*]] = bitcast <16 x i8> [[S0_VAL_CAST]] to <4 x i32>
// LLVM: {{%.*}} = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> [[INTRN_ARG]], i32 7)
// LLVM: ret <4 x i16> {{%.*}}

uint32x2_t test_vqrshrun_n_s64(int64x2_t a) {
  return vqrshrun_n_s64(a, 15);
}

// CIR-LABEL: test_vqrshrun_n_s64
// CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2> 
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
// CIR-SAME: (!cir.vector<!s64i x 2>, !s32i) -> !cir.vector<!u32i x 2>

// LLVM: {{.*}}test_vqrshrun_n_s64(<2 x i64>{{.*}} [[A:%.*]])
// LLVM: store <2 x i64> [[A]], ptr [[A_ADDR:%.*]], align 16
// LLVM: [[A_VAL:%.*]] = load <2 x i64>, ptr [[A_ADDR]], align 16
// LLVM: store <2 x i64> [[A_VAL]], ptr [[S0:%.*]], align 16
// LLVM: [[S0_VAL:%.*]] = load <2 x i64>, ptr [[S0]], align 16
// LLVM: [[S0_VAL_CAST:%.*]] = bitcast <2 x i64> [[S0_VAL]] to <16 x i8>
// LLVM: [[INTRN_ARG:%.*]] = bitcast <16 x i8> [[S0_VAL_CAST]] to <2 x i64>
// LLVM: {{%.*}} = call <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64> [[INTRN_ARG]], i32 15)
// LLVM: ret <2 x i32> {{%.*}}
