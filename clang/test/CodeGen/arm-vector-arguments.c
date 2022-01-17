// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple thumbv7-apple-darwin9 \
// RUN:   -target-abi apcs-gnu \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -mfloat-abi soft \
// RUN:   -target-feature +soft-float-abi \
// RUN:   -ffreestanding \
// RUN:   -emit-llvm -w -o - %s | FileCheck %s

#include <arm_neon.h>

// CHECK: define{{.*}} void @f0(%struct.int8x16x2_t* noalias sret(%struct.int8x16x2_t) align 16 %agg.result, <16 x i8> noundef %{{.*}}, <16 x i8> noundef %{{.*}})
int8x16x2_t f0(int8x16_t a0, int8x16_t a1) {
  return vzipq_s8(a0, a1);
}

// Test direct vector passing.

typedef float T_float32x2 __attribute__ ((__vector_size__ (8)));
typedef float T_float32x4 __attribute__ ((__vector_size__ (16)));
typedef float T_float32x8 __attribute__ ((__vector_size__ (32)));
typedef float T_float32x16 __attribute__ ((__vector_size__ (64)));

// CHECK: define{{.*}} <2 x float> @f1_0(<2 x float> noundef %{{.*}})
T_float32x2 f1_0(T_float32x2 a0) { return a0; }
// CHECK: define{{.*}} <4 x float> @f1_1(<4 x float> noundef %{{.*}})
T_float32x4 f1_1(T_float32x4 a0) { return a0; }
// CHECK: define{{.*}} void @f1_2(<8 x float>* noalias sret(<8 x float>) align 32 %{{.*}}, <8 x float> noundef %{{.*}})
T_float32x8 f1_2(T_float32x8 a0) { return a0; }
// CHECK: define{{.*}} void @f1_3(<16 x float>* noalias sret(<16 x float>) align 64 %{{.*}}, <16 x float> noundef %{{.*}})
T_float32x16 f1_3(T_float32x16 a0) { return a0; }
