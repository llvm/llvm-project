#include "../CodeGenCUDA/Inputs/cuda.h"

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:   -target-cpu gfx900 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:   -target-cpu gfx900 -fcuda-is-device -emit-llvm \
// RUN:   -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:   -target-cpu gfx900 -fcuda-is-device -emit-llvm \
// RUN:   -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

using v3i16 = short __attribute__((ext_vector_type(3)));
using v3i32 = int __attribute__((ext_vector_type(3)));
using v3f32 = float __attribute__((ext_vector_type(3)));

__device__ void copy_i32(v3i32 *dst, const v3i32 *src) {
  *dst = *src;
}

// CIR-LABEL: @_Z8copy_i32PDv3_iPKS_
// CIR: %[[I32:.*]] = cir.load{{.*}} : !cir.ptr<!cir.vector<3 x !s32i>>, !cir.vector<3 x !s32i>
// CIR: cir.store{{.*}} %[[I32]], {{.*}} : !cir.vector<3 x !s32i>, !cir.ptr<!cir.vector<3 x !s32i>>

// LLVM-LABEL: define{{.*}} void @_Z8copy_i32PDv3_iPKS_
// LLVM: %[[I32:.*]] = load <3 x i32>
// LLVM: store <3 x i32> %[[I32]]

__device__ void copy_f32(v3f32 *dst, const v3f32 *src) {
  *dst = *src;
}

// CIR-LABEL: @_Z8copy_f32PDv3_fPKS_
// CIR: %[[F32:.*]] = cir.load{{.*}} : !cir.ptr<!cir.vector<3 x !cir.float>>, !cir.vector<3 x !cir.float>
// CIR: cir.store{{.*}} %[[F32]], {{.*}} : !cir.vector<3 x !cir.float>, !cir.ptr<!cir.vector<3 x !cir.float>>

// LLVM-LABEL: define{{.*}} void @_Z8copy_f32PDv3_fPKS_
// LLVM: %[[F32:.*]] = load <3 x float>
// LLVM: store <3 x float> %[[F32]]

__device__ void copy_i16(v3i16 *dst, const v3i16 *src) {
  *dst = *src;
}

// CIR-LABEL: @_Z8copy_i16PDv3_sPKS_
// CIR: %[[I16_LOAD:.*]] = cir.load{{.*}} : !cir.ptr<!cir.vector<4 x !s16i>>, !cir.vector<4 x !s16i>
// CIR: %[[I16_NARROW:.*]] = cir.vec.shuffle(%[[I16_LOAD]], {{.*}} : !cir.vector<4 x !s16i>) {{.*}} : !cir.vector<3 x !s16i>
// CIR: %[[I16_UNDEF:.*]] = cir.const #cir.undef : !cir.vector<3 x !s16i>
// CIR: %[[I16_WIDE:.*]] = cir.vec.shuffle(%[[I16_NARROW]], %[[I16_UNDEF]] : !cir.vector<3 x !s16i>) {{.*}} : !cir.vector<4 x !s16i>
// CIR: cir.store{{.*}} %[[I16_WIDE]], {{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>

// LLVM-LABEL: define{{.*}} void @_Z8copy_i16PDv3_sPKS_
// LLVM: %[[I16_LOAD:.*]] = load <4 x i16>
// LLVM: %[[I16_NARROW:.*]] = shufflevector <4 x i16> %[[I16_LOAD]], <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// LLVM: %[[I16_WIDE:.*]] = shufflevector <3 x i16> %[[I16_NARROW]], <3 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// LLVM: store <4 x i16> %[[I16_WIDE]]
