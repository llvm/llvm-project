// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu10.11-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu10.12-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.11-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.12-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.11-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu10.12-amd-amdhsa -x hip -std=c++11 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#define __device__ __attribute__((device))

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z15test_permlane16Pjjjjj
// CIR: cir.call_llvm_intrinsic "amdgcn.permlane16" {{.*}} : (!u32i, !u32i, !u32i, !u32i, !cir.bool, !cir.bool) -> !u32i
// LLVM: define{{.*}} void @_Z15test_permlane16Pjjjjj
// LLVM: call i32 @llvm.amdgcn.permlane16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 false)
__device__ void test_permlane16(unsigned int* out, unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
  *out = __builtin_amdgcn_permlane16(a, b, c, d, 0, 0);
}

// CIR-LABEL: @_Z16test_permlanex16Pjjjjj
// CIR: cir.call_llvm_intrinsic "amdgcn.permlanex16" {{.*}} : (!u32i, !u32i, !u32i, !u32i, !cir.bool, !cir.bool) -> !u32i
// LLVM: define{{.*}} void @_Z16test_permlanex16Pjjjjj
// LLVM: call i32 @llvm.amdgcn.permlanex16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 false)
__device__ void test_permlanex16(unsigned int* out, unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 0, 0);
}

// CIR-LABEL: @_Z16test_mov_dpp_intPii
// CIR: %[[OLD:.*]] = cir.const #cir.poison : !u32i
// CIR: cir.call_llvm_intrinsic "amdgcn.update.dpp" %[[OLD]], {{.*}} : (!u32i, !u32i, !u32i, !u32i, !u32i, !cir.int<u, 1>) -> !u32i
// LLVM: define{{.*}} void @_Z16test_mov_dpp_intPii
// LLVM: call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %{{.*}}, i32 257, i32 15, i32 15, i1 false)
__device__ void test_mov_dpp_int(int* out, int src) {
  *out = __builtin_amdgcn_mov_dpp(src, 0x101, 0xf, 0xf, 0);
}

// CIR-LABEL: @_Z18test_mov_dpp_shortsPs
// CIR: cir.cast bitcast {{.*}} : !s16i -> !u16i
// CIR: cir.call_llvm_intrinsic "amdgcn.update.dpp" {{.*}} : (!u32i, !u32i, !u32i, !u32i, !u32i, !cir.int<u, 1>) -> !u32i
// LLVM: define{{.*}} void @_Z18test_mov_dpp_shortsPs
// LLVM: call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %{{.*}}, i32 257, i32 15, i32 15, i1 false)
__device__ void test_mov_dpp_short(short x, short *p) {
  *p = __builtin_amdgcn_mov_dpp(x, 0x101, 0xf, 0xf, 0);
}

// CIR-LABEL: @_Z18test_mov_dpp_floatfPf
// CIR: cir.cast bitcast {{.*}} : !cir.float -> !u32i
// CIR: cir.call_llvm_intrinsic "amdgcn.update.dpp" {{.*}} : (!u32i, !u32i, !u32i, !u32i, !u32i, !cir.int<u, 1>) -> !u32i
// LLVM: define{{.*}} void @_Z18test_mov_dpp_floatfPf
// LLVM: call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %{{.*}}, i32 257, i32 15, i32 15, i1 false)
__device__ void test_mov_dpp_float(float x, float *p) {
  *p = __builtin_amdgcn_mov_dpp(x, 0x101, 0xf, 0xf, 0);
}

// CIR-LABEL: @_Z19test_update_dpp_intPiii
// CIR: cir.call_llvm_intrinsic "amdgcn.update.dpp" {{.*}} : (!u32i, !u32i, !u32i, !u32i, !u32i, !cir.int<u, 1>) -> !u32i
// LLVM: define{{.*}} void @_Z19test_update_dpp_intPiii
// LLVM: call i32 @llvm.amdgcn.update.dpp.i32(i32 %{{.*}}, i32 %{{.*}}, i32 0, i32 0, i32 0, i1 false)
__device__ void test_update_dpp_int(int* out, int arg1, int arg2) {
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, 0, false);
}

// CIR-LABEL: @_Z18test_mov_dpp8_uintPjj
// CIR: cir.call_llvm_intrinsic "amdgcn.mov.dpp8" {{.*}} : (!u32i, !u32i) -> !u32i
// LLVM: define{{.*}} void @_Z18test_mov_dpp8_uintPjj
// LLVM: call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %{{.*}}, i32 1)
__device__ void test_mov_dpp8_uint(unsigned int* out, unsigned int a) {
  *out = __builtin_amdgcn_mov_dpp8(a, 1);
}

// CIR-LABEL: @_Z19test_mov_dpp8_shortsPs
// CIR: cir.cast bitcast {{.*}} : !s16i -> !u16i
// CIR: cir.call_llvm_intrinsic "amdgcn.mov.dpp8" {{.*}} : (!u32i, !u32i) -> !u32i
// LLVM: define{{.*}} void @_Z19test_mov_dpp8_shortsPs
// LLVM: call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %{{.*}}, i32 1)
__device__ void test_mov_dpp8_short(short x, short *p) {
  *p = __builtin_amdgcn_mov_dpp8(x, 1);
}
