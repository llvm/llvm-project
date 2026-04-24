#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z28test_wave_reduce_add_u32_i32Pi
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_add_u32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_add_u32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_add_u64_i64Pl
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_add_u64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.add.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_add_u64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_add_u64(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_sub_u32_i32Pi
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.sub" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_sub_u32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.sub.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_sub_u32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_sub_u32(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_sub_u64_i64Pl
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.sub" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_sub_u64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.sub.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_sub_u64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_sub_u64(in, 0);
}

// CIR-LABEL: @_Z29test_wave_reduce_min_i32_signPii
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.min" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z29test_wave_reduce_min_i32_signPii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.min.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_min_i32_sign(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_min_i32(in, 0);
}

// CIR-LABEL: @_Z31test_wave_reduce_min_u32_unsignPjj
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.umin" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z31test_wave_reduce_min_u32_unsignPjj(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_min_u32_unsign(unsigned int* out, unsigned int in) {
  *out = __builtin_amdgcn_wave_reduce_min_u32(in, 0);
}

// CIR-LABEL: @_Z29test_wave_reduce_min_i64_signPll
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.min" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z29test_wave_reduce_min_i64_signPll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.min.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_min_i64_sign(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_min_i64(in, 0);
}

// CIR-LABEL: @_Z31test_wave_reduce_min_u64_unsignPmm
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.umin" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @_Z31test_wave_reduce_min_u64_unsignPmm(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.umin.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_min_u64_unsign(unsigned long* out, unsigned long in) {
  *out = __builtin_amdgcn_wave_reduce_min_u64(in, 0);
}

// CIR-LABEL: @_Z29test_wave_reduce_max_i32_signPii
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.max" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z29test_wave_reduce_max_i32_signPii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.max.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_max_i32_sign(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_max_i32(in, 0);
}

// CIR-LABEL: @_Z31test_wave_reduce_max_u32_unsignPjj
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.umax" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z31test_wave_reduce_max_u32_unsignPjj(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.umax.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_max_u32_unsign(unsigned int* out, unsigned int in) {
  *out = __builtin_amdgcn_wave_reduce_max_u32(in, 0);
}

// CIR-LABEL: @_Z29test_wave_reduce_max_i64_signPll
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.max" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z29test_wave_reduce_max_i64_signPll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.max.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_max_i64_sign(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_max_i64(in, 0);
}

// CIR-LABEL: @_Z31test_wave_reduce_max_u64_unsignPmm
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.umax" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @_Z31test_wave_reduce_max_u64_unsignPmm(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.umax.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_max_u64_unsign(unsigned long* out, unsigned long in) {
  *out = __builtin_amdgcn_wave_reduce_max_u64(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_and_b32_i32Pii
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.and" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_and_b32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.and.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_and_b32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_and_b32(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_and_b64_i64Pll
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.and" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_and_b64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.and.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_and_b64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_and_b64(in, 0);
}

// CIR-LABEL: @_Z27test_wave_reduce_or_b32_i32Pii
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.or" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z27test_wave_reduce_or_b32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.or.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_or_b32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_or_b32(in, 0);
}

// CIR-LABEL: @_Z27test_wave_reduce_or_b64_i64Pll
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.or" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z27test_wave_reduce_or_b64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.or.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_or_b64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_or_b64(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_xor_b32_i32Pii
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.xor" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_xor_b32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.xor.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_xor_b32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_xor_b32(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_xor_b64_i64Pll
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.xor" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_xor_b64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.xor.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_xor_b64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_xor_b64(in, 0);
}

// CIR-LABEL: @_Z38test_wave_reduce_add_u32_iterative_i32Pii
// CIR: cir.const #cir.int<1> : !s32i
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z38test_wave_reduce_add_u32_iterative_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 1)
__device__ void test_wave_reduce_add_u32_iterative_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 1);
}

// CIR-LABEL: @_Z32test_wave_reduce_add_u32_dpp_i32Pii
// CIR: cir.const #cir.int<2> : !s32i
// CIR: cir.call_llvm_intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z32test_wave_reduce_add_u32_dpp_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 2)
__device__ void test_wave_reduce_add_u32_dpp_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 2);
}
