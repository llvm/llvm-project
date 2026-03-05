#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu tahiti -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z28test_wave_reduce_add_u32_i32Pi
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_add_u32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z28test_wave_reduce_add_u32_i32Pii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_add_u32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_add_u64_i64Pl
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_add_u64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.add.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z28test_wave_reduce_add_u64_i64Pll(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.add.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_add_u64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_add_u64(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_sub_u32_i32Pi
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.sub" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_sub_u32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.sub.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z28test_wave_reduce_sub_u32_i32Pii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.sub.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_sub_u32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_sub_u32(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_sub_u64_i64Pl
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.sub" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_sub_u64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.sub.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z28test_wave_reduce_sub_u64_i64Pll(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.sub.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_sub_u64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_sub_u64(in, 0);
}

// CIR-LABEL: @_Z29test_wave_reduce_min_i32_signPii
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.min" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z29test_wave_reduce_min_i32_signPii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.min.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z29test_wave_reduce_min_i32_signPii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.min.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_min_i32_sign(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_min_i32(in, 0);
}

// CIR-LABEL: @_Z31test_wave_reduce_min_u32_unsignPjj
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umin" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z31test_wave_reduce_min_u32_unsignPjj(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z31test_wave_reduce_min_u32_unsignPjj(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_min_u32_unsign(unsigned int* out, unsigned int in) {
  *out = __builtin_amdgcn_wave_reduce_min_u32(in, 0);
}

// CIR-LABEL: @_Z29test_wave_reduce_min_i64_signPll
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.min" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z29test_wave_reduce_min_i64_signPll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.min.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z29test_wave_reduce_min_i64_signPll(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.min.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_min_i64_sign(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_min_i64(in, 0);
}

// CIR-LABEL: @_Z31test_wave_reduce_min_u64_unsignPmm
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umin" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @_Z31test_wave_reduce_min_u64_unsignPmm(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.umin.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z31test_wave_reduce_min_u64_unsignPmm(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.umin.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_min_u64_unsign(unsigned long* out, unsigned long in) {
  *out = __builtin_amdgcn_wave_reduce_min_u64(in, 0);
}

// CIR-LABEL: @_Z29test_wave_reduce_max_i32_signPii
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.max" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z29test_wave_reduce_max_i32_signPii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.max.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z29test_wave_reduce_max_i32_signPii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.max.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_max_i32_sign(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_max_i32(in, 0);
}

// CIR-LABEL: @_Z31test_wave_reduce_max_u32_unsignPjj
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umax" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z31test_wave_reduce_max_u32_unsignPjj(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.umax.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z31test_wave_reduce_max_u32_unsignPjj(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.umax.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_max_u32_unsign(unsigned int* out, unsigned int in) {
  *out = __builtin_amdgcn_wave_reduce_max_u32(in, 0);
}

// CIR-LABEL: @_Z29test_wave_reduce_max_i64_signPll
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.max" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z29test_wave_reduce_max_i64_signPll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.max.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z29test_wave_reduce_max_i64_signPll(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.max.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_max_i64_sign(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_max_i64(in, 0);
}

// CIR-LABEL: @_Z31test_wave_reduce_max_u64_unsignPmm
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umax" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @_Z31test_wave_reduce_max_u64_unsignPmm(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.umax.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z31test_wave_reduce_max_u64_unsignPmm(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.umax.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_max_u64_unsign(unsigned long* out, unsigned long in) {
  *out = __builtin_amdgcn_wave_reduce_max_u64(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_and_b32_i32Pii
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.and" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_and_b32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.and.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z28test_wave_reduce_and_b32_i32Pii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.and.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_and_b32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_and_b32(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_and_b64_i64Pll
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.and" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_and_b64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.and.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z28test_wave_reduce_and_b64_i64Pll(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.and.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_and_b64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_and_b64(in, 0);
}

// CIR-LABEL: @_Z27test_wave_reduce_or_b32_i32Pii
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.or" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z27test_wave_reduce_or_b32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.or.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z27test_wave_reduce_or_b32_i32Pii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.or.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_or_b32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_or_b32(in, 0);
}

// CIR-LABEL: @_Z27test_wave_reduce_or_b64_i64Pll
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.or" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z27test_wave_reduce_or_b64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.or.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z27test_wave_reduce_or_b64_i64Pll(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.or.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_or_b64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_or_b64(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_xor_b32_i32Pii
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.xor" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_xor_b32_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.xor.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z28test_wave_reduce_xor_b32_i32Pii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.xor.i32(i32 %{{.*}}, i32 0)
__device__ void test_wave_reduce_xor_b32_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_xor_b32(in, 0);
}

// CIR-LABEL: @_Z28test_wave_reduce_xor_b64_i64Pll
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.xor" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @_Z28test_wave_reduce_xor_b64_i64Pll(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.xor.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @_Z28test_wave_reduce_xor_b64_i64Pll(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.xor.i64(i64 %{{.*}}, i32 0)
__device__ void test_wave_reduce_xor_b64_i64(long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_xor_b64(in, 0);
}

// CIR-LABEL: @_Z38test_wave_reduce_add_u32_iterative_i32Pii
// CIR: cir.const #cir.int<1> : !s32i
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z38test_wave_reduce_add_u32_iterative_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 1)
// OGCG: define{{.*}} void @_Z38test_wave_reduce_add_u32_iterative_i32Pii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 1)
__device__ void test_wave_reduce_add_u32_iterative_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 1);
}

// CIR-LABEL: @_Z32test_wave_reduce_add_u32_dpp_i32Pii
// CIR: cir.const #cir.int<2> : !s32i
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @_Z32test_wave_reduce_add_u32_dpp_i32Pii(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 2)
// OGCG: define{{.*}} void @_Z32test_wave_reduce_add_u32_dpp_i32Pii(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 2)
__device__ void test_wave_reduce_add_u32_dpp_i32(int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 2);
}

// CIR-LABEL: @_Z18test_div_scale_f64PdPidd
// CIR: cir.llvm.intrinsic "amdgcn.div.scale" {{.*}} : (!cir.double, !cir.double, !cir.bool)
// LLVM: define{{.*}} void @_Z18test_div_scale_f64PdPidd
// LLVM: call { double, i1 } @llvm.amdgcn.div.scale.f64(double %{{.+}}, double %{{.+}}, i1 true)
// OGCG: define{{.*}} void @_Z18test_div_scale_f64PdPidd
// OGCG: call { double, i1 } @llvm.amdgcn.div.scale.f64(double %{{.+}}, double %{{.+}}, i1 true)
__device__ void test_div_scale_f64(double* out, int* flagout, double a, double b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scale(a, b, true, &flag);
  *flagout = flag;
}

// CIR-LABEL: @_Z18test_div_scale_f32PfPbff
// CIR: cir.llvm.intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @_Z18test_div_scale_f32PfPbff
// LLVM: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
// OGCG: define{{.*}} void @_Z18test_div_scale_f32PfPbff
// OGCG: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
__device__ void test_div_scale_f32(float* out, bool* flagout, float a, float b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scalef(a, b, true, &flag);
  *flagout = flag;
}

// CIR-LABEL: @_Z27test_div_scale_f32_with_ptrPfPiPbff
// CIR: cir.llvm.intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @_Z27test_div_scale_f32_with_ptrPfPiPbff
// LLVM: {{.*}}call{{.*}} { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
// OGCG: define{{.*}} void @_Z27test_div_scale_f32_with_ptrPfPiPbff
// OGCG: {{.*}}call{{.*}} { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
__device__ void test_div_scale_f32_with_ptr(float* out, int* flagout, bool* flag, float a, float b)
{
  *out = __builtin_amdgcn_div_scalef(a, b, true, flag);
}

// CIR-LABEL: @_Z17test_div_fmas_f32Pdfffi
// CIR: cir.llvm.intrinsic "amdgcn.div.fmas" {{.*}} : (!cir.float, !cir.float, !cir.float, !cir.bool) -> !cir.float
// LLVM: define{{.*}} void @_Z17test_div_fmas_f32Pdfffi
// LLVM: call float @llvm.amdgcn.div.fmas.f32(float %{{.+}}, float %{{.+}}, float %{{.+}}, i1 %{{.*}})
// OGCG: define{{.*}} void @_Z17test_div_fmas_f32Pdfffi
// OGCG: call {{.*}} float @llvm.amdgcn.div.fmas.f32(float %{{.+}}, float %{{.+}}, float %{{.+}}, i1 %{{.*}})
__device__ void test_div_fmas_f32(double* out, float a, float b, float c, int d) {
  *out = __builtin_amdgcn_div_fmasf(a, b, c, d);
}

// CIR-LABEL: @_Z17test_div_fmas_f64Pddddi
// CIR: cir.llvm.intrinsic "amdgcn.div.fmas" {{.*}} : (!cir.double, !cir.double, !cir.double, !cir.bool) -> !cir.double
// LLVM: define{{.*}} void @_Z17test_div_fmas_f64Pddddi
// LLVM: call double @llvm.amdgcn.div.fmas.f64(double %{{.+}}, double %{{.+}}, double %{{.+}}, i1 %{{.*}})
// OGCG: define{{.*}} void @_Z17test_div_fmas_f64Pddddi
// OGCG: call {{.*}} double @llvm.amdgcn.div.fmas.f64(double %{{.+}}, double %{{.+}}, double %{{.+}}, i1 %{{.*}})
__device__ void test_div_fmas_f64(double* out, double a, double b, double c, int d) {
  *out = __builtin_amdgcn_div_fmas(a, b, c, d);
}

// CIR-LABEL: @_Z19test_ds_swizzle_i32Pii
// CIR: cir.llvm.intrinsic "amdgcn.ds.swizzle" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z19test_ds_swizzle_i32Pii
// LLVM: call i32 @llvm.amdgcn.ds.swizzle(i32 %{{.*}}, i32 32)
// OGCG: define{{.*}} void @_Z19test_ds_swizzle_i32Pii
// OGCG: call i32 @llvm.amdgcn.ds.swizzle(i32 %{{.*}}, i32 32)
__device__ void test_ds_swizzle_i32(int* out, int a) {
  *out = __builtin_amdgcn_ds_swizzle(a, 32);
}

// CIR-LABEL: @_Z13test_readlanePiii
// CIR: cir.llvm.intrinsic "amdgcn.readlane" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @_Z13test_readlanePiii
// LLVM: call i32 @llvm.amdgcn.readlane.i32(i32 %{{.*}}, i32 %{{.*}})
// OGCG: define{{.*}} void @_Z13test_readlanePiii
// OGCG: call i32 @llvm.amdgcn.readlane.i32(i32 %{{.*}}, i32 %{{.*}})
__device__ void test_readlane(int* out, int a, int b) {
  *out = __builtin_amdgcn_readlane(a, b);
}

// CIR-LABEL: @_Z18test_readfirstlanePii
// CIR: cir.llvm.intrinsic "amdgcn.readfirstlane" {{.*}} : (!s32i) -> !s32i
// LLVM: define{{.*}} void @_Z18test_readfirstlanePii
// LLVM: call i32 @llvm.amdgcn.readfirstlane.i32(i32 %{{.*}})
// OGCG: define{{.*}} void @_Z18test_readfirstlanePii
// OGCG: call i32 @llvm.amdgcn.readfirstlane.i32(i32 %{{.*}})
__device__ void test_readfirstlane(int* out, int a) {
  *out = __builtin_amdgcn_readfirstlane(a);
}

// CIR-LABEL: @_Z18test_div_fixup_f32Pffff
// CIR: cir.llvm.intrinsic "amdgcn.div.fixup" {{.*}} : (!cir.float, !cir.float, !cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z18test_div_fixup_f32Pffff
// LLVM: call{{.*}} float @llvm.amdgcn.div.fixup.f32(float %{{.+}}, float %{{.+}}, float %{{.+}})
// OGCG: define{{.*}} void @_Z18test_div_fixup_f32Pffff
// OGCG: call{{.*}} float @llvm.amdgcn.div.fixup.f32(float %{{.+}}, float %{{.+}}, float %{{.+}})
__device__ void test_div_fixup_f32(float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_div_fixupf(a, b, c);
}

// CIR-LABEL: @_Z18test_div_fixup_f64Pdddd
// CIR: cir.llvm.intrinsic "amdgcn.div.fixup" {{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z18test_div_fixup_f64Pdddd
// LLVM: call{{.*}} double @llvm.amdgcn.div.fixup.f64(double %{{.+}}, double %{{.+}}, double %{{.+}})
// OGCG: define{{.*}} void @_Z18test_div_fixup_f64Pdddd
// OGCG: call{{.*}} double @llvm.amdgcn.div.fixup.f64(double %{{.+}}, double %{{.+}}, double %{{.+}})
__device__ void test_div_fixup_f64(double* out, double a, double b, double c) {
  *out = __builtin_amdgcn_div_fixup(a, b, c);
}

// CIR-LABEL: @_Z19test_trig_preop_f32Pffi
// CIR: cir.llvm.intrinsic "amdgcn.trig.preop" {{.*}} : (!cir.float, !s32i) -> !cir.float
// LLVM: define{{.*}} void @_Z19test_trig_preop_f32Pffi
// LLVM: call{{.*}} float @llvm.amdgcn.trig.preop.f32(float %{{.+}}, i32 %{{.*}})
// OGCG: define{{.*}} void @_Z19test_trig_preop_f32Pffi
// OGCG: call{{.*}} float @llvm.amdgcn.trig.preop.f32(float %{{.+}}, i32 %{{.*}})
__device__ void test_trig_preop_f32(float* out, float a, int b) {
  *out = __builtin_amdgcn_trig_preopf(a, b);
}

// CIR-LABEL: @_Z19test_trig_preop_f64Pddi
// CIR: cir.llvm.intrinsic "amdgcn.trig.preop" {{.*}} : (!cir.double, !s32i) -> !cir.double
// LLVM: define{{.*}} void @_Z19test_trig_preop_f64Pddi
// LLVM: call{{.*}} double @llvm.amdgcn.trig.preop.f64(double %{{.+}}, i32 %{{.*}})
// OGCG: define{{.*}} void @_Z19test_trig_preop_f64Pddi
// OGCG: call{{.*}} double @llvm.amdgcn.trig.preop.f64(double %{{.+}}, i32 %{{.*}})
__device__ void test_trig_preop_f64(double* out, double a, int b) {
  *out = __builtin_amdgcn_trig_preop(a, b);
}

// CIR-LABEL: @_Z12test_rcp_f32Pff
// CIR: cir.llvm.intrinsic "amdgcn.rcp" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z12test_rcp_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.rcp.f32(float %{{.*}})
// OGCG: define{{.*}} void @_Z12test_rcp_f32Pff
// OGCG: call{{.*}} float @llvm.amdgcn.rcp.f32(float %{{.*}})
__device__ void test_rcp_f32(float* out, float a) {
  *out = __builtin_amdgcn_rcpf(a);
}

// CIR-LABEL: @_Z12test_rcp_f64Pdd
// CIR: cir.llvm.intrinsic "amdgcn.rcp" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z12test_rcp_f64Pdd
// LLVM: call{{.*}} double @llvm.amdgcn.rcp.f64(double %{{.*}})
// OGCG: define{{.*}} void @_Z12test_rcp_f64Pdd
// OGCG: call{{.*}} double @llvm.amdgcn.rcp.f64(double %{{.*}})
__device__ void test_rcp_f64(double* out, double a) {
  *out = __builtin_amdgcn_rcp(a);
}

// CIR-LABEL: @_Z13test_sqrt_f32Pff
// CIR: cir.llvm.intrinsic "amdgcn.sqrt" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z13test_sqrt_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.sqrt.f32(float %{{.*}})
// OGCG: define{{.*}} void @_Z13test_sqrt_f32Pff
// OGCG: call{{.*}} float @llvm.amdgcn.sqrt.f32(float %{{.*}})
__device__ void test_sqrt_f32(float* out, float a) {
  *out = __builtin_amdgcn_sqrtf(a);
}

// CIR-LABEL: @_Z13test_sqrt_f64Pdd
// CIR: cir.llvm.intrinsic "amdgcn.sqrt" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z13test_sqrt_f64Pdd
// LLVM: call{{.*}} double @llvm.amdgcn.sqrt.f64(double %{{.*}})
// OGCG: define{{.*}} void @_Z13test_sqrt_f64Pdd
// OGCG: call{{.*}} double @llvm.amdgcn.sqrt.f64(double %{{.*}})
__device__ void test_sqrt_f64(double* out, double a) {
  *out = __builtin_amdgcn_sqrt(a);
}

// CIR-LABEL: @_Z12test_rsq_f32Pff
// CIR: cir.llvm.intrinsic "amdgcn.rsq" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z12test_rsq_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.rsq.f32(float %{{.*}})
// OGCG: define{{.*}} void @_Z12test_rsq_f32Pff
// OGCG: call{{.*}} float @llvm.amdgcn.rsq.f32(float %{{.*}})
__device__ void test_rsq_f32(float* out, float a)
{
  *out = __builtin_amdgcn_rsqf(a);
}

// CIR-LABEL: @_Z12test_rsq_f64Pdd
// CIR: cir.llvm.intrinsic "amdgcn.rsq" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z12test_rsq_f64Pdd
// LLVM: call{{.*}} double @llvm.amdgcn.rsq.f64(double %{{.*}})
// OGCG: define{{.*}} void @_Z12test_rsq_f64Pdd
// OGCG: call{{.*}} double @llvm.amdgcn.rsq.f64(double %{{.*}})
__device__ void test_rsq_f64(double* out, double a) {
  *out = __builtin_amdgcn_rsq(a);
}

// CIR-LABEL: @_Z18test_rsq_clamp_f32Pff
// CIR: cir.llvm.intrinsic "amdgcn.rsq.clamp" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @_Z18test_rsq_clamp_f32Pff
// LLVM: call{{.*}} float @llvm.amdgcn.rsq.clamp.f32(float %{{.*}})
// OGCG: define{{.*}} void @_Z18test_rsq_clamp_f32Pff
// OGCG: call{{.*}} float @llvm.amdgcn.rsq.clamp.f32(float %{{.*}})
__device__ void test_rsq_clamp_f32(float* out, float a)
{
  *out = __builtin_amdgcn_rsq_clampf(a);
}

// CIR-LABEL: @_Z18test_rsq_clamp_f64Pdd
// CIR: cir.llvm.intrinsic "amdgcn.rsq.clamp" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @_Z18test_rsq_clamp_f64Pdd
// LLVM: call{{.*}} double @llvm.amdgcn.rsq.clamp.f64(double %{{.*}})
// OGCG: define{{.*}} void @_Z18test_rsq_clamp_f64Pdd
// OGCG: call{{.*}} double @llvm.amdgcn.rsq.clamp.f64(double %{{.*}})
__device__ void test_rsq_clamp_f64(double* out, double a) {
  *out = __builtin_amdgcn_rsq_clamp(a);
}
