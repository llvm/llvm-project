// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu tahiti -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu tahiti -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu tahiti -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef unsigned long ulong;
typedef unsigned int uint;

// CIR-LABEL: @test_wave_reduce_add_u32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_add_u32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_add_u32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_add_u32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_add_u64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @test_wave_reduce_add_u64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.add.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_add_u64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.add.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_add_u64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_add_u64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_sub_u32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.sub" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_sub_u32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.sub.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_sub_u32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.sub.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_sub_u32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_sub_u32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_sub_u64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.sub" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @test_wave_reduce_sub_u64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.sub.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_sub_u64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.sub.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_sub_u64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_sub_u64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_min_i32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.min" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_min_i32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.min.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_min_i32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.min.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_min_i32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_min_i32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_min_u32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umin" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_min_u32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_min_u32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_min_u32(global uint* out, uint in) {
  *out = __builtin_amdgcn_wave_reduce_min_u32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_min_i64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.min" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_min_i64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.min.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_min_i64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.min.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_min_i64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_min_i64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_min_u64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umin" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @test_wave_reduce_min_u64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.umin.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_min_u64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.umin.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_min_u64(global ulong* out, ulong in) {
  *out = __builtin_amdgcn_wave_reduce_min_u64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_max_i32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.max" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_max_i32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.max.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_max_i32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.max.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_max_i32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_max_i32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_max_u32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umax" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_max_u32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.umax.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_max_u32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.umax.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_max_u32(global uint* out, uint in) {
  *out = __builtin_amdgcn_wave_reduce_max_u32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_max_i64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.max" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_max_i64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.max.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_max_i64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.max.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_max_i64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_max_i64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_max_u64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umax" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @test_wave_reduce_max_u64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.umax.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_max_u64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.umax.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_max_u64(global ulong* out, ulong in) {
  *out = __builtin_amdgcn_wave_reduce_max_u64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_and_b32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.and" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_and_b32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.and.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_and_b32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.and.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_and_b32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_and_b32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_and_b64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.and" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_and_b64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.and.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_and_b64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.and.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_and_b64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_and_b64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_or_b32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.or" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_or_b32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.or.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_or_b32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.or.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_or_b32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_or_b32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_or_b64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.or" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_or_b64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.or.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_or_b64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.or.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_or_b64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_or_b64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_xor_b32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.xor" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_xor_b32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.xor.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_xor_b32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.xor.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_xor_b32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_xor_b32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_xor_b64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.xor" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_xor_b64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.xor.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_xor_b64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.xor.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_xor_b64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_xor_b64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_add_u32_iterative
// CIR: cir.const #cir.int<1> : !s32i
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_add_u32_iterative(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 1)
// OGCG: define{{.*}} void @test_wave_reduce_add_u32_iterative(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 1)
void test_wave_reduce_add_u32_iterative(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 1);
}

// CIR: cir.const #cir.int<2> : !s32i
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_add_u32_dpp(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 2)
// OGCG: define{{.*}} void @test_wave_reduce_add_u32_dpp(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 2)
void test_wave_reduce_add_u32_dpp(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 2);
}

// CIR-LABEL: @test_div_scale_f64
// CIR: cir.llvm.intrinsic "amdgcn.div.scale" {{.*}} : (!cir.double, !cir.double, !cir.bool)
// LLVM: define{{.*}} void @test_div_scale_f64
// LLVM: call { double, i1 } @llvm.amdgcn.div.scale.f64(double %{{.+}}, double %{{.+}}, i1 true)
// OGCG: define{{.*}} void @test_div_scale_f64
// OGCG: call { double, i1 } @llvm.amdgcn.div.scale.f64(double %{{.+}}, double %{{.+}}, i1 true)
void test_div_scale_f64(global double* out, global int* flagout, double a, double b, global bool* flag)
{
  *out = __builtin_amdgcn_div_scale(a, b, true, flag);
  *flagout = *flag;
}

// CIR-LABEL: @test_div_scale_f32
// CIR: cir.llvm.intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @test_div_scale_f32
// LLVM: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
// OGCG: define{{.*}} void @test_div_scale_f32
// OGCG: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
void test_div_scale_f32(global float* out, global bool* flagout, float a, float b, global bool* flag)
{
  *out = __builtin_amdgcn_div_scalef(a, b, true, flag);
  *flagout = *flag;
}

// CIR-LABEL: @test_div_scale_f32_global_ptr
// CIR: cir.llvm.intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @test_div_scale_f32_global_ptr
// LLVM: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
// OGCG: define{{.*}} void @test_div_scale_f32_global_ptr
// OGCG: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
void test_div_scale_f32_global_ptr(global float* out, global int* flagout, float a, float b, global bool* flag)
{
  *out = __builtin_amdgcn_div_scalef(a, b, true, flag);
}

// CIR-LABEL: @test_div_scale_f32_generic_ptr
// CIR: cir.llvm.intrinsic "amdgcn.div.scale" {{.*}} : (!cir.float, !cir.float, !cir.bool)
// LLVM: define{{.*}} void @test_div_scale_f32_generic_ptr
// LLVM: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
// OGCG: define{{.*}} void @test_div_scale_f32_generic_ptr
// OGCG: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{.+}}, float %{{.+}}, i1 true)
void test_div_scale_f32_generic_ptr(global float* out, global int* flagout, float a, float b, generic bool* flag)
{
  *out = __builtin_amdgcn_div_scalef(a, b, true, flag);
}

// CIR-LABEL: @test_div_fmas_f32
// CIR: cir.llvm.intrinsic "amdgcn.div.fmas" {{.*}} : (!cir.float, !cir.float, !cir.float, !cir.bool) -> !cir.float
// LLVM: define{{.*}} void @test_div_fmas_f32
// LLVM: call float @llvm.amdgcn.div.fmas.f32(float %{{.+}}, float %{{.+}}, float %{{.+}}, i1 %{{.*}})
// OGCG: define{{.*}} void @test_div_fmas_f32
// OGCG: call float @llvm.amdgcn.div.fmas.f32(float %{{.+}}, float %{{.+}}, float %{{.+}}, i1 %{{.*}})
void test_div_fmas_f32(global float* out, float a, float b, float c, int d)
{
  *out = __builtin_amdgcn_div_fmasf(a, b, c, d);
}

// CIR-LABEL: @test_div_fmas_f64
// CIR: cir.llvm.intrinsic "amdgcn.div.fmas" {{.*}} : (!cir.double, !cir.double, !cir.double, !cir.bool) -> !cir.double
// LLVM: define{{.*}} void @test_div_fmas_f64
// LLVM: call double @llvm.amdgcn.div.fmas.f64(double %{{.+}}, double %{{.+}}, double %{{.+}}, i1 %{{.*}})
// OGCG: define{{.*}} void @test_div_fmas_f64
// OGCG: call double @llvm.amdgcn.div.fmas.f64(double %{{.+}}, double %{{.+}}, double %{{.+}}, i1 %{{.*}})
void test_div_fmas_f64(global double* out, double a, double b, double c, int d)
{
  *out = __builtin_amdgcn_div_fmas(a, b, c, d);
}

// CIR-LABEL: @test_ds_swizzle
// CIR: cir.llvm.intrinsic "amdgcn.ds.swizzle" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_ds_swizzle
// LLVM: call i32 @llvm.amdgcn.ds.swizzle(i32 %{{.*}}, i32 32)
// OGCG: define{{.*}} void @test_ds_swizzle
// OGCG: call i32 @llvm.amdgcn.ds.swizzle(i32 %{{.*}}, i32 32)
void test_ds_swizzle(global int* out, int a) {
  *out = __builtin_amdgcn_ds_swizzle(a, 32);
}

// CIR-LABEL: @test_readlane
// CIR: cir.llvm.intrinsic "amdgcn.readlane" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_readlane
// LLVM: call i32 @llvm.amdgcn.readlane.i32(i32 %{{.*}}, i32 %{{.*}})
// OGCG: define{{.*}} void @test_readlane
// OGCG: call i32 @llvm.amdgcn.readlane.i32(i32 %{{.*}}, i32 %{{.*}})
void test_readlane(global int* out, int a, int b) {
  *out = __builtin_amdgcn_readlane(a, b);
}

// CIR-LABEL: @test_readfirstlane
// CIR: cir.llvm.intrinsic "amdgcn.readfirstlane" {{.*}} : (!s32i) -> !s32i
// LLVM: define{{.*}} void @test_readfirstlane
// LLVM: call i32 @llvm.amdgcn.readfirstlane.i32(i32 %{{.*}})
// OGCG: define{{.*}} void @test_readfirstlane
// OGCG: call i32 @llvm.amdgcn.readfirstlane.i32(i32 %{{.*}})
void test_readfirstlane(global int* out, int a) {
  *out = __builtin_amdgcn_readfirstlane(a);
}

// CIR-LABEL: @test_div_fixup_f32
// CIR: cir.llvm.intrinsic "amdgcn.div.fixup" {{.*}} : (!cir.float, !cir.float, !cir.float) -> !cir.float
// LLVM: define{{.*}} void @test_div_fixup_f32
// LLVM: call{{.*}} float @llvm.amdgcn.div.fixup.f32(float %{{.+}}, float %{{.+}}, float %{{.+}})
// OGCG: define{{.*}} void @test_div_fixup_f32
// OGCG: call{{.*}} float @llvm.amdgcn.div.fixup.f32(float %{{.+}}, float %{{.+}}, float %{{.+}})
void test_div_fixup_f32(global float* out, float a, float b, float c)
{
  *out = __builtin_amdgcn_div_fixupf(a, b, c);
}

// CIR-LABEL: @test_div_fixup_f64
// CIR: cir.llvm.intrinsic "amdgcn.div.fixup" {{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double
// LLVM: define{{.*}} void @test_div_fixup_f64
// LLVM: call{{.*}} double @llvm.amdgcn.div.fixup.f64(double %{{.+}}, double %{{.+}}, double %{{.+}})
// OGCG: define{{.*}} void @test_div_fixup_f64
// OGCG: call{{.*}} double @llvm.amdgcn.div.fixup.f64(double %{{.+}}, double %{{.+}}, double %{{.+}})
void test_div_fixup_f64(global double* out, double a, double b, double c)
{
  *out = __builtin_amdgcn_div_fixup(a, b, c);
}

// CIR-LABEL: @test_trig_preop_f32
// CIR: cir.llvm.intrinsic "amdgcn.trig.preop" {{.*}} : (!cir.float, !s32i) -> !cir.float
// LLVM: define{{.*}} void @test_trig_preop_f32
// LLVM: call{{.*}} float @llvm.amdgcn.trig.preop.f32(float %{{.+}}, i32 %{{.*}})
// OGCG: define{{.*}} void @test_trig_preop_f32
// OGCG: call{{.*}} float @llvm.amdgcn.trig.preop.f32(float %{{.+}}, i32 %{{.*}})
void test_trig_preop_f32(global float* out, float a, int b) {
  *out = __builtin_amdgcn_trig_preopf(a, b);
}

// CIR-LABEL: @test_trig_preop_f64
// CIR: cir.llvm.intrinsic "amdgcn.trig.preop" {{.*}} : (!cir.double, !s32i) -> !cir.double
// LLVM: define{{.*}} void @test_trig_preop_f64
// LLVM: call{{.*}} double @llvm.amdgcn.trig.preop.f64(double %{{.+}}, i32 %{{.*}})
// OGCG: define{{.*}} void @test_trig_preop_f64
// OGCG: call{{.*}} double @llvm.amdgcn.trig.preop.f64(double %{{.+}}, i32 %{{.*}})
void test_trig_preop_f64(global double* out, double a, int b) {
  *out = __builtin_amdgcn_trig_preop(a, b);
}

// CIR-LABEL: @test_rcp_f32
// CIR: cir.llvm.intrinsic "amdgcn.rcp" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @test_rcp_f32
// LLVM: call{{.*}} float @llvm.amdgcn.rcp.f32(float %{{.*}})
// OGCG: define{{.*}} void @test_rcp_f32
// OGCG: call{{.*}} float @llvm.amdgcn.rcp.f32(float %{{.*}})
void test_rcp_f32(global float* out, float a) {
  *out = __builtin_amdgcn_rcpf(a);
}

// CIR-LABEL: @test_rcp_f64
// CIR: cir.llvm.intrinsic "amdgcn.rcp" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @test_rcp_f64
// LLVM: call{{.*}} double @llvm.amdgcn.rcp.f64(double %{{.*}})
// OGCG: define{{.*}} void @test_rcp_f64
// OGCG: call{{.*}} double @llvm.amdgcn.rcp.f64(double %{{.*}})
void test_rcp_f64(global double* out, double a) {
  *out = __builtin_amdgcn_rcp(a);
}

// CIR-LABEL: @test_sqrt_f32
// CIR: cir.llvm.intrinsic "amdgcn.sqrt" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @test_sqrt_f32
// LLVM: call{{.*}} float @llvm.amdgcn.sqrt.f32(float %{{.*}})
// OGCG: define{{.*}} void @test_sqrt_f32
// OGCG: call{{.*}} float @llvm.amdgcn.sqrt.f32(float %{{.*}})
void test_sqrt_f32(global float* out, float a) {
  *out = __builtin_amdgcn_sqrtf(a);
}

// CIR-LABEL: @test_sqrt_f64
// CIR: cir.llvm.intrinsic "amdgcn.sqrt" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @test_sqrt_f64
// LLVM: call{{.*}} double @llvm.amdgcn.sqrt.f64(double %{{.*}})
// OGCG: define{{.*}} void @test_sqrt_f64
// OGCG: call{{.*}} double @llvm.amdgcn.sqrt.f64(double %{{.*}})
void test_sqrt_f64(global double* out, double a) {
  *out = __builtin_amdgcn_sqrt(a);
}

// CIR-LABEL: @test_rsq_f32
// CIR: cir.llvm.intrinsic "amdgcn.rsq" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @test_rsq_f32
// LLVM: call{{.*}} float @llvm.amdgcn.rsq.f32(float %{{.*}})
// OGCG: define{{.*}} void @test_rsq_f32
// OGCG: call{{.*}} float @llvm.amdgcn.rsq.f32(float %{{.*}})
void test_rsq_f32(global float* out, float a) {
  *out = __builtin_amdgcn_rsqf(a);
}

// CIR-LABEL: @test_rsq_f64
// CIR: cir.llvm.intrinsic "amdgcn.rsq" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @test_rsq_f64
// LLVM: call{{.*}} double @llvm.amdgcn.rsq.f64(double %{{.*}})
// OGCG: define{{.*}} void @test_rsq_f64
// OGCG: call{{.*}} double @llvm.amdgcn.rsq.f64(double %{{.*}})
void test_rsq_f64(global double* out, double a) {
  *out = __builtin_amdgcn_rsq(a);
}

// CIR-LABEL: @test_rsq_clamp_f32
// CIR: cir.llvm.intrinsic "amdgcn.rsq.clamp" {{.*}} : (!cir.float) -> !cir.float
// LLVM: define{{.*}} void @test_rsq_clamp_f32
// LLVM: call{{.*}} float @llvm.amdgcn.rsq.clamp.f32(float %{{.*}})
// OGCG: define{{.*}} void @test_rsq_clamp_f32
// OGCG: call{{.*}} float @llvm.amdgcn.rsq.clamp.f32(float %{{.*}})
void test_rsq_clamp_f32(global float* out, float a) {
  *out = __builtin_amdgcn_rsq_clampf(a);
}

// CIR-LABEL: @test_rsq_clamp_f64
// CIR: cir.llvm.intrinsic "amdgcn.rsq.clamp" {{.*}} : (!cir.double) -> !cir.double
// LLVM: define{{.*}} void @test_rsq_clamp_f64
// LLVM: call{{.*}} double @llvm.amdgcn.rsq.clamp.f64(double %{{.*}})
// OGCG: define{{.*}} void @test_rsq_clamp_f64
// OGCG: call{{.*}} double @llvm.amdgcn.rsq.clamp.f64(double %{{.*}})
void test_rsq_clamp_f64(global double* out, double a) {
  *out = __builtin_amdgcn_rsq_clamp(a);
}
