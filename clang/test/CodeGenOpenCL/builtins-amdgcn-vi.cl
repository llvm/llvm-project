// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu tonga -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef unsigned long ulong;

// CHECK-LABEL: @test_div_fixup_f16
// CHECK: call half @llvm.amdgcn.div.fixup.f16
void test_div_fixup_f16(global half* out, half a, half b, half c)
{
  *out = __builtin_amdgcn_div_fixuph(a, b, c);
}

// CHECK-LABEL: @test_rcp_f16
// CHECK: call half @llvm.amdgcn.rcp.f16
void test_rcp_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_rcph(a);
}

// CHECK-LABEL: @test_rsq_f16
// CHECK: call half @llvm.amdgcn.rsq.f16
void test_rsq_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_rsqh(a);
}

// CHECK-LABEL: @test_sin_f16
// CHECK: call half @llvm.amdgcn.sin.f16
void test_sin_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_sinh(a);
}

// CHECK-LABEL: @test_cos_f16
// CHECK: call half @llvm.amdgcn.cos.f16
void test_cos_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_cosh(a);
}

// CHECK-LABEL: @test_ldexp_f16
// CHECK: call half @llvm.amdgcn.ldexp.f16
void test_ldexp_f16(global half* out, half a, int b)
{
  *out = __builtin_amdgcn_ldexph(a, b);
}

// CHECK-LABEL: @test_frexp_mant_f16
// CHECK: call half @llvm.amdgcn.frexp.mant.f16
void test_frexp_mant_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_frexp_manth(a);
}

// CHECK-LABEL: @test_frexp_exp_f16
// CHECK: call i16 @llvm.amdgcn.frexp.exp.i16.f16
void test_frexp_exp_f16(global short* out, half a)
{
  *out = __builtin_amdgcn_frexp_exph(a);
}

// CHECK-LABEL: @test_fract_f16
// CHECK: call half @llvm.amdgcn.fract.f16
void test_fract_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_fracth(a);
}

// CHECK-LABEL: @test_class_f16
// CHECK: call i1 @llvm.amdgcn.class.f16
void test_class_f16(global half* out, half a, int b)
{
  *out = __builtin_amdgcn_classh(a, b);
}

// CHECK-LABEL: @test_s_memrealtime
// CHECK: call i64 @llvm.amdgcn.s.memrealtime()
void test_s_memrealtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memrealtime();
}

// CHECK-LABEL: @test_mov_dpp
// CHECK: call i32 @llvm.amdgcn.mov.dpp.i32(i32 %src, i32 0, i32 0, i32 0, i1 false)
void test_mov_dpp(global int* out, int src)
{
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0, 0, false);
}

// CHECK-LABEL: @test_ds_fadd
// CHECK: call float @llvm.amdgcn.ds.fadd(float addrspace(3)* %out, float %src, i32 0, i32 0, i1 false)
void test_ds_faddf(local float *out, float src) {
  *out = __builtin_amdgcn_ds_faddf(out, src, 0, 0, false);
}

// CHECK-LABEL: @test_ds_fmin
// CHECK: call float @llvm.amdgcn.ds.fmin(float addrspace(3)* %out, float %src, i32 0, i32 0, i1 false)
void test_ds_fminf(local float *out, float src) {
  *out = __builtin_amdgcn_ds_fminf(out, src, 0, 0, false);
}

// CHECK-LABEL: @test_ds_fmax
// CHECK: call float @llvm.amdgcn.ds.fmax(float addrspace(3)* %out, float %src, i32 0, i32 0, i1 false)
void test_ds_fmaxf(local float *out, float src) {
  *out = __builtin_amdgcn_ds_fmaxf(out, src, 0, 0, false);
}
