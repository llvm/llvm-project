// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx940 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-GFX940

typedef float  v2f   __attribute__((ext_vector_type(2)));

// CHECK-GFX940-LABEL: @test_cvt_f32_bf8
// CHECK-GFX940: call float @llvm.amdgcn.cvt.f32.bf8(i32 %a, i32 0)
void test_cvt_f32_bf8(global int* out, int a)
{
  *out = __builtin_amdgcn_cvt_f32_bf8(a, 0);
}

// CHECK-GFX940-LABEL: @test_cvt_f32_fp8
// CHECK-GFX940: call float @llvm.amdgcn.cvt.f32.fp8(i32 %a, i32 1)
void test_cvt_f32_fp8(global int* out, int a)
{
  *out = __builtin_amdgcn_cvt_f32_fp8(a, 1);
}

// CHECK-GFX940-LABEL: @test_cvt_pk_f32_bf8
// CHECK-GFX940: call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %a, i1 false)
void test_cvt_pk_f32_bf8(global v2f* out, int a)
{
  *out = __builtin_amdgcn_cvt_pk_f32_bf8(a, false);
}

// CHECK-GFX940-LABEL: @test_cvt_pk_f32_fp8
// CHECK-GFX940: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %a, i1 true)
void test_cvt_pk_f32_fp8(global v2f* out, int a)
{
  *out = __builtin_amdgcn_cvt_pk_f32_fp8(a, true);
}

// CHECK-GFX940-LABEL: @test_cvt_pk_bf8_f32
// CHECK-GFX940: call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %a, float %b, i32 %old, i1 false)
void test_cvt_pk_bf8_f32(global int* out, int old, float a, float b)
{
  *out = __builtin_amdgcn_cvt_pk_bf8_f32(a, b, old, false);
}

// CHECK-GFX940-LABEL: @test_cvt_pk_fp8_f32
// CHECK-GFX940: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 %old, i1 true)
void test_cvt_pk_fp8_f32(global int* out, int old, float a, float b)
{
  *out = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, true);
}

// CHECK-GFX940-LABEL: @test_cvt_sr_bf8_f32
// CHECK-GFX940: call i32 @llvm.amdgcn.cvt.sr.bf8.f32(float %a, i32 %b, i32 %old, i32 2)
void test_cvt_sr_bf8_f32(global int* out, int old, float a, int b)
{
  *out = __builtin_amdgcn_cvt_sr_bf8_f32(a, b, old, 2);
}

// CHECK-GFX940-LABEL: @test_cvt_sr_fp8_f32
// CHECK-GFX940: call i32 @llvm.amdgcn.cvt.sr.fp8.f32(float %a, i32 %b, i32 %old, i32 3)
void test_cvt_sr_fp8_f32(global int* out, int old, float a, int b)
{
  *out = __builtin_amdgcn_cvt_sr_fp8_f32(a, b, old, 3);
}
