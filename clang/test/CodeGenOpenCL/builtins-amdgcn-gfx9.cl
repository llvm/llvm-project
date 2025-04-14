// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef unsigned int uint;
typedef unsigned long ulong;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef unsigned short __attribute__((ext_vector_type(2))) ushort2;

// CHECK-LABEL: @test_fmed3_f16
// CHECK: call half @llvm.amdgcn.fmed3.f16(half %a, half %b, half %c)
void test_fmed3_f16(global half* out, half a, half b, half c)
{
  *out = __builtin_amdgcn_fmed3h(a, b, c);
}

// CHECK-LABEL: @test_s_memtime
// CHECK: call i64 @llvm.amdgcn.s.memtime()
void test_s_memtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memtime();
}

// CHECK-LABEL: @test_groupstaticsize
// CHECK: call i32 @llvm.amdgcn.groupstaticsize()
void test_groupstaticsize(global uint* out)
{
  *out = __builtin_amdgcn_groupstaticsize();
}

// CHECK-LABEL: define dso_local void @test_cvt_pk_norm_i16_f16(
// CHECK: call <2 x i16> @llvm.amdgcn.cvt.pk.norm.i16.f16(half %src0, half %src1)
void test_cvt_pk_norm_i16_f16(global short2* out, half src0, half src1)
{
  *out = __builtin_amdgcn_cvt_pk_norm_i16_f16(src0, src1);
}

// CHECK-LABEL: define dso_local void @test_cvt_pk_norm_u16_f16(
// CHECK: call <2 x i16> @llvm.amdgcn.cvt.pk.norm.u16.f16(half %src0, half %src1)
void test_cvt_pk_norm_u16_f16(global ushort2* out, half src0, half src1)
{
  *out = __builtin_amdgcn_cvt_pk_norm_u16_f16(src0, src1);
}
