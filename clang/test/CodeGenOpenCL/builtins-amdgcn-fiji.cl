// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu fiji -emit-llvm -o - %s | FileCheck -enable-var-scope --check-prefixes=CHECK %s


#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef ushort __attribute__((ext_vector_type(2))) ushort2;
typedef uint __attribute__((ext_vector_type(4))) uint4;

// CHECK-LABEL: @test_lerp
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.lerp
void test_lerp(global int* out, int a, int b, int c)
{
  *out = __builtin_amdgcn_lerp(a, b, c);
}

// CHECK-LABEL: @test_cubeid(
// CHECK: {{.*}}call{{.*}} float @llvm.amdgcn.cubeid(float %a, float %b, float %c)
void test_cubeid(global float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_cubeid(a, b, c);
}

// CHECK-LABEL: @test_cubesc(
// CHECK: {{.*}}call{{.*}} float @llvm.amdgcn.cubesc(float %a, float %b, float %c)
void test_cubesc(global float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_cubesc(a, b, c);
}

// CHECK-LABEL: @test_cubetc(
// CHECK: {{.*}}call{{.*}} float @llvm.amdgcn.cubetc(float %a, float %b, float %c)
void test_cubetc(global float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_cubetc(a, b, c);
}

// CHECK-LABEL: @test_cubema(
// CHECK: {{.*}}call{{.*}} float @llvm.amdgcn.cubema(float %a, float %b, float %c)
void test_cubema(global float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_cubema(a, b, c);
}

// CHECK-LABEL: @test_cvt_pknorm_i16(
// CHECK: tail call{{.*}} <2 x i16> @llvm.amdgcn.cvt.pknorm.i16(float %src0, float %src1)
kernel void test_cvt_pknorm_i16(global short2* out, float src0, float src1) {
  *out = __builtin_amdgcn_cvt_pknorm_i16(src0, src1);
}

// CHECK-LABEL: @test_cvt_pknorm_u16(
// CHECK: tail call{{.*}} <2 x i16> @llvm.amdgcn.cvt.pknorm.u16(float %src0, float %src1)
kernel void test_cvt_pknorm_u16(global ushort2* out, float src0, float src1) {
  *out = __builtin_amdgcn_cvt_pknorm_u16(src0, src1);
}

// CHECK-LABEL: @test_sad_u8(
// CHECK: tail call{{.*}} i32 @llvm.amdgcn.sad.u8(i32 %src0, i32 %src1, i32 %src2)
kernel void test_sad_u8(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_sad_u8(src0, src1, src2);
}

// CHECK-LABEL: test_msad_u8(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.msad.u8(i32 %src0, i32 %src1, i32 %src2)
kernel void test_msad_u8(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_msad_u8(src0, src1, src2);
}

// CHECK-LABEL: test_sad_hi_u8(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.sad.hi.u8(i32 %src0, i32 %src1, i32 %src2)
kernel void test_sad_hi_u8(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_sad_hi_u8(src0, src1, src2);
}

// CHECK-LABEL: @test_sad_u16(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.sad.u16(i32 %src0, i32 %src1, i32 %src2)
kernel void test_sad_u16(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_sad_u16(src0, src1, src2);
}

// CHECK-LABEL: @test_qsad_pk_u16_u8(
// CHECK: {{.*}}call{{.*}} i64 @llvm.amdgcn.qsad.pk.u16.u8(i64 %src0, i32 %src1, i64 %src2)
kernel void test_qsad_pk_u16_u8(global ulong* out, ulong src0, uint src1, ulong src2) {
  *out = __builtin_amdgcn_qsad_pk_u16_u8(src0, src1, src2);
}
