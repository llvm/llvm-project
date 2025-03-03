// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1011 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1012 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef unsigned int uint;
typedef unsigned long ulong;

// CHECK-LABEL: @test_permlane16(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.permlane16.i32(i32 %a, i32 %b, i32 %c, i32 %d, i1 false, i1 false)
void test_permlane16(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlane16(a, b, c, d, 0, 0);
}

// CHECK-LABEL: @test_permlanex16(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.permlanex16.i32(i32 %a, i32 %b, i32 %c, i32 %d, i1 false, i1 false)
void test_permlanex16(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 0, 0);
}

// CHECK-LABEL: @test_mov_dpp8_uint(
// CHECK:      {{.*}}call{{.*}} i32 @llvm.amdgcn.mov.dpp8.i32(i32 %a, i32 1)
// CHECK-NEXT: store i32 %0,
void test_mov_dpp8_uint(global uint* out, uint a) {
  *out = __builtin_amdgcn_mov_dpp8(a, 1);
}

// CHECK-LABEL: @test_mov_dpp8_long(
// CHECK:      {{.*}}call{{.*}} i64 @llvm.amdgcn.mov.dpp8.i64(i64 %a, i32 1)
// CHECK-NEXT: store i64 %0,
void test_mov_dpp8_long(global long* out, long a) {
  *out = __builtin_amdgcn_mov_dpp8(a, 1);
}

// CHECK-LABEL: @test_mov_dpp8_float(
// CHECK:      %0 = tail call{{.*}} float @llvm.amdgcn.mov.dpp8.f32(float %a, i32 1)
// CHECK-NEXT: store float %0,
void test_mov_dpp8_float(global float* out, float a) {
  *out = __builtin_amdgcn_mov_dpp8(a, 1);
}

// CHECK-LABEL: @test_mov_dpp8_double
// CHECK:      %0 = tail call{{.*}} double @llvm.amdgcn.mov.dpp8.f64(double %x, i32 1)
// CHECK-NEXT: store double %0,
void test_mov_dpp8_double(double x, global double *p) {
  *p = __builtin_amdgcn_mov_dpp8(x, 1);
}

// CHECK-LABEL: @test_mov_dpp8_short
// CHECK:      %0 = tail call{{.*}} i16 @llvm.amdgcn.mov.dpp8.i16(i16 %x, i32 1)
// CHECK-NEXT: store i16 %0,
void test_mov_dpp8_short(short x, global short *p) {
  *p = __builtin_amdgcn_mov_dpp8(x, 1);
}

// CHECK-LABEL: @test_mov_dpp8_char
// CHECK:      %0 = tail call{{.*}} i8 @llvm.amdgcn.mov.dpp8.i8(i8 %x, i32 1)
// CHECK-NEXT: store i8 %0,
void test_mov_dpp8_char(char x, global char *p) {
  *p = __builtin_amdgcn_mov_dpp8(x, 1);
}

// CHECK-LABEL: @test_mov_dpp8_half
// CHECK:      %0 = load half,
// CHECK-NEXT: %1 = tail call{{.*}} half @llvm.amdgcn.mov.dpp8.f16(half %0, i32 1)
// CHECK-NEXT: store half %1,
void test_mov_dpp8_half(half *x, global half *p) {
  *p = __builtin_amdgcn_mov_dpp8(*x, 1);
}

// CHECK-LABEL: @test_s_memtime
// CHECK: {{.*}}call{{.*}} i64 @llvm.amdgcn.s.memtime()
void test_s_memtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memtime();
}

// CHECK-LABEL: @test_groupstaticsize
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.groupstaticsize()
void test_groupstaticsize(global uint* out)
{
  *out = __builtin_amdgcn_groupstaticsize();
}

// CHECK-LABEL: @test_ballot_wave32(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.ballot.i32(i1 %{{.+}})
void test_ballot_wave32(global uint* out, int a, int b)
{
  *out = __builtin_amdgcn_ballot_w32(a == b);
}
