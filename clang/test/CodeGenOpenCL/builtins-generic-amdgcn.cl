// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -Wno-error=int-conversion -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CHECK-LABEL: @test_builtin_clz(
// CHECK: tail call i32 @llvm.ctlz.i32(i32 %a, i1 true)
void test_builtin_clz(global int* out, int a)
{
  *out = __builtin_clz(a);
}

// CHECK-LABEL: @test_builtin_clzl(
// CHECK: tail call i64 @llvm.ctlz.i64(i64 %a, i1 true)
void test_builtin_clzl(global long* out, long a)
{
  *out = __builtin_clzl(a);
}

// CHECK: tail call ptr addrspace(5) @llvm.frameaddress.p5(i32 0)
void test_builtin_frame_address(int *out) {
    *out = __builtin_frame_address(0);
}

// CHECK-LABEL: @test_builtin_ldexpf16(
// CHECK: tail call half @llvm.ldexp.f16.i32(half %v, i32 %e)
half test_builtin_ldexpf16(half v, int e) {
  return __builtin_ldexpf16(v, e);
}

// CHECK-LABEL: @test_builtin_ldexpf(
// CHECK: tail call float @llvm.ldexp.f32.i32(float %v, i32 %e)
float test_builtin_ldexpf(float v, int e) {
  return __builtin_ldexpf(v, e);
}

// CHECK-LABEL: @test_builtin_ldexp(
// CHECK: tail call double @llvm.ldexp.f64.i32(double %v, i32 %e)
double test_builtin_ldexp(double v, int e) {
  return __builtin_ldexp(v, e);
}
