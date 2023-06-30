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

// CHECK-LABEL: @test_builtin_frexpf16(
// CHECK: [[VAL:%.+]] = tail call { half, i32 } @llvm.frexp.f16.i32(half %v)
// CHECK: [[EXTRACT_1:%.+]] = extractvalue { half, i32 } [[VAL]], 1
// CHECK: store i32 [[EXTRACT_1]], ptr addrspace(5)
// CHECK: [[EXTRACT_0:%.+]] = extractvalue { half, i32 } [[VAL]], 0
// CHECK: ret half [[EXTRACT_0]]
half test_builtin_frexpf16(half v, int* e) {
  return __builtin_frexpf16(v, e);
}

// CHECK-LABEL: @test_builtin_frexpf(
// CHECK: [[VAL:%.+]] = tail call { float, i32 } @llvm.frexp.f32.i32(float %v)
// CHECK: [[EXTRACT_1:%.+]] = extractvalue { float, i32 } [[VAL]], 1
// CHECK: store i32 [[EXTRACT_1]], ptr addrspace(5)
// CHECK: [[EXTRACT_0:%.+]] = extractvalue { float, i32 } [[VAL]], 0
// CHECK: ret float [[EXTRACT_0]]
float test_builtin_frexpf(float v, int* e) {
  return __builtin_frexpf(v, e);
}

// CHECK-LABEL: @test_builtin_frexp(
// CHECK: [[VAL:%.+]] = tail call { double, i32 } @llvm.frexp.f64.i32(double %v)
// CHECK: [[EXTRACT_1:%.+]] = extractvalue { double, i32 } [[VAL]], 1
// CHECK: store i32 [[EXTRACT_1]], ptr addrspace(5)
// CHECK: [[EXTRACT_0:%.+]] = extractvalue { double, i32 } [[VAL]], 0
// CHECK: ret double [[EXTRACT_0]]
double test_builtin_frexp(double v, int* e) {
  return __builtin_frexp(v, e);
}
