// RUN: %clang_cc1 -no-opaque-pointers %s -cl-std=cl2.0 -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -cl-std=cl1.2 -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -cl-std=cl1.1 -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64:enable

// CHECK-LABEL: @test_store_float(float noundef %foo, half addrspace({{.}}){{.*}} %bar)
__kernel void test_store_float(float foo, __global half* bar)
{
	__builtin_store_halff(foo, bar);
// CHECK: [[HALF_VAL:%.*]] = fptrunc float %foo to half
// CHECK: store half [[HALF_VAL]], half addrspace({{.}})* %bar, align 2
}

// CHECK-LABEL: @test_store_double(double noundef %foo, half addrspace({{.}}){{.*}} %bar)
__kernel void test_store_double(double foo, __global half* bar)
{
	__builtin_store_half(foo, bar);
// CHECK: [[HALF_VAL:%.*]] = fptrunc double %foo to half
// CHECK: store half [[HALF_VAL]], half addrspace({{.}})* %bar, align 2
}

// CHECK-LABEL: @test_load_float(float addrspace({{.}}){{.*}} %foo, half addrspace({{.}}){{.*}} %bar)
__kernel void test_load_float(__global float* foo, __global half* bar)
{
	foo[0] = __builtin_load_halff(bar);
// CHECK: [[HALF_VAL:%.*]] = load half, half addrspace({{.}})* %bar
// CHECK: [[FULL_VAL:%.*]] = fpext half [[HALF_VAL]] to float
// CHECK: store float [[FULL_VAL]], float addrspace({{.}})* %foo
}

// CHECK-LABEL: @test_load_double(double addrspace({{.}}){{.*}} %foo, half addrspace({{.}}){{.*}} %bar)
__kernel void test_load_double(__global double* foo, __global half* bar)
{
	foo[0] = __builtin_load_half(bar);
// CHECK: [[HALF_VAL:%.*]] = load half, half addrspace({{.}})* %bar
// CHECK: [[FULL_VAL:%.*]] = fpext half [[HALF_VAL]] to double
// CHECK: store double [[FULL_VAL]], double addrspace({{.}})* %foo
}
