// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s
// CHECK: define{{.*}} amdgpu_kernel void @test_call_kernel(i32 addrspace(1)* nocapture noundef %out)
// CHECK: store i32 4, i32 addrspace(1)* %out, align 4

kernel void test_kernel(global int *out)
{
  out[0] = 4;
}

__kernel void test_call_kernel(__global int *out)
{
  test_kernel(out);
}
