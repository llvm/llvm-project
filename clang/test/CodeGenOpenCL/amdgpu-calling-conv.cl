// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple amdgpu-unknown-unknown -emit-llvm -o - %s | FileCheck %s %}

// CHECK: define{{.*}} amdgpu_kernel void @calling_conv_amdgpu_kernel()
kernel void calling_conv_amdgpu_kernel()
{
}

// CHECK: define{{.*}} void @calling_conv_none()
void calling_conv_none()
{
}
