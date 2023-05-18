// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

// CHECK: define{{.*}} amdgpu_kernel void @calling_conv_amdgpu_kernel()
kernel void calling_conv_amdgpu_kernel()
{
}

// CHECK: define{{.*}} void @calling_conv_none()
void calling_conv_none()
{
}
