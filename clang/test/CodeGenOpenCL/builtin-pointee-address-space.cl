// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -Wno-deprecated-attributes -emit-llvm -o - %s | FileCheck %s

int global_as(__global int *p) {
  return __builtin_pointee_address_space(p);
}

// CHECK-LABEL: define{{.*}} i32 @global_as(
// CHECK-SAME: ptr addrspace(1)
// CHECK: ret i32 1

int local_as(__local int *p) {
  return __builtin_pointee_address_space(p);
}

// CHECK-LABEL: define{{.*}} i32 @local_as(
// CHECK-SAME: ptr addrspace(3)
// CHECK: ret i32 2

int global_device_as(__attribute__((opencl_global_device)) int *p) {
  return __builtin_pointee_address_space(p);
}

// CHECK-LABEL: define{{.*}} i32 @global_device_as(
// CHECK-SAME: ptr addrspace(1)
// CHECK: ret i32 1

int global_host_as(__attribute__((opencl_global_host)) int *p) {
  return __builtin_pointee_address_space(p);
}

// CHECK-LABEL: define{{.*}} i32 @global_host_as(
// CHECK-SAME: ptr addrspace(1)
// CHECK: ret i32 1
