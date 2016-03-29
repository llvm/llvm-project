// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -fcuda-is-device -Wno-cuda-compat -emit-llvm -o - | FileCheck %s

#include "Inputs/cuda.h"

extern "C"
void host_function() {}

// CHECK-LABEL: define void @hd_function_a
extern "C"
__host__ __device__ void hd_function_a() {
  // CHECK: call void @host_function
  host_function();
}

// CHECK: declare void @host_function

// CHECK-LABEL: define void @hd_function_b
extern "C"
__host__ __device__ void hd_function_b(bool b) { if (b) host_function(); }

// CHECK-LABEL: define void @device_function_b
extern "C"
__device__ void device_function_b() { hd_function_b(false); }

// CHECK-LABEL: define void @global_function
extern "C"
__global__ void global_function() {
  // CHECK: call void @device_function_b
  device_function_b();
}

// CHECK: !{{[0-9]+}} = !{void ()* @global_function, !"kernel", i32 1}
