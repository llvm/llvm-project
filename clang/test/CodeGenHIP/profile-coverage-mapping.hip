// REQUIRES: amdgpu-registered-target

// Device-side compilation must emit coverage mapping sections and call the
// GPU profile instrumentation runtime.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1100 \
// RUN:   -fcuda-is-device -emit-llvm -o - %s \
// RUN:   -fprofile-instrument=clang -fcoverage-mapping | FileCheck %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

// CHECK-DAG: section "__llvm_covfun"
// CHECK-DAG: @__llvm_coverage_mapping = {{.*}}section "__llvm_covmap"
// CHECK-DAG: @__profc__Z3dfni = {{.*}}section "__llvm_prf_cnts"
// CHECK-DAG: @__profc__Z6kerneli = {{.*}}section "__llvm_prf_cnts"
// CHECK-DAG: @__llvm_prf_nm = {{.*}}section "__llvm_prf_names"
// CHECK: call void @__llvm_profile_instrument_gpu

__device__ int dfn(int i) { return i ? i + 1 : 0; }

__global__ void kernel(int i) { dfn(i); }
