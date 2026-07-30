// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: llvm-dis < %t-ppc-host.bc | FileCheck %s -check-prefix=HOST

// device side declarations
#pragma omp declare target
extern "C" float cosf(float __x);
#pragma omp end declare target

// host side declaration
extern "C" float cosf(float __x);

void test_amdgcn_openmp_device(float __x) {
  // the default case where predefined library functions are treated as
  // builtins on the host
  // HOST: call float @llvm.cos.f32(float
  __x = cosf(__x);

#pragma omp target
  {
    // cosf should not be treated as builtin on device
    // CHECK-NOT: call float @llvm.cos.f32(float
    __x = cosf(__x);
  }
}
