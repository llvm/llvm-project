// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ | FileCheck %s
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ -std=c++14 | FileCheck %s


// expected-no-diagnostics

// CHECK-LABEL: amdgpu_kernel void @_Z4kernPff
__global__ void kern(float *x, float y) {
  *x = sin(y);
}

// CHECK-LABEL: define{{.*}} i64 @_Z11test_size_tv
// CHEC: ret i64 8
__device__ size_t test_size_t() {
  return sizeof(size_t);
}
