// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -x c -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -x c -fopenmp -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefixes=CHECK-C,CHECK
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -x c++ -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -x c++ -fopenmp -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefixes=CHECK-CPP,CHECK

#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

void test_math_f64(double x) {
// CHECK-LABEL: define {{.*}}test_math_f64
#pragma omp target
  {
    // CHECK: call double @__ocml_sin_f64
    double l1 = sin(x);
    // CHECK: call double @__ocml_cos_f64
    double l2 = cos(x);
    // CHECK: call double @__ocml_fabs_f64
    double l3 = fabs(x);
  }
}

void test_math_f32(float x) {
// CHECK-LABEL: define {{.*}}test_math_f32
#pragma omp target
  {
    // CHECK-C: call double @__ocml_sin_f64
    // CHECK-CPP: call float @__ocml_sin_f32
    float l1 = sin(x);
    // CHECK-C: call double @__ocml_cos_f64
    // CHECK-CPP: call float @__ocml_cos_f32
    float l2 = cos(x);
    // CHECK-C: call double @__ocml_fabs_f64
    // CHECK-CPP: call float @__ocml_fabs_f32
    float l3 = fabs(x);
  }
}
void test_math_f32_suffix(float x) {
// CHECK-LABEL: define {{.*}}test_math_f32_suffix
#pragma omp target
  {
    // CHECK: call float @__ocml_sin_f32
    float l1 = sinf(x);
    // CHECK: call float @__ocml_cos_f32
    float l2 = cosf(x);
    // CHECK: call float @__ocml_fabs_f32
    float l3 = fabsf(x);
  }
}
