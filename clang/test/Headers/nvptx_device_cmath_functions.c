// Test calling of device math functions.
///==========================================================================///

// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -triple nvptx64-nvidia-cuda -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck -check-prefix CHECK-YES %s

#include <stdlib.h>
#include <math.h>

void test_sqrt(double a1) {
  #pragma omp target
  {
    // CHECK-YES: call double @__nv_sqrt(double
    double l1 = sqrt(a1);
    // CHECK-YES: call double @__nv_pow(double
    double l2 = pow(a1, a1);
    // CHECK-YES: call double @__nv_modf(double
    double l3 = modf(a1 + 3.5, &a1);
    // CHECK-YES: call double @__nv_fabs(double
    double l4 = fabs(a1);
    // CHECK-YES: call i32 @__nv_abs(i32
    double l5 = abs((int)a1);
  }
}
