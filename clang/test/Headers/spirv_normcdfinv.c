// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel-unknown -emit-llvm-bc %s  -o %t-host.bc
// RUN: %clang_cc1 -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -disable-llvm-passes -fopenmp -triple spirv64 -fopenmp-targets=spirv64 -emit-llvm %s -fopenmp-is-target-device  -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -disable-llvm-passes -fopenmp -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device  -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// expected-no-diagnostics
#include <math.h>

// Test that normcdfinvf is properly defined and uses SPIRV OCL builtins
// CHECK-LABEL: define {{.*}} @{{.*}}test_normcdfinvf
void test_normcdfinvf(float x, float *result) {
  #pragma omp target map(from: result[0:1])
  {
    // CHECK: call {{.*}} float @{{.*}}normcdfinvf{{.*}}(float
    result[0] = normcdfinvf(x);
  }
}

// Test that normcdfinv is properly defined and uses SPIRV OCL builtins
// CHECK-LABEL: define {{.*}} @{{.*}}test_normcdfinv
void test_normcdfinv(double x, double *result) {
  #pragma omp target
  {
    // CHECK: call {{.*}} double @{{.*}}normcdfinv{{.*}}(double
    result[0] = normcdfinv(x);
  }
}

 
