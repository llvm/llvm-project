// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -x c -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -x c -fopenmp -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefixes=CHECK

#include <complex.h>

void test_complex_f64(double _Complex a) {
// CHECK-LABEL: define {{.*}}test_complex_f64
#pragma omp target
  {
    // CHECK: call { double, double } @__divdc3
    // CHECK: call { double, double } @__muldc3
    (void)(a * (a / a));
  }
}

// CHECK: define weak {{.*}} @__divdc3
// CHECK-DAG: call double @__ocml_fabs_f64(
// CHECK-DAG: call i32 @__ocml_isnan_f64(
// CHECK-DAG: call i32 @__ocml_isfinite_f64(
// CHECK-DAG: call double @__ocml_copysign_f64(
// CHECK-DAG: call double @__ocml_scalbn_f64(
// CHECK-DAG: call double @__ocml_logb_f64(

// CHECK: define weak {{.*}} @__muldc3
// CHECK-DAG: call i32 @__ocml_isnan_f64(
// CHECK-DAG: call i32 @__ocml_isinf_f64(
// CHECK-DAG: call double @__ocml_copysign_f64(

void test_complex_f32(float _Complex a) {
// CHECK-LABEL: define {{.*}}test_complex_f32
#pragma omp target
  {
    // CHECK: call [2 x i32] @__divsc3
    // CHECK: call [2 x i32] @__mulsc3
    (void)(a * (a / a));
  }
}

// CHECK: define weak {{.*}} @__divsc3
// CHECK-DAG: call float @__ocml_fabs_f32(
// CHECK-DAG: call i32 @__ocml_isnan_f32(
// CHECK-DAG: call i32 @__ocml_isfinite_f32(
// CHECK-DAG: call float @__ocml_copysign_f32(
// CHECK-DAG: call float @__ocml_scalbn_f32(
// CHECK-DAG: call float @__ocml_logb_f32(

// CHECK: define weak {{.*}} @__mulsc3
// CHECK-DAG: call i32 @__ocml_isnan_f32(
// CHECK-DAG: call i32 @__ocml_isinf_f32(
// CHECK-DAG: call float @__ocml_copysign_f32(
