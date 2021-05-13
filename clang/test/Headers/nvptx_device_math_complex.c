// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -verify -internal-isystem %S/Inputs/include -fopenmp -x c -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -x c -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -aux-triple powerpc64le-unknown-unknown -o - | FileCheck %s
// RUN: %clang_cc1 -verify -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -aux-triple powerpc64le-unknown-unknown -o - | FileCheck %s
// expected-no-diagnostics

#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

// CHECK: define weak {{.*}} @__divsc3
// CHECK-DAG: call i32 @__nv_isnanf(
// CHECK-DAG: call i32 @__nv_isinff(
// CHECK-DAG: call i32 @__nv_finitef(
// CHECK-DAG: call float @__nv_copysignf(
// CHECK-DAG: call float @__nv_scalbnf(
// CHECK-DAG: call float @__nv_fabsf(
// CHECK-DAG: call float @__nv_logbf(

// CHECK: define weak {{.*}} @__mulsc3
// CHECK-DAG: call i32 @__nv_isnanf(
// CHECK-DAG: call i32 @__nv_isinff(
// CHECK-DAG: call float @__nv_copysignf(

// CHECK: define weak {{.*}} @__divdc3
// CHECK-DAG: call i32 @__nv_isnand(
// CHECK-DAG: call i32 @__nv_isinfd(
// CHECK-DAG: call i32 @__nv_isfinited(
// CHECK-DAG: call double @__nv_copysign(
// CHECK-DAG: call double @__nv_scalbn(
// CHECK-DAG: call double @__nv_fabs(
// CHECK-DAG: call double @__nv_logb(

// CHECK: define weak {{.*}} @__muldc3
// CHECK-DAG: call i32 @__nv_isnand(
// CHECK-DAG: call i32 @__nv_isinfd(
// CHECK-DAG: call double @__nv_copysign(

void test_scmplx(float _Complex a) {
#pragma omp target
  {
    (void)(a * (a / a));
  }
}

void test_dcmplx(double _Complex a) {
#pragma omp target
  {
    (void)(a * (a / a));
  }
}
