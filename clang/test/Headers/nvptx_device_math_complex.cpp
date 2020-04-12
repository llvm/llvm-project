// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -verify -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -aux-triple powerpc64le-unknown-unknown -o - | FileCheck %s
// expected-no-diagnostics

#include <complex>

// CHECK-DAG: define {{.*}} @__mulsc3
// CHECK-DAG: define {{.*}} @__muldc3
// CHECK-DAG: define {{.*}} @__divsc3
// CHECK-DAG: define {{.*}} @__divdc3

// CHECK-DAG: call float @__nv_scalbnf(
void test_scmplx(std::complex<float> a) {
#pragma omp target
  {
    (void)(a * (a / a));
  }
}

// CHECK-DAG: call double @__nv_scalbn(
void test_dcmplx(std::complex<double> a) {
#pragma omp target
  {
    (void)(a * (a / a));
  }
}
