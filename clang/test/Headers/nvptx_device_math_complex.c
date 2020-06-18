// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

// CHECK-DAG: call { float, float } @__divsc3(
// CHECK-DAG: call { float, float } @__mulsc3(
void test_scmplx(float _Complex a) {
#pragma omp target
  {
    (void)(a * (a / a));
  }
}


// CHECK-DAG: call { double, double } @__divdc3(
// CHECK-DAG: call { double, double } @__muldc3(
void test_dcmplx(double _Complex a) {
#pragma omp target
  {
    (void)(a * (a / a));
  }
}
