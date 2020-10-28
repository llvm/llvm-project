// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -verify -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -aux-triple powerpc64le-unknown-unknown -o - | FileCheck %s
// expected-no-diagnostics

#include <cmath>
#include <complex>

// CHECK: define weak {{.*}} @__muldc3
// CHECK-DAG: call i32 @__nv_isnand(
// CHECK-DAG: call i32 @__nv_isinfd(
// CHECK-DAG: call double @__nv_copysign(

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

// CHECK: define weak {{.*}} @__divsc3
// CHECK-DAG: call i32 @__nv_isnanf(
// CHECK-DAG: call i32 @__nv_isinff(
// CHECK-DAG: call i32 @__nv_finitef(
// CHECK-DAG: call float @__nv_copysignf(
// CHECK-DAG: call float @__nv_scalbnf(
// CHECK-DAG: call float @__nv_fabsf(
// CHECK-DAG: call float @__nv_logbf(

// We actually check that there are no declarations of non-OpenMP functions.
// That is, as long as we don't call an unkown function with a name that
// doesn't start with '__' we are good :)

// CHECK-NOT: declare.*@[^_]

void test_scmplx(std::complex<float> a) {
#pragma omp target
  {
    (void)(a * (a / a));
  }
}

void test_dcmplx(std::complex<double> a) {
#pragma omp target
  {
    (void)(a * (a / a));
  }
}

template <typename T>
std::complex<T> test_template_math_calls(std::complex<T> a) {
  decltype(a) r = a;
#pragma omp target
  {
    r = std::sin(r);
    r = std::cos(r);
    r = std::exp(r);
    r = std::atan(r);
    r = std::acos(r);
  }
  return r;
}

std::complex<float> test_scall(std::complex<float> a) {
  decltype(a) r;
#pragma omp target
  {
    r = std::sin(a);
  }
  return test_template_math_calls(r);
}

std::complex<double> test_dcall(std::complex<double> a) {
  decltype(a) r;
#pragma omp target
  {
    r = std::exp(a);
  }
  return test_template_math_calls(r);
}
