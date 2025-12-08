// RUN: %clang_cc1 -verify -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -aux-triple x86_64-unknown-unknown -o - | FileCheck %s
// expected-no-diagnostics

#include <cmath>
#include <complex>

// CHECK: define weak {{.*}} @__muldc3
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 3)
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 516)

// CHECK: define weak {{.*}} @__mulsc3
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 3)
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 516)
// CHECK-DAG: call float @llvm.copysign.f32(

// CHECK: define weak {{.*}} @__divdc3
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 3)
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 504)
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 516)
// CHECK-DAG: call double @llvm.copysign.f64(
// CHECK-DAG: call double @llvm.ldexp.f64.i32(
// CHECK-DAG: call double @llvm.fabs.f64(
// CHECK-DAG: call { double, i32 } @llvm.frexp.f64.i32

// CHECK: define weak {{.*}} @__divsc3
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 3)
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 504)
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 516)
// CHECK-DAG: call float @llvm.copysign.f32(
// CHECK-DAG: call float @llvm.ldexp.f32.i32(
// CHECK-DAG: call float @llvm.fabs.f32(
// CHECK-DAG: call { float, i32 } @llvm.frexp.f32.i32

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
