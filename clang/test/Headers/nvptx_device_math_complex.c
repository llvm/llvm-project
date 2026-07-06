// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -verify -internal-isystem %S/Inputs/include -fopenmp -x c -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -x c -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -aux-triple powerpc64le-unknown-unknown -o - | FileCheck %s
// RUN: %clang_cc1 -verify -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -aux-triple powerpc64le-unknown-unknown -o - | FileCheck %s
// expected-no-diagnostics

#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

// CHECK: define weak {{.*}} @__divsc3
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 3)
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 516)
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 504)
// CHECK-DAG: call float @llvm.copysign.f32(
// CHECK-DAG: call float @__nv_scalbnf(
// CHECK-DAG: call nsz float @llvm.fabs.f32(
// CHECK-DAG: call nsz float @llvm.fabs.f32(
// CHECK-DAG: call nsz float @llvm.maxnum.f32(
// CHECK-DAG: call float @__nv_logbf(

// CHECK: define weak {{.*}} @__mulsc3
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 3)
// CHECK-DAG: call i1 @llvm.is.fpclass.f32(float %{{.+}}, i32 516)
// CHECK-DAG: call float @llvm.copysign.f32(

// CHECK: define weak {{.*}} @__divdc3
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 3)
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 504)
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 516)
// CHECK-DAG: call double @llvm.copysign.f64(
// CHECK-DAG: call double @__nv_scalbn(
// CHECK-DAG: call nsz double @llvm.fabs.f64(
// CHECK-DAG: call nsz double @llvm.fabs.f64(
// CHECK-DAG: call nsz double @llvm.maxnum.f64(
// CHECK-DAG: call double @__nv_logb(

// CHECK: define weak {{.*}} @__muldc3
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 3)
// CHECK-DAG: call i1 @llvm.is.fpclass.f64(double %{{.+}}, i32 516)
// CHECK-DAG: call double @llvm.copysign.f64(

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
