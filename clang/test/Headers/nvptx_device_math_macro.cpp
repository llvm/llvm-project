// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -x c++ -internal-isystem %S/Inputs/include -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -x c++ -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -fopenmp -triple nvptx64-nvidia-cuda -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

#include <cmath>

#pragma omp declare target
int use_macro() {
  double a(0);
  return (std::fpclassify(a) != FP_ZERO);
// CHECK: call i1 @llvm.is.fpclass.f64(double {{.*}}, i32 {{.*}})
// CHECK: call i1 @llvm.is.fpclass.f64(double {{.*}}, i32 {{.*}})
// CHECK: call i1 @llvm.is.fpclass.f64(double {{.*}}, i32 {{.*}})
// CHECK: call i1 @llvm.is.fpclass.f64(double {{.*}}, i32 {{.*}})
}
#pragma omp end declare target
