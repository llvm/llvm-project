// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -x c++ -internal-isystem %S/Inputs/include -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -x c++ -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -fopenmp -triple nvptx64-nvidia-cuda -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=SLOW
// RUN: %clang_cc1 -x c++ -internal-isystem %S/Inputs/include -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc -ffast-math -ffp-contract=fast
// RUN: %clang_cc1 -x c++ -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -fopenmp -triple nvptx64-nvidia-cuda -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -ffast-math -ffp-contract=fast | FileCheck %s --check-prefix=FAST
// expected-no-diagnostics

#include <cmath>

double math(float f, double d) {
  double r = 0;
// SLOW:  call float @__nv_sinf(float
// FAST:  call fast nofpclass(nan inf) float @__nv_fast_sinf(float
  r += sin(f);
// SLOW:  call double @__nv_sin(double
// FAST:  call fast nofpclass(nan inf) double @__nv_sin(double
  r += sin(d);
  return r;
}

long double foo(float f, double d, long double ld) {
  double r = ld;
  r += math(f, d);
#pragma omp target map(r)
  { r += math(f, d); }
  return r;
}
