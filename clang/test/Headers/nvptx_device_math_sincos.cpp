// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers         -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -triple nvptx64-nvidia-cuda -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -DCMATH -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -triple nvptx64-nvidia-cuda -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s

#ifdef CMATH
#include <cmath>
#else
#include <math.h>
#endif

// 4 calls to sincos(f), all translated to __nv_sincos calls:

// CHECK-NOT: _Z.sincos
// CHECK: call void @__nv_sincos(double
// CHECK-NOT: _Z.sincos
// CHECK: call void @__nv_sincosf(float
// CHECK-NOT: _Z.sincos
// CHECK: call void @__nv_sincos(double
// CHECK-NOT: _Z.sincos
// CHECK: call void @__nv_sincosf(float
// CHECK-NOT: _Z.sincos

// single precision wrapper
inline void sincos(float x, float* __restrict__ sin, float* __restrict__ cos)
{
  sincosf(x, sin, cos);
}

template<typename T>
void test_sincos(T x)
{
  T res_sin, res_cos;

  #pragma omp target map(from: res_sin, res_cos)
  {
    sincos(x, &res_sin, &res_cos);
  }

}

int main(int argc, char **argv)
{

#if !defined(C_ONLY)
  test_sincos<double>(0.0);
  test_sincos<float>(0.0);
#endif

  #pragma omp target
  {
    double s, c;
    sincos(0, &s, &c);
  }

  #pragma omp target
  {
    float s, c;
    sincosf(0.f, &s, &c);
  }

  return 0;
}
