// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -triple nvptx64-nvidia-cuda -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s

#include <cmath>

// 6 calls to sin/cos(f), all translated to __nv_sin/__nv_cos calls:

// CHECK-NOT: _Z.sin
// CHECK-NOT: _Z.cos
// CHECK: call double @__nv_sin(double
// CHECK-NOT: _Z.sin
// CHECK-NOT: _Z.cos
// CHECK: call float @__nv_sinf(float
// CHECK-NOT: _Z.sin
// CHECK-NOT: _Z.cos
// CHECK: call double @__nv_sin(double
// CHECK-NOT: _Z.sin
// CHECK-NOT: _Z.cos
// CHECK: call double @__nv_cos(double
// CHECK-NOT: _Z.sin
// CHECK-NOT: _Z.cos
// CHECK: call float @__nv_sinf(float
// CHECK-NOT: _Z.sin
// CHECK-NOT: _Z.cos
// CHECK: call float @__nv_cosf(float
// CHECK-NOT: _Z.sin
// CHECK-NOT: _Z.cos

template<typename T>
void test_sin_cos(T x)
{
  T res_sin, res_cos;

  #pragma omp target map(from: res_sin, res_cos)
  {
    res_sin = std::sin(x);
    res_cos = std::cos(x);
  }
}

int main()
{

#if !defined(C_ONLY)
  test_sin_cos<double>(0.0);
  test_sin_cos<float>(0.0);
#endif

  #pragma omp target
  {
    double res;
    res = sin(1.0);
  }

  #pragma omp target
  {
    float res;
    res = sinf(1.0f);
  }

  return 0;
}
