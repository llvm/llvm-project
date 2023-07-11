// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -fopenmp -triple nvptx64-nvidia-cuda -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s

#include <cmath>

// 4 calls to modf(f), all translated to __nv_modf calls:

// CHECK-NOT: _Z.modf
// CHECK: call double @__nv_modf(double
// CHECK-NOT: _Z.modf
// CHECK: call float @__nv_modff(float
// CHECK-NOT: _Z.modf
// CHECK: call double @__nv_modf(double
// CHECK-NOT: _Z.modf
// CHECK: call float @__nv_modff(float
// CHECK-NOT: _Z.modf

template<typename T>
void test_modf(T x)
{
  T dx;
  int intx;

  #pragma omp target map(from: intx, dx)
  {
    T ipart;
    dx = std::modf(x, &ipart);
    intx = static_cast<int>(ipart);
  }
}

int main()
{

#if !defined(C_ONLY)
  test_modf<double>(1.0);
  test_modf<float>(1.0);
#endif

  #pragma omp target
  {
    double intpart, res;
    res = modf(1.1, &intpart);
  }

  #pragma omp target
  {
    float intpart, res;
    res = modff(1.1f, &intpart);
  }

}
