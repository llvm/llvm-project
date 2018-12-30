// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -I %S/Inputs -I %S/../Headers/Inputs/include/ -fsycl-is-device -ast-dump %s | FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 -triple spir-unknown-linux-sycldevice -I %S/Inputs -I %S/../Headers/Inputs/include/ -fsycl-is-device -ast-dump %s | FileCheck %s --check-prefix=CHECK-32
#include <sycl.hpp>
#include <stdlib.h>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

template <typename T>
class SimpleVadd;

int main() {
  kernel<class SimpleVadd<int>>(
      [=](){});

  kernel<class SimpleVadd<double>>(
      [=](){});

  kernel<class SimpleVadd<size_t>>(
      [=](){});
  return 0;
}

// CHECK: _ZTS10SimpleVaddIiE
// CHECK: _ZTS10SimpleVaddIdE
// CHECK-64: _ZTS10SimpleVaddImE
// CHECK-32: _ZTS10SimpleVaddIjE
