// RUN: %clang_cc1 -I %S/Inputs -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorA;
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor"* %{{.*}}, i32 addrspace(1)* %{{.*}}, %"struct.{{.*}}.cl::sycl::range"* byval align 1 %{{.*}}, %"struct.{{.*}}.cl::sycl::id"* byval align 1 %{{.*}})
    kernel<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}
