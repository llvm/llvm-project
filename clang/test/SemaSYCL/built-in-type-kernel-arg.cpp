// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

#include <sycl.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> acc1;
  kernel<class kernel_function>(
      [=]() {
        acc1.use();
      });
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
    cl::sycl::access::target::local> tile;
  kernel<class kernel_local_acc>(
      [=]() {
        tile.use();
      });
  return 0;
}
// CHECK: kernel_function 'void (__global int *, range<1>, id<1>)
// CHECK: kernel_local_acc 'void (__local int *, range<1>, id<1>)
