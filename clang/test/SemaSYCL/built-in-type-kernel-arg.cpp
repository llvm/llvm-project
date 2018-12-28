// RUN: %clang -S --sycl -Xclang -ast-dump %s | FileCheck %s
#include <CL/sycl.hpp>

int main() {
  int data = 5;
  cl::sycl::queue deviceQueue;
  cl::sycl::buffer<int, 1> bufferA(&data, cl::sycl::range<1>(1));

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task<class kernel_function>(
      [=]() {
        accessorA[0] += data;
      });
  });
  return 0;
}
// CHECK: kernel_function 'void (__global int *__global, int)
