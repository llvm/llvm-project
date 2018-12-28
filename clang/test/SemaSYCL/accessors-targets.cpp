// RUN: %clang -S --sycl -Xclang -ast-dump %s | FileCheck %s
// XFAIL: *
#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {

  queue myQueue;
  const int size = 64;
  int data[size];
  buffer<int, 1> buf(data, range<1>(size));

  myQueue.submit([&](handler &cgh) {
    auto ptr = buf.get_access<access::mode::read_write>(cgh);

    accessor<int, 1, access::mode::read_write,
             access::target::local>
        tile(range<1>(2), cgh);
    cgh.single_task<class kernel_function>([=]() {
        tile[0] = 0;
        ptr[0] = 0;
    });
  });

  myQueue.wait();
}
// CHECK: kernel_function 'void (__local int *__local, __global int *__global)'
