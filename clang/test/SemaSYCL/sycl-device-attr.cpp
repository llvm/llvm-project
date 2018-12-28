// RUN: %clang -S --sycl -I /sycl_include_path -I /opencl_include_path -Xclang -ast-dump %s | FileCheck %s
#include <CL/sycl.hpp>


using namespace cl::sycl;

int main() {

  queue myQueue;

  myQueue.submit([&](handler &cgh) {

    cgh.single_task<class kernel_function>([=]() {
    });
  });

  myQueue.wait();
}
// CHECK: SYCLDeviceAttr
