// RUN: %clang -cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -S -I /sycl_include_path -I /opencl_include_path -I /usr/include/c++/4.8.5 -I /usr/include/c++/4.8.5/x86_64-redhat-linux -I /usr/include/c++/4.8.5/backward -I /include -I /usr/include -fcxx-exceptions -fexceptions -emit-llvm -x c++ %s -o - | FileCheck %s

// CHECK: define {{.*}}spir_kernel void @kernel_function() {{[^{]+}} !kernel_arg_addr_space ![[MD:[0-9]+]] !kernel_arg_access_qual ![[MD]] !kernel_arg_type ![[MD]] !kernel_arg_base_type ![[MD]] !kernel_arg_type_qual ![[MD]] {
// CHECK: ![[MD]] = !{}
// XFAIL: *

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
