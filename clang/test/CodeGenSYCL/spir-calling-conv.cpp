// RUN: %clang -cc1 -DCL_TARGET_OPENCL_VERSION=220 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -S -I /sycl_include_path -I /opencl_include_path -I /usr/include/c++/4.8.5 -I /usr/include/c++/4.8.5/x86_64-redhat-linux -I /usr/include/c++/4.8.5/backward -I /include -I /usr/include -fcxx-exceptions -fexceptions -emit-llvm -x c++ %s -o - | FileCheck %s

// XFAIL:*

#include <CL/sycl.hpp>


using namespace cl::sycl;

int main() {

  queue myQueue;

  myQueue.submit([&](handler &cgh) {

// CHECK: define spir_kernel void @kernel_function()

// CHECK: call spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%class.anon* %0)

// CHECK: define internal spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%class.anon* %this)

    cgh.single_task<class kernel_function>([=]() {
    });
  });

  myQueue.wait();
}
