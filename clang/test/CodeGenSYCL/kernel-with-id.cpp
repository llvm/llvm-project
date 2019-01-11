// RUN: %clang -cc1 -DCL_TARGET_OPENCL_VERSION=220 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -S -I /sycl_include_path -I /opencl_include_path -I /usr/include/c++/4.8.5 -I /usr/include/c++/4.8.5/x86_64-redhat-linux -I /usr/include/c++/4.8.5/backward -I /include -I /usr/include -fcxx-exceptions -fexceptions -emit-llvm -x c++ %s -o - | FileCheck %s

// XFAIL:*

#include <CL/sycl.hpp>
#include <array>



int main() {
  const size_t array_size = 1;
  std::array<cl::sycl::cl_int, array_size> A = {1};
  cl::sycl::queue deviceQueue;
  cl::sycl::range<1> numOfItems{array_size};
  cl::sycl::buffer<cl::sycl::cl_int, 1> bufferA(A.data(), numOfItems);

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<cl::sycl::access::mode::read_write>(cgh);
// CHECK: %wiID = alloca %"struct.cl::sycl::id", align 8
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0EE13__set_pointerEPU3AS1i(%"class.cl::sycl::accessor"* %1, i32 addrspace(1)* %2)
// CHECK: call spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_2idILm1EEEE_clES5_"(%class.anon* %0, %"struct.cl::sycl::id"* byval align 8 %wiID)
// CHECK: %call = call spir_func i64 @_Z13get_global_idj(i32 0)
    cgh.parallel_for<class kernel_function>(numOfItems,
      [=](cl::sycl::id<1> wiID) {
        accessorA[wiID] = accessorA[wiID] * accessorA[wiID];
      });
  });
  return 0;
}
