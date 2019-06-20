// RUN: %clang_cc1 -I %S/Inputs -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck %s

// This test checks that compiler generates correct kernel wrapper for basic
// case.

#include "sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorA;
    kernel<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}

// CHECK: define spir_kernel void @{{.*}}kernel_function(i32 addrspace(1)* [[MEM_ARG:%[a-zA-Z0-9_]+]], %"struct.{{.*}}.cl::sycl::range"* byval align 4 [[ACC_RANGE:%[a-zA-Z0-9_]+]], %"struct.{{.*}}.cl::sycl::range"* byval align 4 [[MEM_RANGE:%[a-zA-Z0-9_]+]], %"struct.{{.*}}.cl::sycl::id"* byval align 4 [[OFFSET:%[a-zA-Z0-9_]+]])
//
// Check alloca for pointer argument
// CHECK: [[MEM_ARG]].addr = alloca i32 addrspace(1)*
// Check lambda object alloca
// CHECK: [[ANON:%[0-9]+]] = alloca %"class.{{.*}}.anon"
// Check allocas for ranges
//
// Check store of kernel pointer argument to alloca
// CHECK: store i32 addrspace(1)* [[MEM_ARG]], i32 addrspace(1)** [[MEM_ARG]].addr, align 8

// Check accessor GEP
// CHECK: [[ACCESSOR:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[ANON]], i32 0, i32 0

// Check load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[MEM_ARG]].addr

// Check accessor __init method call
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor"* [[ACCESSOR]], i32 addrspace(1)* [[MEM_LOAD]], %"struct.{{.*}}.cl::sycl::range"* byval align 4 [[ACC_RANGE]], %"struct.{{.*}}.cl::sycl::range"* byval align 4 [[MEM_RANGE]], %"struct.{{.*}}.cl::sycl::id"* byval align 4 [[OFFSET]])

// Check lambda "()" operator call
// CHECK: call spir_func void @{{.*}}(%"class.{{.*}}.anon"* [[ANON]])
