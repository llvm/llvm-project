// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -I %S/Inputs  -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | FileCheck --enable-var-scope %s
// CHECK: define spir_kernel void @{{[a-zA-Z0-9_]+}}(%opencl.sampler_t addrspace(2)* [[SAMPLER_ARG:%[a-zA-Z0-9_]+]])
// CHECK-NEXT: entry:
// CHECK-NEXT: [[SAMPLER_ARG]].addr = alloca %opencl.sampler_t addrspace(2)*, align 8
// CHECK-NEXT: [[ANON:%[0-9]+]] = alloca %"class.{{.*}}.anon", align 8
// CHECK-NEXT: store %opencl.sampler_t addrspace(2)* [[SAMPLER_ARG]], %opencl.sampler_t addrspace(2)** [[SAMPLER_ARG]].addr, align 8, !tbaa !9
// CHECK-NEXT: [[BITCAST:%[0-9]+]] = bitcast %"class.{{.*}}.anon"* [[ANON]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[BITCAST]]) #4
// CHECK-NEXT: [[GEP:%[0-9]+]]  = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[ANON]], i32 0, i32 0
// CHECK-NEXT: [[LOAD_SAMPLER_ARG:%[0-9]+]] = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** [[SAMPLER_ARG]].addr, align 8, !tbaa !9
// CHECK-NEXT: call spir_func void @{{[a-zA-Z0-9_]+}}(%"class.{{.*}}.cl::sycl::sampler"* [[GEP]], %opencl.sampler_t addrspace(2)* [[LOAD_SAMPLER_ARG]])
//
#include "sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::sampler smplr;
  kernel_single_task<class first_kernel>([=]() {
    smplr.use();
  });

  return 0;
}
