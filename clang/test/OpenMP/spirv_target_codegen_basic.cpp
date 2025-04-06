// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s

// expected-no-diagnostics

// CHECK: @__omp_offloading_{{.*}}_dynamic_environment = weak_odr protected addrspace(1) global %struct.DynamicEnvironmentTy zeroinitializer
// CHECK: @__omp_offloading_{{.*}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy

// CHECK: define weak_odr protected spir_kernel void @__omp_offloading_{{.*}}

int main() {
  int ret = 0;
  #pragma omp target
  for(int i = 0; i < 5; i++)
    ret++;
  return ret;
}
