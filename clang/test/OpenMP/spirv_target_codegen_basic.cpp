// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s

// expected-no-diagnostics

// CHECK: @__omp_offloading_{{.*}}_dynamic_environment = weak_odr protected addrspace(1) global %struct.DynamicEnvironmentTy zeroinitializer
// CHECK: @__omp_offloading_{{.*}}_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy

// CHECK: @"_gomp_critical_user_$var" = common global [8 x i32] zeroinitializer, align 8

// CHECK: define weak_odr protected spir_kernel void @__omp_offloading_{{.*}}

// CHECK: call spir_func addrspace(9) void @__kmpc_critical(ptr addrspace(4) addrspacecast (ptr addrspace(1) @{{.*}} to ptr addrspace(4)), i32 %{{.*}}, ptr addrspace(4) addrspacecast (ptr @"_gomp_critical_user_$var" to ptr addrspace(4)))
// CHECK: call spir_func addrspace(9) void @__kmpc_end_critical(ptr addrspace(4) addrspacecast (ptr addrspace(1) @{{.*}} to ptr addrspace(4)), i32 %{{.*}}, ptr addrspace(4) addrspacecast (ptr @"_gomp_critical_user_$var" to ptr addrspace(4)))

int main() {
  int ret = 0;
  #pragma omp target
  for(int i = 0; i < 5; i++)
    #pragma omp critical
    ret++;
  return ret;
}
