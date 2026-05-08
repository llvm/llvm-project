// Test that target teams reduction codegen handles address space casts correctly.

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s

// expected-no-diagnostics

// Verify the kernel is generated.
// CHECK: define weak_odr protected spir_kernel void @__omp_offloading_{{.*}}_main_{{.*}}

// Verify __kmpc_alloc_shared is called for reduction variable.
// The return type should be ptr addrspace(4) (generic pointer).
// CHECK: call spir_func align 8 addrspace(9) ptr addrspace(4) @__kmpc_alloc_shared(i64 4)

// Verify the reduction runtime function is called.
// CHECK: call spir_func addrspace(9) i32 @__kmpc_nvptx_teams_reduce_nowait_v2(

// Verify __kmpc_free_shared is called.
// CHECK: call spir_func addrspace(9) void @__kmpc_free_shared(ptr addrspace(4)

// Verify the reduction function is generated.
// CHECK: define internal void @{{.*}}reduction{{.*}}func

int main() {
  int x = 0;

  #pragma omp target teams num_teams(2) reduction(+ : x)
  {
    x += 2;
  }

  return x;
}
