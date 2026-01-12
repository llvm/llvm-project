// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc -DTEAMS
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -DTEAMS -o - | FileCheck %s
// expected-no-diagnostics

// CHECK: define weak_odr protected spir_kernel void @__omp_offloading_{{.*}}(ptr addrspace(1) noalias noundef %{{.*}}, ptr addrspace(1) noundef align 4 dereferenceable(128) %{{.*}}) 

int main() {
  int x[32] = {0};

#ifdef TEAMS
#pragma omp target teams
#else
#pragma omp target
#endif
  for(int i = 0; i < 32; i++) {
    if(i > 0)
      x[i] = x[i-1] + i;
  }

return x[31];
}

