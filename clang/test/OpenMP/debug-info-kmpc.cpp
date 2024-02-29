// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device "-debug-info-kind=constructor" -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s

// Check that we properly attach debug info to the __kmpc_global_thread_num call
// CHECK: call {{.*}} @__kmpc_global_thread_num{{.*}}!dbg

extern int bar();
void foo() {
#pragma omp target teams
  {
#pragma omp parallel
    {
      bar();
    }
  }
}

