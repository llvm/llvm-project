// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK

// expected-no-diagnostics

// CHECK-DAG: @__omp_offloading_[[KERNEL:.*]]_wg_size = weak addrspace(1) constant
// CHECK-DAG: @__omp_offloading_[[KERNEL]]_exec_mode = weak addrspace(1) constant
template <typename T>
class foo {
public:
  foo() {
    int a = 0;

    // CHECK: define weak amdgpu_kernel void @__omp_offloading_[[KERNEL]](
#pragma omp target
    {
      a += 1;
    }
  }
};


int main() {
  foo<float> local;
  return 0;
}
