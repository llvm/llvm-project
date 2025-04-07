// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=spirv64 -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -O0 -fopenmp -fopenmp-targets=spirv64 -fopenmp-is-target-device -triple spirv64 -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s

extern int printf(char[]);

#pragma omp declare target
// CHECK: @global = addrspace(1) global i32 0, align 4
// CHECK: @.str = private unnamed_addr addrspace(1) constant [4 x i8] c"foo\00", align 1
int global = 0;
#pragma omp end declare target
int main() {
  // CHECK: = call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @__omp_offloading_{{.*}}_kernel_environment to ptr), ptr %{{.*}})
#pragma omp target
  {
    for(int i = 0; i < 5; i++)
      global++;
    printf("foo");
  }
  return global;
}
