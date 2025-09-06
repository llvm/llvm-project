// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s

// expected-no-diagnostics

// CHECK: @[[#LOC:]] = private unnamed_addr addrspace(1) constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
// CHECK: = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{.*}}, ptr addrspacecast (ptr addrspace(1) @[[#LOC]] to ptr) }, align 8

int main() {
  int ret = 0;
  #pragma omp target
  for(int i = 0; i < 5; i++)
    ret++;
  return ret;
}
