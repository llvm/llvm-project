// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-device -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-is-device -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
//
// This test ensures that variables marked 'groupprivate' are emitted as
// device globals in the GPU shared address space (addrspace(3)).
// The test is GPU-only and checks the LLVM IR for addrspace(3).
//

int group_var;

#pragma omp groupprivate(group_var)

void foo() {
#pragma omp target teams num_teams(4) thread_limit(100)
{
  // simple use so the var is referenced in device codegen
  group_var = group_var + 1;
}
}

// CHECK: @group_var = global i32 0, align 4, addrspace(3)
// CHECK: store i32 %{{.*}}, i32 addrspace(3)* @group_var, align 4

// CHECK: @group_var = global i32 0, align 4, addrspace(3)
// CHECK: store i32 %{{.*}}, i32 addrspace(3)* @group_var, align 4
