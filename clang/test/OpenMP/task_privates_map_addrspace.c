// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-amdgcn-host.bc
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-amdgcn-host.bc -o - | FileCheck %s --check-prefix=AMDGCN
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-spirv-host.bc
// RUN: %clang_cc1 -fopenmp -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-spirv-host.bc -o - | FileCheck %s --check-prefix=SPIRV
// expected-no-diagnostics

// The privates mapping function of a task takes pointers in the default
// address space. Check that the temporaries passed to it, which are allocas,
// are cast to match: allocas are in address space 5 on AMDGPU, and in address
// space 0 on SPIR-V, whose default address space is the generic one (4).

void use(long);

void foo(void) {
#pragma omp target
  {
    long x = 5;
#pragma omp task firstprivate(x)
    use(x);
  }
}

// AMDGCN-LABEL: define internal void @"_omp_task_privates_map$"(
// AMDGCN-SAME: ptr noalias noundef %{{.*}}, ptr noalias noundef %{{.*}})

// AMDGCN-LABEL: define internal i32 @"_omp_task_entry$"(
// AMDGCN: [[FP:%.*firstpriv.*]] = alloca ptr, align 8, addrspace(5)
// AMDGCN: [[FP_CAST:%.*]] = addrspacecast ptr addrspace(5) [[FP]] to ptr
// AMDGCN: call void %{{.*}}(ptr %{{.*}}, ptr [[FP_CAST]])
// AMDGCN: load ptr, ptr addrspace(5) [[FP]]

// SPIRV-LABEL: define internal spir_func void @"_omp_task_privates_map$"(
// SPIRV-SAME: ptr addrspace(4) noalias noundef %{{.*}}, ptr addrspace(4) noalias noundef %{{.*}})

// SPIRV-LABEL: define internal spir_func i32 @"_omp_task_entry$"(
// SPIRV: [[FP:%.*firstpriv.*]] = alloca ptr addrspace(4), align 8
// SPIRV: [[FP_CAST:%.*]] = addrspacecast ptr [[FP]] to ptr addrspace(4)
// SPIRV: call spir_func addrspace(9) void %{{.*}}(ptr addrspace(4) %{{.*}}, ptr addrspace(4) [[FP_CAST]])
// SPIRV: load ptr addrspace(4), ptr [[FP]]
