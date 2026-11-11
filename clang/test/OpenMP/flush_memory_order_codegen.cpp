// Check that the memory-order clauses of 'flush' reach the host runtime and
// become fences on GPUs, and that the flushes implied by 'atomic' stay seq_cst.

// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefixes=DEV,AMDGPU
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefixes=DEV,AMDGPU
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host-nv.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-nv.bc -o - | FileCheck %s --check-prefixes=DEV,NVPTX
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host-spirv.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-spirv.bc -o - | FileCheck %s --check-prefixes=DEV,SPIRV
// expected-no-diagnostics

int x, v, e, d;

void foo() {
#pragma omp target
  {
#pragma omp flush
#pragma omp flush(x)
#pragma omp flush seq_cst
#pragma omp flush acq_rel
#pragma omp flush acquire
#pragma omp flush release
#pragma omp atomic read acquire
    v = x;
#pragma omp atomic write release
    x = v;
#pragma omp atomic compare seq_cst
    if (x == e)
      x = d;
  }
}

// The host keeps calling __kmpc_flush for a seq_cst flush, and passes any
// other ordering to __kmpc_flush_explicit as an __ATOMIC_* value, with the
// device scope as a __MEMORY_SCOPE_* value.
// HOST-LABEL: define {{.*}}@{{.*}}foo{{.*}}(
// HOST:      call void @__kmpc_flush(ptr
// HOST-NEXT: call void @__kmpc_flush(ptr
// HOST-NEXT: call void @__kmpc_flush(ptr
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 4, i32 1)
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 2, i32 1)
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 3, i32 1)
// HOST:      load atomic i32, ptr {{.*}} acquire
// HOST-NEXT: call void @__kmpc_flush(ptr
// HOST:      store atomic i32 {{.*}} release
// HOST-NEXT: call void @__kmpc_flush(ptr
// HOST:      cmpxchg ptr {{.*}} seq_cst seq_cst
// HOST:      call void @__kmpc_flush(ptr

// The device emits fences directly, at device scope.
// DEV-LABEL: define {{.*}}@__omp_offloading_{{.*}}foo{{.*}}(
// DEV-NOT:     call {{.*}}@__kmpc_flush
// AMDGPU:      fence syncscope("agent") seq_cst
// AMDGPU-NEXT: fence syncscope("agent") seq_cst
// AMDGPU-NEXT: fence syncscope("agent") seq_cst
// AMDGPU-NEXT: fence syncscope("agent") acq_rel
// AMDGPU-NEXT: fence syncscope("agent") acquire
// AMDGPU-NEXT: fence syncscope("agent") release
// AMDGPU:      load atomic i32, ptr {{.*}} acquire
// AMDGPU-NEXT: fence syncscope("agent") seq_cst
// AMDGPU:      store atomic i32 {{.*}} release
// AMDGPU-NEXT: fence syncscope("agent") seq_cst
// AMDGPU:      cmpxchg ptr {{.*}} seq_cst seq_cst
// AMDGPU:      fence syncscope("agent") seq_cst
// NVPTX:       fence syncscope("device") seq_cst
// NVPTX-NEXT:  fence syncscope("device") seq_cst
// NVPTX-NEXT:  fence syncscope("device") seq_cst
// NVPTX-NEXT:  fence syncscope("device") acq_rel
// NVPTX-NEXT:  fence syncscope("device") acquire
// NVPTX-NEXT:  fence syncscope("device") release
// NVPTX:       load atomic i32, ptr {{.*}} acquire
// NVPTX-NEXT:  fence syncscope("device") seq_cst
// NVPTX:       store atomic i32 {{.*}} release
// NVPTX-NEXT:  fence syncscope("device") seq_cst
// NVPTX:       cmpxchg ptr {{.*}} seq_cst seq_cst
// NVPTX:       fence syncscope("device") seq_cst
// SPIRV:       fence syncscope("device") seq_cst
// SPIRV-NEXT:  fence syncscope("device") seq_cst
// SPIRV-NEXT:  fence syncscope("device") seq_cst
// SPIRV-NEXT:  fence syncscope("device") acq_rel
// SPIRV-NEXT:  fence syncscope("device") acquire
// SPIRV-NEXT:  fence syncscope("device") release
// SPIRV:       load atomic i32, ptr {{.*}} acquire
// SPIRV-NEXT:  fence syncscope("device") seq_cst
// SPIRV:       store atomic i32 {{.*}} release
// SPIRV-NEXT:  fence syncscope("device") seq_cst
// SPIRV:       cmpxchg ptr {{.*}} seq_cst seq_cst
// SPIRV:       fence syncscope("device") seq_cst
// DEV-NOT:     call {{.*}}@__kmpc_flush
