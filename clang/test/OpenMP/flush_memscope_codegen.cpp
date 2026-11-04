// Check that the OpenMP 6.0 memscope clause sets the scope of a flush and of
// the flushes implied by 'atomic': a __MEMORY_SCOPE_* value passed to
// __kmpc_flush_explicit on the host, and the sync scope of the fence on GPUs.

// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefixes=DEV,AMDGPU
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -fopenmp-enable-irbuilder -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefixes=DEV,AMDGPU
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host-nv.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-nv.bc -o - | FileCheck %s --check-prefixes=DEV,NVPTX
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host-spirv.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-spirv.bc -o - | FileCheck %s --check-prefixes=DEV,SPIRV
// expected-no-diagnostics

int x, v, e, d;

void foo() {
#pragma omp target
  {
#pragma omp flush memscope(device)
#pragma omp flush memscope(all)
#pragma omp flush(x) memscope(all)
#pragma omp flush release memscope(all)
#pragma omp flush acquire memscope(cgroup)
#pragma omp flush acq_rel memscope(device)
#pragma omp atomic read acquire memscope(cgroup)
    v = x;
#pragma omp atomic write release memscope(all)
    x = v;
#pragma omp atomic update seq_cst memscope(cgroup)
    x += v;
#pragma omp atomic compare seq_cst memscope(all)
    if (x == e)
      x = d;
  }
}

// memscope(device) is the default, so a seq_cst flush with it still calls
// __kmpc_flush. Scopes: 0 = system (all), 1 = device, 2 = workgroup (cgroup).
// HOST-LABEL: define {{.*}}@{{.*}}foo{{.*}}(
// HOST:      call void @__kmpc_flush(ptr
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 5, i32 0)
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 5, i32 0)
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 3, i32 0)
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 2, i32 2)
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 4, i32 1)
// HOST:      load atomic i32, ptr {{.*}} acquire
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 5, i32 2)
// HOST:      store atomic i32 {{.*}} release
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 5, i32 0)
// HOST:      atomicrmw add ptr {{.*}} seq_cst
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 5, i32 2)
// HOST:      cmpxchg ptr {{.*}} seq_cst seq_cst
// HOST:      call void @__kmpc_flush_explicit(ptr {{.*}}, i32 5, i32 0)

// DEV-LABEL: define {{.*}}@__omp_offloading_{{.*}}foo{{.*}}(
// DEV-NOT:     call {{.*}}@__kmpc_flush
// AMDGPU:      fence syncscope("agent") seq_cst
// AMDGPU-NEXT: fence seq_cst
// AMDGPU-NEXT: fence seq_cst
// AMDGPU-NEXT: fence release
// AMDGPU-NEXT: fence syncscope("workgroup") acquire
// AMDGPU-NEXT: fence syncscope("agent") acq_rel
// AMDGPU:      load atomic i32, ptr {{.*}} acquire
// AMDGPU-NEXT: fence syncscope("workgroup") seq_cst
// AMDGPU:      store atomic i32 {{.*}} release
// AMDGPU-NEXT: fence seq_cst
// AMDGPU:      atomicrmw add ptr {{.*}} seq_cst
// AMDGPU-NEXT: fence syncscope("workgroup") seq_cst
// AMDGPU:      cmpxchg ptr {{.*}} seq_cst seq_cst
// AMDGPU:      fence seq_cst
// NVPTX:       fence syncscope("device") seq_cst
// NVPTX-NEXT:  fence seq_cst
// NVPTX-NEXT:  fence seq_cst
// NVPTX-NEXT:  fence release
// NVPTX-NEXT:  fence syncscope("block") acquire
// NVPTX-NEXT:  fence syncscope("device") acq_rel
// NVPTX:       load atomic i32, ptr {{.*}} acquire
// NVPTX-NEXT:  fence syncscope("block") seq_cst
// NVPTX:       store atomic i32 {{.*}} release
// NVPTX-NEXT:  fence seq_cst
// NVPTX:       atomicrmw add ptr {{.*}} seq_cst
// NVPTX-NEXT:  fence syncscope("block") seq_cst
// NVPTX:       cmpxchg ptr {{.*}} seq_cst seq_cst
// NVPTX:       fence seq_cst
// SPIRV:       fence syncscope("device") seq_cst
// SPIRV-NEXT:  fence seq_cst
// SPIRV-NEXT:  fence seq_cst
// SPIRV-NEXT:  fence release
// SPIRV-NEXT:  fence syncscope("workgroup") acquire
// SPIRV-NEXT:  fence syncscope("device") acq_rel
// SPIRV:       load atomic i32, ptr {{.*}} acquire
// SPIRV-NEXT:  fence syncscope("workgroup") seq_cst
// SPIRV:       store atomic i32 {{.*}} release
// SPIRV-NEXT:  fence seq_cst
// SPIRV:       atomicrmw add ptr {{.*}} seq_cst
// SPIRV-NEXT:  fence syncscope("workgroup") seq_cst
// SPIRV:       cmpxchg ptr {{.*}} seq_cst seq_cst
// SPIRV:       fence seq_cst
// DEV-NOT:     call {{.*}}@__kmpc_flush
