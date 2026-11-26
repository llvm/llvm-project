// Check that the memory-order clauses of 'flush' reach the host runtime and
// become fences on GPUs.

// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST50
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefix=DEV -DSCOPE=agent
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefix=DEV -DSCOPE=agent
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host-nv.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-nv.bc -o - | FileCheck %s --check-prefix=DEV -DSCOPE=device
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host-spirv.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-spirv.bc -o - | FileCheck %s --check-prefix=DEV -DSCOPE=device
// expected-no-diagnostics

int x;

void foo() {
#pragma omp target
  {
#pragma omp flush
#pragma omp flush(x)
#if _OPENMP >= 202011
#pragma omp flush seq_cst
#endif
#if _OPENMP >= 201811
#pragma omp flush acq_rel
#pragma omp flush acquire
#pragma omp flush release
#endif
  }
}

// The host keeps calling __kmpc_flush for a seq_cst flush, and passes any
// other ordering to __kmpc_flush_explicit as an __ATOMIC_* value, with the
// device scope as a __MEMORY_SCOPE_* value.
// HOST-LABEL: define internal void @__omp_offloading_{{.*}}foo{{.*}}(
// HOST:      call void @__kmpc_flush(ptr
// HOST-NEXT: call void @__kmpc_flush(ptr
// HOST-NEXT: call void @__kmpc_flush(ptr
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 4, i32 1)
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 2, i32 1)
// HOST-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 3, i32 1)

// OpenMP 5.0 gives a flush with no list and no memory-order clause acq_rel
// semantics; from 5.1 it is seq_cst.
// HOST50-LABEL: define internal void @__omp_offloading_{{.*}}foo{{.*}}(
// HOST50:      call void @__kmpc_flush_explicit(ptr {{.*}}, i32 4, i32 1)
// HOST50-NEXT: call void @__kmpc_flush(ptr
// HOST50-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 4, i32 1)
// HOST50-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 2, i32 1)
// HOST50-NEXT: call void @__kmpc_flush_explicit(ptr {{.*}}, i32 3, i32 1)

// Before OpenMP 5.0 there are no memory-order clauses, and a flush with no list
// is a full flush, as from 5.1.
// HOST45-LABEL: define internal void @__omp_offloading_{{.*}}foo{{.*}}(
// HOST45:      call void @__kmpc_flush(ptr
// HOST45-NEXT: call void @__kmpc_flush(ptr
// HOST45-NOT:  call void @__kmpc_flush

// The device emits fences directly, at device scope.
// DEV-LABEL: define {{.*}}@__omp_offloading_{{.*}}foo{{.*}}(
// DEV-NOT:  call {{.*}}@__kmpc_flush
// DEV:      fence syncscope("[[SCOPE]]") seq_cst
// DEV-NEXT: fence syncscope("[[SCOPE]]") seq_cst
// DEV-NEXT: fence syncscope("[[SCOPE]]") seq_cst
// DEV-NEXT: fence syncscope("[[SCOPE]]") acq_rel
// DEV-NEXT: fence syncscope("[[SCOPE]]") acquire
// DEV-NEXT: fence syncscope("[[SCOPE]]") release
// DEV-NOT:  call {{.*}}@__kmpc_flush
