// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50  -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

enum omp_allocator_handle_t {
  omp_null_allocator = 0,
  omp_default_mem_alloc = 1,
  omp_large_cap_mem_alloc = 2,
  omp_const_mem_alloc = 3,
  omp_high_bw_mem_alloc = 4,
  omp_low_lat_mem_alloc = 5,
  omp_cgroup_mem_alloc = 6,
  omp_pteam_mem_alloc = 7,
  omp_thread_mem_alloc = 8,
  KMP_ALLOCATOR_MAX_HANDLE = __UINTPTR_MAX__
};

// CHECK: define {{.*}}[[FIE:@.+]]()
void fie(void) {
  int x;
  #pragma omp target uses_allocators(omp_null_allocator) allocate(omp_null_allocator: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_default_mem_alloc) allocate(omp_default_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_large_cap_mem_alloc) allocate(omp_large_cap_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_const_mem_alloc) allocate(omp_const_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_high_bw_mem_alloc) allocate(omp_high_bw_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_low_lat_mem_alloc) allocate(omp_low_lat_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_cgroup_mem_alloc) allocate(omp_cgroup_mem_alloc: x) firstprivate(x)
  {}
  #pragma omp target uses_allocators(omp_pteam_mem_alloc) allocate(omp_pteam_mem_alloc: x) firstprivate(x)
  {}
}

#endif
