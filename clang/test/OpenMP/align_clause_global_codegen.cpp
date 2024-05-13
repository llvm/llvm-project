// RUN: %clang_cc1 -emit-llvm -o - -fopenmp \
// RUN:  -triple i386-unknown-unknown %s \
// RUN:  | FileCheck %s --check-prefixes=CHECK,CHECK-32

// RUN: %clang_cc1 -emit-llvm -o - -fopenmp \
// RUN:  -triple x86_64-unknown-linux-gnu %s \
// RUN:  | FileCheck %s --check-prefixes=CHECK,CHECK-64

typedef enum omp_allocator_handle_t {
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
} omp_allocator_handle_t;

//
// Should allow larger alignment.
//

// CHECK: @foo_global1 = global float 0.000000e+00, align 16
float foo_global1;
#pragma omp allocate(foo_global1) align(16)

// CHECK: @foo_global2 = global float 0.000000e+00, align 16
float foo_global2;
#pragma omp allocate(foo_global2) allocator(omp_default_mem_alloc) align(16)

// CHECK: @foo_global3 = global float 0.000000e+00, align 16
float foo_global3;
#pragma omp allocate(foo_global3) allocator(omp_large_cap_mem_alloc) align(16)

// CHECK: @foop_global1 = global ptr null, align 16
int *foop_global1;
#pragma omp allocate(foop_global1) align(16)

//
// Should use natural alignment when alignment specified is too small.
//

// CHECK: @foo_global4 = global float 0.000000e+00, align 4
float foo_global4;
#pragma omp allocate(foo_global4) align(2)

// CHECK: @foo_global5 = global float 0.000000e+00, align 4
float foo_global5;
#pragma omp allocate(foo_global5) allocator(omp_default_mem_alloc) align(2)

// CHECK: @foo_global6 = global float 0.000000e+00, align 4
float foo_global6;
#pragma omp allocate(foo_global6) allocator(omp_large_cap_mem_alloc) align(2)

// CHECK-32: @foop_global2 = global ptr null, align 4
// CHECK-64: @foop_global2 = global ptr null, align 8
int *foop_global2;
#pragma omp allocate(foop_global2) align(2)
