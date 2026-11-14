// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

// omp_allocator_handle_t is declared but the predefined allocators are not, so
// every directive naming an allocator has to be diagnosed.

typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_const_mem_alloc;
extern const omp_allocator_handle_t omp_pteam_mem_alloc;

int a, b;
#pragma omp allocate(a) allocator(omp_const_mem_alloc) // expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}}
#pragma omp allocate(b) allocator(omp_pteam_mem_alloc) // expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}}

void foo(int *r) {
  int x = 0;
#pragma omp parallel private(x) allocate(omp_const_mem_alloc : x) // expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}}
  *r = x;
}
