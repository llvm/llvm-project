// RUN: %clang_cc1 -fopenmp -verify %s

typedef enum omp_allocator_handle_t {
  omp_default_mem_alloc = 1,
  __omp_allocator_handle_t_max__ = __UINTPTR_MAX__
} omp_allocator_handle_t;

void foo(void) {
  omp_allocator_handle_t my_handle;
  int A[2];
  // expected-error@+2 {{'omp_allocator_handle_t' type not found; include <omp.h>}}
  // expected-note@+1 {{previous allocator is specified here}}
  #pragma omp allocate(A) allocator(my_handle)
  // expected-warning@+1 {{allocate directive specifies 'my_handle' allocator while previously used default}}
  #pragma omp allocate(A) allocator(my_handle)
}
