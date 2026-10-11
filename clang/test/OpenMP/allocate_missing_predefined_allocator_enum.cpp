// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

// omp_allocator_handle_t is declared as an enum but the predefined allocators
// are not, so every directive naming an allocator has to be diagnosed. The
// allocator named here is a handle of the user's own and the directives are in
// a function body rather than at file scope.

typedef enum omp_allocator_handle_t {
  omp_default_mem_alloc = 1,
  __omp_allocator_handle_t_max__ = __UINTPTR_MAX__
} omp_allocator_handle_t;

void foo() {
  omp_allocator_handle_t my_handle;
  int A[2];
#pragma omp allocate(A) allocator(my_handle) // expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}}
#pragma omp allocate(A) allocator(my_handle) // expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}}
}
