// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100  %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=99 -DOMP99 -verify=expected,rev -ferror-limit 100  %s

void foo2() {
  int a;
  #pragma omp target // expected-note 3 {{'target' previously encountered here}} rev-note {{'target' previously encountered here}}
  {
    a = a + 1;
  }
}

#pragma omp requires atomic_default_mem_order(seq_cst)
#pragma omp requires unified_address //expected-error {{'target' region encountered before requires directive with 'unified_address' clause}}
#pragma omp requires unified_shared_memory //expected-error {{'target' region encountered before requires directive with 'unified_shared_memory' clause}}
#ifdef OMP99
#pragma omp requires reverse_offload //expected-error {{'target' region encountered before requires directive with 'reverse_offload' clause}}
#endif
#pragma omp requires dynamic_allocators //expected-error {{'target' region encountered before requires directive with 'dynamic_allocators' clause}}
