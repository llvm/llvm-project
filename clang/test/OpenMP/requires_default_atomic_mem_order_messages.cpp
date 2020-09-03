// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100  %s

void foo2() {
  int a;
  #pragma omp atomic update // expected-note 3 {{'atomic' previously encountered here}}
    a = a + 1;
}

#pragma omp requires atomic_default_mem_order(seq_cst) // expected-error {{'atomic' region encountered before requires directive with 'atomic_default_mem_order' clause}} expected-note 2 {{atomic_default_mem_order clause previously used here}}
#pragma omp requires atomic_default_mem_order(acq_rel) // expected-error {{'atomic' region encountered before requires directive with 'atomic_default_mem_order' clause}} expected-error {{Only one atomic_default_mem_order clause can appear on a requires directive in a single translation unit}}
#pragma omp requires atomic_default_mem_order(relaxed) // expected-error {{'atomic' region encountered before requires directive with 'atomic_default_mem_order' clause}} expected-error {{Only one atomic_default_mem_order clause can appear on a requires directive in a single translation unit}}
#pragma omp requires atomic_default_mem_order(release) // expected-error {{expected 'seq_cst', 'acq_rel' or 'relaxed' in OpenMP clause 'atomic_default_mem_order'}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}
