// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-unknown %s
 // RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-unknown %s
 // RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64 %s

void foo() {
}

void bar() {
#pragma omp target ompx_bare // expected-error {{unexpected OpenMP clause 'ompx_bare' in directive '#pragma omp target'}} expected-note {{OpenMP extension clause 'ompx_bare' only allowed with '#pragma omp target teams'}}
  foo();

#pragma omp target teams distribute ompx_bare // expected-error {{unexpected OpenMP clause 'ompx_bare' in directive '#pragma omp target teams distribute'}} expected-note {{OpenMP extension clause 'ompx_bare' only allowed with '#pragma omp target teams'}}
  for (int i = 0; i < 10; ++i) {}

#pragma omp target teams distribute parallel for ompx_bare // expected-error {{unexpected OpenMP clause 'ompx_bare' in directive '#pragma omp target teams distribute parallel for'}} expected-note {{OpenMP extension clause 'ompx_bare' only allowed with '#pragma omp target teams'}}
  for (int i = 0; i < 10; ++i) {}

#pragma omp target
#pragma omp teams ompx_bare // expected-error {{unexpected OpenMP clause 'ompx_bare' in directive '#pragma omp teams'}} expected-note {{OpenMP extension clause 'ompx_bare' only allowed with '#pragma omp target teams'}}
  foo();
}
