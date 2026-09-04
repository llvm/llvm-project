// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fsyntax-only -verify %s

void test(void) {
#pragma omp taskloop transparent // expected-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}}
  for (int i = 0; i < 2; ++i)
    ;

#pragma omp parallel absent(target) // expected-error {{unexpected OpenMP clause 'absent' in directive '#pragma omp parallel'}}
  {}

#pragma omp parallel contains(target) // expected-error {{unexpected OpenMP clause 'contains' in directive '#pragma omp parallel'}}
  {}

#pragma omp parallel no_openmp // expected-error {{unexpected OpenMP clause 'no_openmp' in directive '#pragma omp parallel'}}
  {}

#pragma omp parallel no_openmp_routines // expected-error {{unexpected OpenMP clause 'no_openmp_routines' in directive '#pragma omp parallel'}}
  {}

#pragma omp parallel no_openmp_constructs // expected-error {{unexpected OpenMP clause 'no_openmp_constructs' in directive '#pragma omp parallel'}}
  {}

#pragma omp parallel no_parallelism // expected-error {{unexpected OpenMP clause 'no_parallelism' in directive '#pragma omp parallel'}}
  {}
}
