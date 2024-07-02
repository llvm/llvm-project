// RUN: %clang_cc1 -triple=x86_64-linux-gnu -verify -fopenmp -x c -std=c99 %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -verify -fopenmp-simd -x c -std=c99 %s

#pragma omp assume no_openmp // expected-error {{unexpected OpenMP directive '#pragma omp assume'}}

void foo(void) {
  #pragma omp assume hold(1==1) // expected-warning {{valid assume clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
  {}
}

void bar(void) {
  #pragma omp assume absent(target)
} // expected-error {{expected statement}}

void qux(void) {
  #pragma omp assume extra_bits // expected-warning {{valid assume clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
  {}
}

void quux(void) {
  #pragma omp assume ext_spelled_properly
  {}
}
