// RUN: %clang_cc1 -triple=x86_64-linux-gnu -verify -fopenmp -x c -std=c99 %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -verify -fopenmp-simd -x c -std=c99 %s

[[omp::directive(assume no_openmp)]] // expected-error {{unexpected OpenMP directive '#pragma omp assume'}}

void foo(void) {
  [[omp::directive(assume hold(1==1))]] // expected-warning {{valid assume clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
  {}
}

void bar(void) {
  [[omp::directive(assume absent(target))]]
} // expected-error {{expected statement}}

void qux(void) {
  [[omp::directive(assume extra_bits)]] // expected-warning {{valid assume clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
  {}
}

void quux(void) {
  [[omp::directive(assume ext_spelled_properly)]]
  {}
}
