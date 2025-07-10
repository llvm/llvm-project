// RUN: %clang_cc1 -triple=x86_64-linux-gnu -verify -fopenmp -x c -std=c99 %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -verify -fopenmp-simd -x c -std=c99 %s

[[omp::directive(assume no_openmp)]] // expected-error {{unexpected OpenMP directive '#pragma omp assume'}}

void foo(void) {
  [[omp::directive(assume hold(1==1))]] // expected-warning {{extra tokens at the end of '#pragma omp assume' are ignored}}
  {}
}

void bar(void) {
  [[omp::directive(assume absent(target))]]
} // expected-error {{expected statement}}

void qux(void) {
  [[omp::directive(assume extra_bits)]] // expected-warning {{extra tokens at the end of '#pragma omp assume' are ignored}}
  {}
}

void quux(void) {
  // This form of spelling for assumption clauses is supported for
  // "omp assumes" (as a non-standard extension), but not here.
  [[omp::directive(assume ext_spelled_like_this)]] // expected-warning {{extra tokens at the end of '#pragma omp assume' are ignored}}
  {}
}

void dups(void) {
  [[omp::directive(assume no_openmp no_openmp)]] // expected-error {{directive '#pragma omp assume' cannot contain more than one 'no_openmp' clause}}
  {}
}
