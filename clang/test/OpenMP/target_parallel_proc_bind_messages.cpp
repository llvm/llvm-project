// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s

void foo();

int main(int argc, char **argv) {
  #pragma omp target parallel proc_bind // expected-error {{expected '(' after 'proc_bind'}}
  foo();
  #pragma omp target parallel proc_bind ( // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel proc_bind () // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}}
  foo();
  #pragma omp target parallel proc_bind (master // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel proc_bind (close), proc_bind(spread) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'proc_bind' clause}}
  foo();
  #pragma omp target parallel proc_bind (x) // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}}
  foo();

  #pragma omp target parallel proc_bind(master)
  ++argc;

  #pragma omp target parallel proc_bind(close)
  foo();
  #pragma omp target parallel proc_bind(spread)
  ++argc;
  return 0;
}
