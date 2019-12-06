// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo();

int main(int argc, char **argv) {
#pragma omp parallel master proc_bind // expected-error {{expected '(' after 'proc_bind'}}
  { foo(); }
#pragma omp parallel master proc_bind( // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  { foo(); }
#pragma omp parallel master proc_bind() // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}}
  { foo(); }
#pragma omp parallel master proc_bind(master // expected-error {{expected ')'}} expected-note {{to match this '('}}
  { foo(); }
#pragma omp parallel master proc_bind(close), proc_bind(spread) // expected-error {{directive '#pragma omp parallel master' cannot contain more than one 'proc_bind' clause}}
  { foo(); }
#pragma omp parallel master proc_bind(x) // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}}
  { foo(); }

#pragma omp parallel master proc_bind(master)
  { ++argc; }

#pragma omp parallel master proc_bind(close)
  {
#pragma omp parallel master proc_bind(spread)
    { ++argc; }
  }
  return 0;
}
