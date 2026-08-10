// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 -o - %s -Wuninitialized

void foo();

int main(int argc, char **argv) {
#pragma omp parallel sections proc_bind // expected-error {{expected '(' after 'proc_bind'}}
  { foo(); }
#pragma omp parallel sections proc_bind( // omp50-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}} omp51-error {{expected 'master', 'close', 'spread' or 'primary' in OpenMP clause 'proc_bind'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  { foo(); }
#pragma omp parallel sections proc_bind() // omp50-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}} omp51-error {{expected 'master', 'close', 'spread' or 'primary' in OpenMP clause 'proc_bind'}}
  { foo(); }
#pragma omp parallel sections proc_bind(master // expected-error {{expected ')'}} expected-note {{to match this '('}}
  { foo(); }
#pragma omp parallel sections proc_bind(close), proc_bind(spread) // expected-error {{directive '#pragma omp parallel sections' cannot contain more than one 'proc_bind' clause}}
  { foo(); }
#pragma omp parallel sections proc_bind(x) // omp50-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}} omp51-error {{expected 'master', 'close', 'spread' or 'primary' in OpenMP clause 'proc_bind'}}
  { foo(); }

#pragma omp parallel sections proc_bind(master)
  { ++argc; }

#pragma omp parallel sections proc_bind(close)
  {
#pragma omp parallel sections proc_bind(spread)
    { ++argc; }
  }
  return 0;
}
