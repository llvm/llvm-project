// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo();

int main(int argc, char **argv) {
  int i;
#pragma omp target parallel for proc_bind // expected-error {{expected '(' after 'proc_bind'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for proc_bind( // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for proc_bind() // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for proc_bind(master // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for proc_bind(close), proc_bind(spread) // expected-error {{directive '#pragma omp target parallel for' cannot contain more than one 'proc_bind' clause}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for proc_bind(x) // expected-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}}
  for (i = 0; i < argc; ++i)
    foo();

#pragma omp target parallel for proc_bind(master)
  for (i = 0; i < argc; ++i)
    foo();

#pragma omp parallel proc_bind(close)
#pragma omp target parallel for proc_bind(spread)
  for (i = 0; i < argc; ++i)
    foo();
  return 0;
}
