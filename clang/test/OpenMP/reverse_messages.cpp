// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -fsyntax-only -Wuninitialized -verify %s

void func() {

  // expected-error@+2 {{statement after '#pragma omp reverse' must be a for loop}}
  #pragma omp reverse
    ;

  // expected-error@+2 {{statement after '#pragma omp reverse' must be a for loop}}
  #pragma omp reverse
  int b = 0;

  // expected-error@+2 {{statement after '#pragma omp reverse' must be a for loop}}
  #pragma omp reverse
  #pragma omp for
  for (int i = 0; i < 7; ++i)
    ;

  {
    // expected-error@+2 {{expected statement}}
    #pragma omp reverse
  }

  // expected-error@+2 {{condition of OpenMP for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'i'}}
  #pragma omp reverse
  for (int i = 0; i/3<7; ++i)
    ;

  // expected-error@+1 {{unexpected OpenMP clause 'sizes' in directive '#pragma omp reverse'}}
  #pragma omp reverse sizes(5)
  for (int i = 0; i < 7; ++i)
    ;

  // expected-warning@+1 {{extra tokens at the end of '#pragma omp reverse' are ignored}}
  #pragma omp reverse foo
  for (int i = 0; i < 7; ++i)
    ;

}

