// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -fopenmp -fopenmp-version=60 -fsyntax-only -Wuninitialized -verify %s

void func() {

  // expected-warning@+1 {{extra tokens at the end of '#pragma omp interchange' are ignored}}
  #pragma omp interchange foo
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{unexpected OpenMP clause 'collapse' in directive '#pragma omp interchange'}}
  #pragma omp interchange collapse(2)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  {
    // expected-error@+2 {{expected statement}}
    #pragma omp interchange
  }

  // expected-error@+2 {{statement after '#pragma omp interchange' must be a for loop}}
  #pragma omp interchange
  int b = 0;

  // expected-error@+3 {{statement after '#pragma omp interchange' must be a for loop}}
  #pragma omp interchange
  for (int i = 0; i < 7; ++i)
    ;

  // expected-error@+2 {{statement after '#pragma omp interchange' must be a for loop}}
  #pragma omp interchange
  for (int i = 0; i < 7; ++i) {
    int k = 3;
    for (int j = 0; j < 7; ++j)
      ;
  }

  // expected-error@+3 {{expected loop invariant expression}}
  #pragma omp interchange
  for (int i = 0; i < 7; ++i)
    for (int j = i; j < 7; ++j)
      ;

  // expected-error@+3 {{expected loop invariant expression}}
  #pragma omp interchange
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < i; ++j)
      ;

  // expected-error@+3 {{expected loop invariant expression}}
  #pragma omp interchange
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < i; ++j)
      ;

  // expected-error@+6 {{expected 3 for loops after '#pragma omp for', but found only 2}}
  // expected-note@+1 {{as specified in 'collapse' clause}}
  #pragma omp for collapse(3)
  #pragma omp interchange 
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+2 {{statement after '#pragma omp interchange' must be a for loop}}
  #pragma omp interchange
  #pragma omp for
  for (int i = 0; i < 7; ++i)
    ;

  // expected-error@+3 {{condition of OpenMP for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'j'}}
  #pragma omp interchange 
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j/3<7; ++j)
      ;
}

