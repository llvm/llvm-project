// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -fopenmp -fopenmp-version=51 -fsyntax-only -Wuninitialized -verify %s

void func() {

  // expected-error@+1 {{expected '('}}
  #pragma omp tile sizes
    ;

  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp tile  sizes(
    ;

  // expected-error@+1 {{expected expression}}
  #pragma omp tile sizes()
    ;

  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp tile sizes(5
    for (int i = 0; i < 7; ++i);

  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp tile sizes(5,
    ;

  // expected-error@+1 {{expected expression}}
  #pragma omp tile sizes(5,)
    ;

  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp tile sizes(5+
    ;

  // expected-error@+1 {{expected expression}}
  #pragma omp tile sizes(5+)
    ;

  // expected-error@+1 {{expected expression}}
  #pragma omp tile sizes(for)
    ;

  // expected-error@+1 {{argument to 'sizes' clause must be a strictly positive integer value}}
  #pragma omp tile sizes(0)
    ;

  // expected-error@+4 {{expression is not an integral constant expression}}
  // expected-note@+3 {{read of non-const variable 'a' is not allowed in a constant expression}}
  // expected-note@+1 {{declared here}}
  int a;
  #pragma omp tile sizes(a)
    ;

  // expected-warning@+2 {{extra tokens at the end of '#pragma omp tile' are ignored}}
  // expected-error@+1 {{directive '#pragma omp tile' requires the 'sizes' clause}}
  #pragma omp tile foo
    ;

  // expected-error@+1 {{directive '#pragma omp tile' cannot contain more than one 'sizes' clause}}
  #pragma omp tile sizes(5) sizes(5)
  for (int i = 0; i < 7; ++i)
    ;

  // expected-error@+1 {{unexpected OpenMP clause 'collapse' in directive '#pragma omp tile'}}
  #pragma omp tile sizes(5) collapse(2)
  for (int i = 0; i < 7; ++i)
    ;

  {
    // expected-error@+2 {{expected statement}}
    #pragma omp tile sizes(5)
  }

  // expected-error@+2 {{statement after '#pragma omp tile' must be a for loop}}
  #pragma omp tile sizes(5)
  int b = 0;

  // expected-error@+3 {{statement after '#pragma omp tile' must be a for loop}}
  #pragma omp tile sizes(5,5)
  for (int i = 0; i < 7; ++i)
    ;

  // expected-error@+2 {{statement after '#pragma omp tile' must be a for loop}}
  #pragma omp tile sizes(5,5)
  for (int i = 0; i < 7; ++i) {
    int k = 3;
    for (int j = 0; j < 7; ++j)
      ;
  }

  // expected-error@+3 {{expected loop invariant expression}}
  #pragma omp tile sizes(5,5)
  for (int i = 0; i < 7; ++i)
    for (int j = i; j < 7; ++j)
      ;

  // expected-error@+3 {{expected loop invariant expression}}
  #pragma omp tile sizes(5,5)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < i; ++j)
      ;

  // expected-error@+3 {{expected loop invariant expression}}
  #pragma omp tile sizes(5,5)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < i; ++j)
      ;

  // expected-error@+5 {{expected 3 for loops after '#pragma omp for', but found only 2}}
  // expected-note@+1 {{as specified in 'collapse' clause}}
  #pragma omp for collapse(3)
  #pragma omp tile sizes(5)
  for (int i = 0; i < 7; ++i)
    ;

  // expected-error@+2 {{statement after '#pragma omp tile' must be a for loop}}
  #pragma omp tile sizes(5)
  #pragma omp for
  for (int i = 0; i < 7; ++i)
    ;

  // expected-error@+2 {{condition of OpenMP for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'i'}}
  #pragma omp tile sizes(5)
  for (int i = 0; i/3<7; ++i)
    ;
}
