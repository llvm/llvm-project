// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -fopenmp -fopenmp-version=60 -fsyntax-only -Wuninitialized -verify %s

void func() {

  // expected-error@+1 {{expected '('}}
  #pragma omp interchange permutation
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp interchange permutation(
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{expected expression}}
  #pragma omp interchange permutation()
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp interchange permutation(1
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp interchange permutation(1,
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{expected expression}}
  #pragma omp interchange permutation(1,)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp interchange permutation(5+
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{expected expression}}
  #pragma omp interchange permutation(5+)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{expected expression}}
  #pragma omp interchange permutation(for)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{permutation index must be at least 1 and at most 1}}
  #pragma omp interchange permutation(0)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{permutation index must be at least 1 and at most 2}}
  #pragma omp interchange permutation(1,3)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{index 1 must appear exactly once in the permutation clause}}
  #pragma omp interchange permutation(1,1)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+4 {{expression is not an integral constant expression}}
  // expected-note@+3 {{read of non-const variable 'a' is not allowed in a constant expression}}
  // expected-note@+1 {{declared here}}
  int a;
  #pragma omp interchange permutation(a)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-warning@+1 {{extra tokens at the end of '#pragma omp interchange' are ignored}}
  #pragma omp interchange foo
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 13; ++j)
      ;

  // expected-error@+1 {{directive '#pragma omp interchange' cannot contain more than one 'permutation' clause}}
  #pragma omp interchange permutation(2,1) permutation(2,1)
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



template <typename T>
static void templated_func() {
  // In a template context, but expression itself not instantiation-dependent

  // expected-error@+1 {{permutation index must be at least 1 and at most 2}}
  #pragma omp interchange permutation(0,1)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j<7; ++j)
      ;

  // expected-error@+1 {{index 1 must appear exactly once in the permutation clause}}
  #pragma omp interchange permutation(1,1)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j<7; ++j)
      ;
}



template <int S>
static void templated_func_value_dependent() {
  // expected-error@+1 {{permutation index must be at least 1 and at most 2}}
  #pragma omp interchange permutation(S,S+1)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j<7; ++j)
      ;

  // expected-error@+1 {{index 1 must appear exactly once in the permutation clause}}
  #pragma omp interchange permutation(S+1,S+1)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j<7; ++j)
      ;
}


template <typename T>
static void templated_func_type_dependent() {
  constexpr T s = 0;

  // expected-error@+1 {{permutation index must be at least 1 and at most 2}}
  #pragma omp interchange permutation(s,s+1)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j<7; ++j)
      ;

  // expected-error@+1 {{index 1 must appear exactly once in the permutation clause}}
  #pragma omp interchange permutation(s+1,s+1)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j<7; ++j)
      ;
}


void template_inst() {
  // expected-note@+1 {{in instantiation of function template specialization 'templated_func<int>' requested here}}
  templated_func<int>();
  // expected-note@+1 {{in instantiation of function template specialization 'templated_func_value_dependent<0>' requested here}}
  templated_func_value_dependent<0>();
  // expected-note@+1 {{in instantiation of function template specialization 'templated_func_type_dependent<int>' requested here}}
  templated_func_type_dependent<int>();
}

