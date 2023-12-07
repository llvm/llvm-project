// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -fopenmp -fsyntax-only -Wuninitialized -verify %s

void func(int n) {
  // expected-error@+2 {{statement after '#pragma omp unroll' must be a for loop}}
  #pragma omp unroll
  func(n);

  // expected-error@+2 {{statement after '#pragma omp unroll' must be a for loop}}
  #pragma omp unroll
    ;

  // expected-error@+2 {{the loop condition expression depends on the current loop control variable}}
  #pragma omp unroll
  for (int i = 0; i < 2*(i-4); ++i) {}

  // expected-error@+2 {{condition of OpenMP for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'i'}}
  #pragma omp unroll
  for (int i = 0; i/3 < 7; ++i) {}

  // expected-warning@+1 {{extra tokens at the end of '#pragma omp unroll' are ignored}}
  #pragma omp unroll foo
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp unroll partial(
  for (int i = 0; i < n; ++i) {}
  
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp unroll partial(4
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp unroll partial(4+
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp unroll partial(for)
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'void (int)'}}
  #pragma omp unroll partial(func)
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{expected expression}}
  #pragma omp unroll partial()
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp unroll partial(4,4)
  for (int i = 0; i < n; ++i) {}

  // expected-error@+3 {{expression is not an integral constant expression}} expected-note@+3 {{read of non-const variable 'a' is not allowed in a constant expression}}
  // expected-note@+1 {{declared here}}
  int a;
  #pragma omp unroll partial(a)
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{argument to 'partial' clause must be a strictly positive integer value}} 
  #pragma omp unroll partial(0)
  for (int i = 0; i < n; ++i) {}
    
  // expected-error@+1 {{directive '#pragma omp unroll' cannot contain more than one 'partial' clause}} 
  #pragma omp unroll partial partial
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{directive '#pragma omp unroll' cannot contain more than one 'partial' clause}} 
  #pragma omp unroll partial(4) partial
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{directive '#pragma omp unroll' cannot contain more than one 'full' clause}}
  #pragma omp unroll full full
  for (int i = 0; i < 128; ++i) {}

  // expected-error@+1 {{'full' and 'partial' clause are mutually exclusive and may not appear on the same directive}} expected-note@+1 {{'partial' clause is specified here}}
  #pragma omp unroll partial full
  for (int i = 0; i < n; ++i) {}

  // expected-error@+1 {{'partial' and 'full' clause are mutually exclusive and may not appear on the same directive}} expected-note@+1 {{'full' clause is specified here}}
  #pragma omp unroll full partial
  for (int i = 0; i < n; ++i) {}

  // expected-error@+2 {{loop to be fully unrolled must have a constant trip count}} expected-note@+1 {{'#pragma omp unroll full' directive found here}}
  #pragma omp unroll full
  for (int i = 0; i < n; ++i) {}

  // expected-error@+2 {{statement after '#pragma omp for' must be a for loop}}
  #pragma omp for
  #pragma omp unroll
  for (int i = 0; i < n; ++i) {}

    // expected-error@+2 {{statement after '#pragma omp for' must be a for loop}}
  #pragma omp for
  #pragma omp unroll full
  for (int i = 0; i < 128; ++i) {}

  // expected-error@+2 {{statement after '#pragma omp unroll' must be a for loop}}
  #pragma omp unroll
  #pragma omp unroll
  for (int i = 0; i < n; ++i) {}
  
  // expected-error@+2 {{statement after '#pragma omp tile' must be a for loop}}
  #pragma omp tile sizes(4)
  #pragma omp unroll
  for (int i = 0; i < n; ++i) {}
  
  // expected-error@+4 {{expected 2 for loops after '#pragma omp for', but found only 1}} 
  // expected-note@+1 {{as specified in 'collapse' clause}}
  #pragma omp for collapse(2)
  for (int i = 0; i < n; ++i) {
    #pragma omp unroll full
    for (int j = 0; j < 128; ++j) {}
  }
}


template<typename T, int Factor>
void templated_func(int n) {
  // expected-error@+1 {{argument to 'partial' clause must be a strictly positive integer value}} 
  #pragma omp unroll partial(Factor)
  for (T i = 0; i < n; ++i) {}

  // expected-error@+2 {{loop to be fully unrolled must have a constant trip count}} expected-note@+1 {{'#pragma omp unroll full' directive found here}}
  #pragma omp unroll full
  for (int i = 0; i < n; i-=Factor) {}
}

void template_inst(int n) {
  // expected-note@+1 {{in instantiation of function template specialization 'templated_func<int, -1>' requested here}}
  templated_func<int, -1>(n);
}
