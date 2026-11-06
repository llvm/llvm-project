// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=61 -fsyntax-only -Wuninitialized -verify %s

extern "C" void body(...);

// expected-note@+1 {{declared here}}
void func(int n) {

  // The depth argument must be a positive integer constant expression.
  // expected-error@+1 {{argument to 'depth' clause must be a strictly positive integer value}}
  #pragma omp flatten depth(0)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);

  // expected-error@+1 {{argument to 'depth' clause must be a strictly positive integer value}}
  #pragma omp flatten depth(-1)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);

  // A non-constant argument is rejected.
  // expected-error@+2 {{expression is not an integral constant expression}}
  // expected-note@+1 {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  #pragma omp flatten depth(n)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);

  // At most one 'depth' clause may appear on the directive.
  // expected-error@+1 {{directive '#pragma omp flatten' cannot contain more than one 'depth' clause}}
  #pragma omp flatten depth(2) depth(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);

  // The depth must be at most the loop nest depth: depth(3) requires three
  // perfectly nested loops, but only two are present here.
  #pragma omp flatten depth(3)
  // expected-error@+1 {{expected 3 for loops after '#pragma omp flatten', but found only 2}}
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);

  // depth(1) flattens a single loop and is well-formed (no diagnostic).
  #pragma omp flatten depth(1)
  for (int i = 0; i < n; ++i)
    body(i);

  // A depth too big for 'unsigned' now caps at UINT_MAX instead of wrapping to 0
  // (which used to crash).
  #pragma omp flatten depth(4294967296)
  // expected-error@+1 {{expected 4294967295 for loops after '#pragma omp flatten', but found only 2}}
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);
}
