// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=61 -fsyntax-only -Wall -Wuninitialized -verify=expected,omp61 %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -fsyntax-only -Wall -Wuninitialized -verify=expected %s

extern "C" void body(...);

void func(int n) {

  // The associated statement must be a for loop.
  // expected-error@+2 {{statement after '#pragma omp flatten' must be a for loop}}
  #pragma omp flatten
  ;

  // A non-loop statement is rejected as well.
  // expected-error@+2 {{statement after '#pragma omp flatten' must be a for loop}}
  #pragma omp flatten
  int b = 0;

  // A single loop is not enough: flatten combines two perfectly nested loops,
  // so the body of the outer loop must itself be a for loop.
  #pragma omp flatten
  for (int i = 0; i < 7; ++i)
    // expected-error@+1 {{statement after '#pragma omp flatten' must be a for loop}}
    ;

  // The associated statement of a directive is not a for loop.
  // expected-error@+2 {{statement after '#pragma omp flatten' must be a for loop}}
  #pragma omp flatten
  #pragma omp for
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 7; ++j)
      body(i, j);

  {
    // expected-error@+2 {{expected statement}}
    #pragma omp flatten
  }

  // The loops must be perfectly nested: no code is allowed between them.
  #pragma omp flatten
  for (int i = 0; i < n; ++i) {
    int x = 0;
    // expected-error@-2 {{statement after '#pragma omp flatten' must be a for loop}}
    for (int j = 0; j < n; ++j)
      body(i, j, x);
  }

  // Each affected loop must be in OpenMP canonical form.
  #pragma omp flatten
  for (int i = 0; i < n; ++i)
    // expected-error@+1 {{condition of OpenMP for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'j'}}
    for (int j = 0; j / 3 < n; ++j)
      body(i, j);

  // The spec allows non-rectangular nests, but this implementation does not
  // yet: an inner bound that depends on an outer loop counter is diagnosed.
  #pragma omp flatten
  for (int i = 0; i < n; ++i)
    // expected-error@+1 {{expected loop invariant expression}}
    for (int j = i; j < n; ++j)
      body(i, j);

  // The 'sizes' clause is not allowed on 'flatten'.
  // expected-error@+1 {{unexpected OpenMP clause 'sizes' in directive '#pragma omp flatten'}}
  #pragma omp flatten sizes(2)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 9; ++j)
      body(i, j);

  // The 'permutation' clause is not allowed on 'flatten'.
  // expected-error@+1 {{unexpected OpenMP clause 'permutation' in directive '#pragma omp flatten'}}
  #pragma omp flatten permutation(2, 1)
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 9; ++j)
      body(i, j);

  // Tokens after the directive name are ignored with a warning.
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp flatten' are ignored}}
  #pragma omp flatten foo
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 9; ++j)
      body(i, j);

  // Without a depth clause, only the outermost two loops are flattened. Warn
  // when a deeper perfect nest is left partially unflattened.
  // expected-warning@+2 {{'flatten' without a 'depth' clause only combines 2 loops, but 3 or more loops are perfectly nested}}
  // omp61-note@+1 {{add 'depth(2)' to make it explicit that only the two outermost loops are flattened and to silence this warning}}
  #pragma omp flatten
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        body(i, j, k);

  // Flatten applied to reverse/unroll generated loops: those transforms wrap
  // helper statements around the remaining inner loop, so the nest is no
  // longer perfectly nested from flatten's point of view.
  #pragma omp flatten
  #pragma omp reverse
  for (int i = 0; i < n; ++i)
    // expected-error@+1 {{statement after '#pragma omp flatten' must be a for loop}}
    for (int j = 0; j < n; ++j)
      body(i, j);

  // Stacked flatten: the inner flatten injects helper statements, so the outer
  // flatten does not see a perfect nest of remaining loops.
  #pragma omp flatten
  // expected-warning@+2 {{'flatten' without a 'depth' clause only combines 2 loops, but 3 or more loops are perfectly nested}}
  // omp61-note@+1 {{add 'depth(2)' to make it explicit that only the two outermost loops are flattened and to silence this warning}}
  #pragma omp flatten
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      // expected-error@+1 {{statement after '#pragma omp flatten' must be a for loop}}
      for (int k = 0; k < n; ++k)
        body(i, j, k);

  // Interchange needs two canonical loops; flatten produces one, so the
  // leftover inner loop is no longer a perfect nest for interchange.
  #pragma omp interchange
  #pragma omp flatten
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      // expected-error@+1 {{statement after '#pragma omp interchange' must be a for loop}}
      body(i, j);

  // A compile-time-constant empty inner loop flattens to zero
  // iterations. 
  #pragma omp flatten
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 0; ++j)
      body(i, j);

  // Reverse order: an empty outer loop runs zero times.
  #pragma omp flatten
  for (int i = 0; i < 0; ++i)
    for (int j = 0; j < 3; ++j)
      body(i, j);

  // Unsigned empty inner: skip the signed `N < 0` clamp (would warn under
  // -Wall) and still do not divide by zero.
  #pragma omp flatten
  for (unsigned i = 0; i < 3u; ++i)
    for (unsigned j = 0; j < 0u; ++j)
      body(i, j);
}
