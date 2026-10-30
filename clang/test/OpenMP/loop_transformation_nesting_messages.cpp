// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=51 \
// RUN:   -triple x86_64-unknown-unknown -verify=expected,omp5x %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-unknown -verify=expected,omp5x %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=60 \
// RUN:   -triple x86_64-unknown-unknown -verify=expected,omp60 %s

// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=51 \
// RUN:   -triple x86_64-unknown-unknown -verify=expected,omp5x %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-unknown -verify=expected,omp5x %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=60 \
// RUN:   -triple x86_64-unknown-unknown -verify=expected,omp60 %s

// A loop-transforming construct creates no region of its own, so a construct
// nested in one is still subject to the restrictions of an enclosing simd
// region or of a loop with an order(concurrent) clause. OpenMP 6.0 allows
// loop-transforming constructs in both; earlier versions do not.

void use(int);

void tile_in_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+1 {{atomic, or loop directive}}
#pragma omp tile sizes(4)
    for (int i = 0; i < n; ++i)
      use(i);
  }
}

void unroll_in_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+1 {{atomic, or loop directive}}
#pragma omp unroll partial(2)
    for (int i = 0; i < n; ++i)
      use(i);
  }
}

void tile_in_for_simd(int n) {
#pragma omp for simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+1 {{atomic, or loop directive}}
#pragma omp tile sizes(4)
    for (int i = 0; i < n; ++i)
      use(i);
  }
}

// Constructs that are not simdizable are still rejected in OpenMP 6.0.
void parallel_in_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+2 {{atomic, or loop directive}}
    // omp60-error@+1 {{loop, or loop-transforming directive}}
#pragma omp parallel
    use(j);
  }
}

// A loop-transforming construct creates no region, so a construct nested in
// one is still inside the enclosing simd region.
void parallel_in_tile_in_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+1 {{atomic, or loop directive}}
#pragma omp tile sizes(4)
    for (int i = 0; i < n; ++i) {
      // omp5x-error@+2 {{atomic, or loop directive}}
      // omp60-error@+1 {{loop, or loop-transforming directive}}
#pragma omp parallel
      use(i);
    }
  }
}

// The body of an ordered simd construct is also inside the simd region.
void parallel_in_ordered_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
#pragma omp ordered simd
    {
      // omp5x-error@+2 {{atomic, or loop directive}}
      // omp60-error@+1 {{loop, or loop-transforming directive}}
#pragma omp parallel
      use(j);
    }
  }
}

// A loop-transforming construct associated with the simd construct itself.
void simd_applied_to_tile(int n) {
#pragma omp simd
// omp5x-error@+1 {{atomic, or loop directive}}
#pragma omp tile sizes(4)
  for (int i = 0; i < n; ++i)
    use(i);
}

// A construct nested in a loop-transforming construct is still in the region
// associated with an order(concurrent) clause.
void critical_in_tile_in_order_concurrent(int n) {
#pragma omp parallel for order(concurrent)
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+1 {{construct 'tile' not allowed in a region}}
#pragma omp tile sizes(2)
    for (int i = 0; i < n; ++i) {
      // expected-error@+1 {{construct 'critical' not allowed in a region}}
#pragma omp critical
      use(i);
    }
  }
}

#if _OPENMP >= 202411
// Loop-transforming constructs introduced in OpenMP 6.0.
void omp60_transformations_in_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
#pragma omp reverse
    for (int i = 0; i < n; ++i)
      use(i);
#pragma omp interchange
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k)
        use(i + k);
#pragma omp stripe sizes(4)
    for (int i = 0; i < n; ++i)
      use(i);
#pragma omp fuse
    {
      for (int i = 0; i < n; ++i)
        use(i);
      for (int k = 0; k < n; ++k)
        use(k);
    }
#pragma omp split counts(2, omp_fill)
    for (int i = 0; i < n; ++i)
      use(i);
  }
}
#endif
