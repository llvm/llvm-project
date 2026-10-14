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

// OpenMP 6.0 allows simdizable constructs, which include all loop-transforming
// constructs, to be encountered during execution of a simd region. Earlier
// versions do not.

void use(int);

void tile_in_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+1 {{scan, or atomic directive}}
#pragma omp tile sizes(4)
    for (int i = 0; i < n; ++i)
      use(i);
  }
}

void unroll_in_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+1 {{scan, or atomic directive}}
#pragma omp unroll partial(2)
    for (int i = 0; i < n; ++i)
      use(i);
  }
}

void tile_in_for_simd(int n) {
#pragma omp for simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+1 {{scan, or atomic directive}}
#pragma omp tile sizes(4)
    for (int i = 0; i < n; ++i)
      use(i);
  }
}

// Constructs that are not simdizable are still rejected in OpenMP 6.0.
void parallel_in_simd(int n) {
#pragma omp simd
  for (int j = 0; j < n; ++j) {
    // omp5x-error@+2 {{scan, or atomic directive}}
    // omp60-error@+1 {{atomic, or loop-transforming directive}}
#pragma omp parallel
    use(j);
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
