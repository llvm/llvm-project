// RUN: %check_clang_tidy %s readability-redundant-parentheses %t -- -- -fopenmp=libomp

void linearClause(int *a, int n) {
  int i = 0;
#pragma omp simd linear(i)
  for (int k = 0; k < n; ++k)
    a[k] = i;
}
