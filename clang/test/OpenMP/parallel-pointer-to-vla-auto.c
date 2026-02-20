// RUN: %clang -std=c23 -emit-llvm -S %s -o %t.ll
// RUN: %clang -fopenmp -std=c23 -emit-llvm -S %s -o %t.ll

// Case 1: C23 auto
void foo1(int N, int (*arr_)[N][N]) {
  auto arr = arr_;
  #pragma omp parallel for
  for (int n = 0; n < N; ++n)
    (*arr)[n][n] = 1;
}

// Case 2: GNU __auto_type
void foo2(int N, int (*arr_)[N][N]) {
  __auto_type arr = arr_;
  #pragma omp parallel for
  for (int n = 0; n < N; ++n)
    (*arr)[n][n] = 1;
}

// Case 3: auto with explicit cast
void foo3(int N, int (*arr_)[N][N]) {
  auto arr = (int (*)[N][N])arr_;
  #pragma omp parallel for
  for (int n = 0; n < N; ++n)
    (*arr)[n][n] = 1;
}
