// RUN: %clang_cc1 -std=c++20 -emit-llvm %s -o %t.ll
// RUN: %clang_cc1 -fopenmp -std=c++20 -emit-llvm %s -o %t.ll

void bar(int N, int (*arr_)[N][N]) {
  auto &ref = arr_;  // reference to pointer-to-VLA
  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
    (*ref)[i][i] = 1;
}

// Direct VLA capture.
void direct_vla(int N) {
  int arr[N];

  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
    arr[i] = i;
}

// Lvalue reference to VLA.
void lvalue_ref_vla(int N, int (&arr)[N]) {
  auto &ref = arr;

  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
    ref[i] = i;
}

// Rvalue reference to VLA.
void rvalue_ref_vla(int N, int (&&arr)[N]) {
  auto &&ref = arr;

  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
    ref[i] = i;
}

// Pointer to 2D VLA.
void ptr_to_2d_vla(int N, int M, int (*p)[N][M]) {
  auto &ref = p;

  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
    (*ref)[i][i % M] = i;
}

