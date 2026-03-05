// RUN: %clang -x c++ -std=c++20 -emit-llvm -S %s -o %t.ll
// RUN: %clang -x c++ -fopenmp -std=c++20 -emit-llvm -S %s -o %t.ll

void bar(int N, int (*arr_)[N][N]) {
  auto &ref = arr_;  // reference to pointer-to-VLA
  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
    (*ref)[i][i] = 1;
}
