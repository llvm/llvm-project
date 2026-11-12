// Check that the CHECK lines are generated for clang-generated functions
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp %s -emit-llvm -o - | FileCheck --check-prefix=OMP %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck --check-prefix=NOOMP %s

const int size = 1024 * 1024 * 32;

double A[size];

void foo(void);

int main(void) {
  int i = 0;

#pragma omp parallel for
  for (i = 0; i < size; ++i) {
    A[i] = 0.0;
  }

  foo();

  return 0;
}

void foo(void) {
  int i = 0;

#pragma omp parallel for
  for (i = 0; i < size; ++i) {
    A[i] = 1.0;
  }
}
