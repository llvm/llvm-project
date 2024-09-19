// RUN: %clang_cc1 -emit-llvm-only -verify -fopenmp %s

int main () {
  int r = 0;
#pragma omp scope reduction(+:r) // expected-error {{cannot compile this scope with FE outlining yet}}
  r++;
  return r;
}
