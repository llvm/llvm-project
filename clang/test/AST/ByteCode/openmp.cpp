// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both -fopenmp %s
// RUN: %clang_cc1 -verify=ref,both -fopenmp %s

int test1() {
  int i;
  int &j = i; // both-note {{declared here}}
  float *f;
  // both-note@+2 {{initializer of 'j' is not a constant expression}}
  // both-error@+1 {{integral constant expression}}
  #pragma omp for simd aligned(f:j)
  for (int i = 0; i < 10; ++i);
}

