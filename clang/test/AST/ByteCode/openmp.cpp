// RUN: %clang_cc1 -verify=expected,both -fopenmp -fopenmp-version=60 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify=ref,both      -fopenmp -fopenmp-version=60 %s

int test1() {
  int i;
  int &j = i; // both-note {{declared here}}
  float *f;
  // both-note@+2 {{initializer of 'j' is not a constant expression}}
  // both-error@+1 {{integral constant expression}}
  #pragma omp for simd aligned(f:j)
  for (int i = 0; i < 10; ++i);
}

extern int omp_get_thread_num(void);

#define N 64

int test2() {
  int x = 0;
  int result[N] = {0};

  #pragma omp parallel loop num_threads(strict: N) severity(warning) message("msg")
  for (int i = 0; i < N; i++) {
    x = omp_get_thread_num();
    result[i] = i + x;
  }
}


