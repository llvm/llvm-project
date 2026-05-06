// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized

// expected-no-diagnostics

#define omp_initial_device -1
#define omp_invalid_device -2

void foo(void) {}

int main(void) {
  int a = 0;

  // Literal values allowed by the spec for the 'device' clause.
  #pragma omp target device(-1)
  foo();
  #pragma omp target device(-2)
  foo();
  #pragma omp target device(0)
  foo();
  #pragma omp target device(1)
  foo();

  // Using the predefined identifiers.
  #pragma omp target device(omp_initial_device)
  foo();
  #pragma omp target device(omp_invalid_device)
  foo();

  // Also accepted on other target-capable directives.
  #pragma omp target data map(to: a) device(omp_initial_device)
  foo();
  #pragma omp target data map(to: a) device(omp_invalid_device)
  foo();

  #pragma omp target enter data map(to: a) device(omp_initial_device)
  #pragma omp target enter data map(to: a) device(omp_invalid_device)

  #pragma omp target exit data map(from: a) device(omp_initial_device)
  #pragma omp target exit data map(from: a) device(omp_invalid_device)

  #pragma omp target update to(a) device(omp_initial_device)
  #pragma omp target update to(a) device(omp_invalid_device)

  // Runtime-determined device numbers still pass the semantic check. The
  // initializer is deliberately -3 (an otherwise-rejected value) to prove
  // that the check does not fold a non-constant expression to its init; a
  // value of -1 or -2 here would silently pass for the wrong reason.
  int dev = -3;
  #pragma omp target device(dev)
  foo();

  return 0;
}
