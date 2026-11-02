// RUN: %clang_cc1 -fopenmp -fopenmp-version=61 -fsyntax-only -verify %s
// expected-no-diagnostics

// Flatten may create integer comparisons from floating trip counts.
// Guard against that to avoid assertion failures.
void test_floating_init(void) {
#pragma omp flatten
  for (int i = 0; i < 2; ++i)
    for (int j = 0.0; j < 2; ++j)
      ;
}

void test_floating_bound(void) {
#pragma omp flatten
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2.0; ++j)
      ;
}
