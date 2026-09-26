// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -std=c++11 \
// RUN:   -fsyntax-only %s

// expected-no-diagnostics

void test_variant_clauses_function_scope(int flag) {
  int x = 0;
  int y = 0;

#pragma omp metadirective \
  when(user={condition(flag)}: parallel private(x)) \
  when(user={condition(!flag)}: parallel shared(y))
  {
    x = 1;
    y = 2;
  }
}

void test_nested_metadirective(int flag1, int flag2) {
  int x = 0;

#pragma omp parallel private(x)
  {
#pragma omp metadirective \
  when(user={condition(flag1)}: for) \
  when(user={condition(flag2)}: single private(x))
    for (int i = 0; i < 10; ++i) {
      x = i;
    }
  }
}

void test_multiple_clauses(int flag) {
  int x = 0, y = 0, z = 0;

#pragma omp metadirective \
  when(user={condition(flag)}: parallel private(x) shared(y) firstprivate(z)) \
  otherwise(single private(x))
  {
    x = y + z;
  }
}
