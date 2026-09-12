// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -std=c++11 \
// RUN:   -fsyntax-only %s

// expected-no-diagnostics

void test_runtime_condition(int flag) {
#pragma omp metadirective			\
  when(user={condition(flag)}: parallel)	\
  otherwise(single)
  {
    int x = 0;
  }
}

void test_two_conditions(int flag1, int flag2) {
#pragma omp metadirective			\
  when(user={condition(flag1)}: parallel)	\
  when(user={condition(flag2)}: single)		\
  otherwise()
  {
    int y = 1;
  }
}

void test_complex_condition(int a, int b) {
#pragma omp metadirective			\
  when(user={condition(a > b)}: parallel)	\
  otherwise(single)
  {
    int z = 2;
  }
}

void test_logical_condition(bool flag1, bool flag2) {
#pragma omp metadirective				\
  when(user={condition(flag1 && flag2)}: parallel)	\
  otherwise()
  {
    int w = 3;
  }
}

void test_multiple_variants(int flag1, int flag2, int flag3) {
#pragma omp metadirective			\
  when(user={condition(flag1)}: parallel)	\
  when(user={condition(flag2)}: single)		\
  when(user={condition(flag3)}: teams)		\
  otherwise()
  {
    int v = 4;
  }
}

void test_otherwise_only() {
#pragma omp metadirective otherwise(parallel)
  {
    int u = 5;
  }
}

void test_different_directives(int flag) {
#pragma omp metadirective		\
  when(user={condition(flag)}: teams)	\
  otherwise(task)
  {
    int t = 6;
  }
}

void test_nested_statement(int flag) {
#pragma omp metadirective			\
  when(user={condition(flag)}: parallel)	\
  otherwise()
  {
    for (int i = 0; i < 10; ++i) {
      int s = i;
    }
  }
}

