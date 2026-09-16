// RUN: %clang_cc1 -verify=expected,omp60 -fopenmp -fopenmp-version=60 %s
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 %s
// RUN: %clang_cc1 -verify=expected,omp60 -fopenmp-simd -fopenmp-version=60 %s
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp-simd -fopenmp-version=51 %s


void test_induction_basic() {
  int i;
  // omp60-warning@+4{{extra tokens at the end of '#pragma omp parallel for' are ignored}}
  // omp60-error@+3{{unexpected OpenMP clause 'induction' in directive '#pragma omp parallel for'}}
  // omp51-warning@+2{{extra tokens at the end of '#pragma omp parallel for' are ignored}}
  // omp51-error@+1{{unexpected OpenMP clause 'induction' in directive '#pragma omp parallel for'}}
#pragma omp parallel for induction(i)
  for (i = 0; i < 10; ++i)
    ;
}

void test_apply() {
  // omp60-warning@+3{{extra tokens at the end of '#pragma omp tile' are ignored}}
  // omp51-error@+2{{unexpected OpenMP clause 'apply' in directive '#pragma omp tile'}}
  // omp51-warning@+1{{extra tokens at the end of '#pragma omp tile' are ignored}}
#pragma omp tile sizes(10) apply(intratile: unroll)
  for (int i = 0; i < 10; ++i)
    ;
}

void test_empty_apply() {
 // omp60-warning@+3{{extra tokens at the end of '#pragma omp tile' are ignored}}
  // omp51-error@+2{{unexpected OpenMP clause 'apply' in directive '#pragma omp tile'}}
  // omp51-warning@+1{{extra tokens at the end of '#pragma omp tile' are ignored}}
#pragma omp tile sizes(10) apply()
  for (int i = 0; i < 10; ++i)
    ;
}

void test_nested_apply()
{
  // omp60-warning@+4{{extra tokens at the end of '#pragma omp tile' are ignored}}
  //omp51-error@+3{{unexpected OpenMP clause 'apply' in directive '#pragma omp tile'}}
  // omp51-warning@+2{{extra tokens at the end of '#pragma omp tile' are ignored}}
#pragma omp tile sizes(10) \
            apply(intratile: unroll partial(2) apply(reverse))
  for (int i = 0; i < 100; ++i)
    ;
}

void test_induction_with_following_clause() {
  int i;
  // omp60-warning@+4{{extra tokens at the end of '#pragma omp parallel for' are ignored}}
  // omp60-error@+3{{unexpected OpenMP clause 'induction' in directive '#pragma omp parallel for'}}
  // omp51-error@+2{{unexpected OpenMP clause 'induction' in directive '#pragma omp parallel for'}}
  // omp51-warning@+1{{extra tokens at the end of '#pragma omp parallel for' are ignored}}
#pragma omp parallel for induction(i) num_threads(4)
  for (i = 0; i < 10; ++i)
    ;
}

class Point {
  float x, y, m;
  char color;

};

void processPointsInLine() {
  float separation;
  // omp60-error@+4{{unexpected OpenMP clause 'induction' in directive '#pragma omp parallel for'}}
  // omp60-warning@+3{{extra tokens at the end of '#pragma omp parallel for' are ignored}}
  // omp51-error@+2{{unexpected OpenMP clause 'induction' in directive '#pragma omp parallel for'}}
  // omp51-warning@+1{{extra tokens at the end of '#pragma omp parallel for' are ignored}}
#pragma omp parallel for induction(step(Separation))
  for (int i = 0; i < 10; ++i) {
    ;
  }
}

// Make sure test doesn't crash.
void test_tasgraph()
{
  // omp60-error@+2{{unexpected OpenMP directive '#pragma omp taskgraph'}}
  // omp51-error@+1{{unexpected OpenMP directive '#pragma omp taskgraph'}}
#pragma omp taskgraph
  for (int i = 0; i < 10; ++i)
    ;
}

void test_implemented_clause() {
#pragma omp tile sizes(10)
  for (int i = 0; i < 10; ++i)
    ;
}
