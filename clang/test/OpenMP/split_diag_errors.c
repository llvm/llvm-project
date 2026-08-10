/*
 * Error and delayed-transformation cases for #pragma omp split counts(...).
 */
// 1) Required clause missing: err_omp_required_clause
// RUN: not %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -DTEST_REQUIRED_CLAUSE %s 2>&1 | FileCheck %s --check-prefix=REQ
// 2) counts(negative): non-negative diagnostic
// RUN: not %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -DTEST_NEGATIVE %s 2>&1 | FileCheck %s --check-prefix=NEG
// 3) counts(non-integer): integral type diagnostic
// RUN: not %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -DTEST_FLOAT %s 2>&1 | FileCheck %s --check-prefix=FLOAT
// 6) Loop not transformable (while): must be a for loop
// RUN: not %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -DTEST_WHILE %s 2>&1 | FileCheck %s --check-prefix=WHILE
// Two invalid counts — two diagnostics on the clause
// RUN: not %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -DTEST_DOUBLE_NEGATIVE %s 2>&1 | FileCheck %s --check-prefix=DBLNEG

#ifdef TEST_REQUIRED_CLAUSE
void test_required_clause_missing(void) {
#pragma omp split
  for (int i = 0; i < 10; ++i) {
  }
}
// REQ: error: {{.*}}requires the 'counts' clause
#endif

#ifdef TEST_NEGATIVE
void test_negative_count(void) {
#pragma omp split counts(-1, omp_fill)
  for (int i = 0; i < 10; ++i) {
  }
}
// NEG: error: {{.*}}counts{{.*}}non-negative integer
#endif

#ifdef TEST_FLOAT
void test_float_count(void) {
#pragma omp split counts(2.5, omp_fill)
  for (int i = 0; i < 10; ++i) {
  }
}
// FLOAT: error: {{.*}}integer constant expression must have integer type
#endif

#ifdef TEST_WHILE
void test_while_not_for(void) {
  int i = 0;
#pragma omp split counts(5, omp_fill)
  while (i < 10) {
    ++i;
  }
}
// WHILE: error: {{.*}}must be a for loop
#endif

#ifdef TEST_DOUBLE_NEGATIVE
void test_two_negative_counts(void) {
#pragma omp split counts(-1, -1, omp_fill)
  for (int i = 0; i < 10; ++i) {
  }
}
// DBLNEG: error: {{.*}}counts{{.*}}non-negative integer
// DBLNEG: error: {{.*}}counts{{.*}}non-negative integer
#endif
