// RUN: %clang_analyze_cc1 -verify %s -analyzer-checker=core

// expected-no-diagnostics

typedef unsigned long int A;

extern int fill(A **values, int *nvalues);

void foo() {
  A *values;
  int nvalues;
  fill(&values, &nvalues);

  int i = 1;
  double x, y;

  y = values[i - 1];
  x = values[i];

  if (x <= y) {
  }
}
