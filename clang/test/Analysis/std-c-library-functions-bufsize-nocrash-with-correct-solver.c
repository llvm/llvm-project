// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-max-loop 6 \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify

#include "Inputs/std-c-library-functions-POSIX.h"

void clang_analyzer_value(int);
void clang_analyzer_warnIfReached();
void clang_analyzer_printState();

void _add_one_to_index_C(int *indices, int *shape) {
  int k = 1;
  for (; k >= 0; k--)
    if (indices[k] < shape[k])
      indices[k]++;
    else
      indices[k] = 0;
}

void PyObject_CopyData_sptr(char *i, char *j, int *indices, int itemsize,
    int *shape, struct sockaddr *restrict sa) {
  int elements = 1;
  for (int k = 0; k < 2; k++)
    elements += shape[k];

  // no contradiction after 3 iterations when 'elements' could be
  // simplified to 0
  int iterations = 0;
  while (elements--) {
    iterations++;
    _add_one_to_index_C(indices, shape);
    getnameinfo(sa, 10, i, itemsize, i, itemsize, 0); // no crash here
  }

  if (shape[0] == 1 && shape[1] == 1 && indices[0] == 0 && indices[1] == 0) {
    clang_analyzer_value(iterations == 3 && elements == -1);
    // expected-warning@-1{{1}}
  }
}
