// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify

// expected-no-diagnostics

#include "Inputs/std-c-library-functions-POSIX.h"

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
  while (elements--) {
    _add_one_to_index_C(indices, shape);
    getnameinfo(sa, 10, i, itemsize, i, itemsize, 0);
  }
}
