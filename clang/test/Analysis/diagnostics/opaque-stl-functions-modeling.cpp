// RUN: %clang_analyze_cc1 -verify %s        \
// RUN:   -analyzer-checker=core,apiModeling \
// RUN:   -analyzer-dump-egraph=%t.dot       \
// RUN:   -analyze-function="test_opaque_handling()"
// RUN: grep 'apiModeling.OpaqueSTLFunctionsModeling' %t.dot | count 3

// expected-no-diagnostics

#include "../Inputs/system-header-simulator-cxx-std-suppression.h"

void test_opaque_handling() {
  int arr[5] = {};
  std::sort(arr, arr + 5); // no-warning
  std::stable_sort(arr, arr + 5); // no-warning
  std::inplace_merge(arr, arr + 2, arr + 5); // no-warning
}
