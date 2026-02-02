// RUN: %clang --analyze -Xclang -analyzer-dump-egraph=%t.dot -std=c++11 -Xclang -verify %s
// RUN: cat %t.dot | FileCheck %s

#include "../Inputs/system-header-simulator-cxx-std-suppression.h"

// expected-no-diagnostics

void test_opaque_handling() {
  int arr[5] = {};
  std::sort(arr, arr + 5); // no-warning
// CHECK: \"tag\": \"apiModeling.OpaqueSTLFunctionsModeling\"
  std::stable_sort(arr, arr + 5); // no-warning
// CHECK: \"tag\": \"apiModeling.OpaqueSTLFunctionsModeling\"
  std::inplace_merge(arr, arr + 2, arr + 5); // no-warning
// CHECK: \"tag\": \"apiModeling.OpaqueSTLFunctionsModeling\"
}
