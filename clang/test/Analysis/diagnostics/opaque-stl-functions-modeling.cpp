// RUN: %clang --analyze -Xclang -verify %s
//
// RUN: %clang --analyze -Xclang -analyzer-dump-egraph=%t.dot -std=c++11 %s
// RUN: cat %t.dot | FileCheck %s

#include "../Inputs/system-header-simulator-cxx-std-suppression.h"

// expected-no-diagnostics

void test_sort() {
  int arr[5];
  std::sort(arr, arr + 5); // no-warning
  // CHECK: \"tag\": \"apiModeling.OpaqueSTLFunctionsModeling
}

void test_stable_sort() {
  int arr[5];
  std::stable_sort(arr, arr + 5); // no-warning
  // CHECK: \"tag\": \"apiModeling.OpaqueSTLFunctionsModeling
}

void test_inplace_merge() {
  int arr[5];
  std::inplace_merge(arr, arr + 2, arr + 5); // no-warning
  // CHECK: \"tag\": \"apiModeling.OpaqueSTLFunctionsModeling
}
