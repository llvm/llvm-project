// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.OpaqueSTLFunctionsModeling \
// RUN:   -analyzer-dump-egraph=%t.dot -std=c++11 %s
// RUN: cat %t.dot | FileCheck %s

#include "../Inputs/system-header-simulator-cxx-std-suppression.h"

void testOpaqueSTLTags() {
  int arr[5];
  std::stable_sort(arr, arr + 5);
// CHECK: \"tag\": \"cplusplus.OpaqueSTLFunctionsModeling : Forced Opaque Call\"
  std::inplace_merge(arr, arr + 2, arr + 5);
// CHECK: \"tag\": \"cplusplus.OpaqueSTLFunctionsModeling : Forced Opaque Call\"
}

