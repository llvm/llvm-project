// RUN: %clang_cc1 -fdouble-square-bracket-attributes -std=c11 -E %s -o - | FileCheck %s

// CHECK: has_fallthrough
#if __has_c_attribute(fallthrough)
  int has_fallthrough();
#endif

// CHECK: does_not_have_selectany
#if !__has_c_attribute(selectany)
  int does_not_have_selectany();
#endif

