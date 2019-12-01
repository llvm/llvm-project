// RUN: %clang_cc1 -fdouble-square-bracket-attributes -std=c11 -E %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c2x -E %s -o - | FileCheck %s

// CHECK: has_fallthrough
#if __has_c_attribute(fallthrough)
  int has_fallthrough();
#endif

// CHECK: does_not_have_selectany
#if !__has_c_attribute(selectany)
  int does_not_have_selectany();
#endif

// CHECK: has_nodiscard_underscore
#if __has_c_attribute(__nodiscard__)
  int has_nodiscard_underscore();
#endif

// CHECK: has_clang_annotate
#if __has_c_attribute(clang::annotate)
  int has_clang_annotate();
#endif
