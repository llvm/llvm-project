// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection %s 2>&1 | FileCheck %s

// CHECK: TRUE
// CHECK-NOT: garbage
// CHECK-NOT: uninitialized

void clang_analyzer_eval(int);

void test_zero_initialized_new_array() {
  int *p = new int[10]{};
  clang_analyzer_eval(*p == 0);
  delete[] p;
}
