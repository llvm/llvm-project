// RUN: %clang_cc1 -analyze -analyzer-checker=core.builtin.NoReturnFunctions -analyzer-display-progress %s 2>&1 | FileCheck %s

// Do not analyze test1() again because it was inlined
void test1();

void test2() {
  test1();
}

void test1() {
}

// CHECK: analysis-order.c test2
// CHECK-NEXT: analysis-order.c test1
// CHECK-NEXT: analysis-order.c test2
