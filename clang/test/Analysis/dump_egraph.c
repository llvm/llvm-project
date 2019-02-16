// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-dump-egraph=%t.dot %s
// RUN: cat %t.dot | FileCheck %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-dump-egraph=%t.dot -trim-egraph %s
// REQUIRES: asserts

int getJ();

int foo() {
  int *x = 0, *y = 0;
  return *x + *y;
}

// CHECK: digraph "Exploded Graph" {
// CHECK: Edge: (B2, B1)
// CHECK: Block Entrance: B1
// CHECK: Bug report attached
