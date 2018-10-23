// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-dump-egraph=%t.dot %s
// RUN: cat %t.dot | FileCheck %s
// REQUIRES: asserts


struct S {
  ~S();
};

void foo() {
  // Test that dumping symbols conjured on null statements doesn't crash.
  S s;
}

// CHECK: conj_$0\{int, LC1, no stmt, #1\}
