// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-dump-egraph=%t.dot %s
// RUN: cat %t.dot | FileCheck %s
// REQUIRES: asserts

struct S {
  ~S();
};

struct T {
  S s;
  T() : s() {}
};

void foo() {
  // Test that dumping symbols conjured on null statements doesn't crash.
  T t;
}

// CHECK: (LC1,S{{[0-9]*}},construct into local variable) T t;\n : &t
// CHECK: (LC2,I{{[0-9]*}},construct into member variable) s : &t-\>s
// CHECK: conj_$5\{int, LC3, no stmt, #1\}

