int f(int coin) {
  if (coin) {
    int *x = 0;
    return *x;
  } else {
    return 0;
  }
}

// RUN: rm -rf %t.output
// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core -analyzer-output html -o %t.output %s
// RUN: cat %t.output/* | FileCheck %s --match-full-lines
// CHECK: var relevant_lines = {"1": {"1": 1, "2": 1, "3": 1, "4": 1}};
