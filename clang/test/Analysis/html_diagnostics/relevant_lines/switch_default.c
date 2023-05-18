enum E {
  A, B, C
};

int f(enum E input) {
  int *x = 0;
  switch (input) {
    case A:
      return 1;
    case B:
      return 0;
    default:
      return *x;
  }
}

// RUN: rm -rf %t.output
// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core -analyzer-output html -o %t.output %s
// RUN: cat %t.output/* | FileCheck %s --match-full-lines
// CHECK: var relevant_lines = {"1": {"5": 1, "6": 1, "7": 1, "12": 1, "13": 1}};
