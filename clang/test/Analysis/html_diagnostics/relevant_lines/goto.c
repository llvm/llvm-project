int goto_test(int input) {
  int *p = 0;
  if (input)
    goto mylabel;
  return 0;
mylabel:
  return *p;
}

// RUN: rm -rf %t.output
// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core -analyzer-output html -o %t.output %s
// RUN: cat %t.output/* | FileCheck %s --match-full-lines
// CHECK: var relevant_lines = {"1": {"1": 1,  "2": 1, "3": 1, "4": 1, "6": 1, "7": 1}};
