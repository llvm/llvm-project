#include "header.h"

int f(int coin) {
  char *p = 0;
  if (coin) {
    return helper(p, coin);
  }
  return 0;
}

// RUN: rm -rf %t.output
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output html -o %t.output %s
// RUN: cat %t.output/* | FileCheck %s --match-full-lines
// CHECK: var relevant_lines = {"1": {"3": 1, "4": 1, "5": 1, "6": 1}, "4": {"3": 1, "4": 1, "5": 1, "6": 1, "7": 1}};
