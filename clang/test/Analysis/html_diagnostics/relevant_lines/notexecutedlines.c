int f(void) {
  int zzz = 200;
  zzz += 100;
  return 0;
}

// Show line with the warning even if it wasn't executed (e.g. warning given
// by path-insensitive analysis).
// RUN: rm -rf %t.output
// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core,deadcode -analyzer-output html -o %t.output %s
// RUN: cat %t.output/* | FileCheck %s --match-full-lines
// CHECK: var relevant_lines = {"1": {"3": 1}};
