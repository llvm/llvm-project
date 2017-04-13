// REQUIRES: asserts
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-stats %s 2>&1 | FileCheck %s

void foo() {
  int x;
}
// CHECK: ... Statistics Collected ...
// CHECK:100 AnalysisConsumer - The % of reachable basic blocks.
// CHECK:The # of times RemoveDeadBindings is called
