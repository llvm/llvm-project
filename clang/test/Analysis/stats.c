// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-stats %s 2>&1 | FileCheck %s

void foo() {
  int x;
}
// CHECK: ... Statistics Collected ...
// CHECK:100 AnalysisConsumer - The % of reachable basic blocks.
// CHECK:The # of times RemoveDeadBindings is called
