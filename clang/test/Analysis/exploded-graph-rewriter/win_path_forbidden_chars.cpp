// FIXME: Figure out how to use %clang_analyze_cc1 with our lit.local.cfg.
// RUN: %clang_cc1 -analyze -triple x86_64-unknown-linux-gnu \
// RUN:                     -analyzer-checker=core \
// RUN:                     -analyzer-dump-egraph=%t.dot %s
// RUN: %exploded_graph_rewriter --verbose %t.dot 2>&1 | FileCheck %s
// REQUIRES: system-windows

// Angle brackets shall not be presented in the field `file`,
// because exploded_graph_rewriter handles it as a file path
// and such symbols are forbidden on Windows platform.

void test() {
  // This produces angle brackets.
  char text[] = __FILE__;
}

// This test is passed if exploded_graph_rewriter handles dot file without errors.
// CHECK: DEBUG:root:Line: digraph "Exploded Graph"
// CHECK: \"file\": \"scratch space\"
